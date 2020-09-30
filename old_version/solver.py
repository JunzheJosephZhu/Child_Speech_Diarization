import os
import time
import numpy as np
import torch
from loss import CCE_acc as loss_func

from torch.utils.tensorboard import SummaryWriter

class Solver(object):
    def __init__(self, data, model, optimizer, scheduler, epochs, save_folder, checkpoint, continue_from, model_path, print_freq, 
                early_stop, max_norm, lr, lr_override, log_dir, config):
        '''
            config: config name
            model_path: best model path
        '''
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Training config
        self.epochs = epochs
        self.early_stop = early_stop
        self.max_norm = max_norm
        # save and load model
        self.save_folder = save_folder
        self.checkpoint = checkpoint
        self.continue_from = continue_from
        self.model_path = model_path
        self.config = config
        # logging
        self.print_freq = print_freq
        # visualizing loss using visdom
        self.tr_acc = torch.Tensor(self.epochs)
        self.cv_acc = torch.Tensor(self.epochs)

        self._reset()

        self.writer = SummaryWriter(log_dir)
        # learning rate override
        if lr_override:
            optim_state = self.optimizer.state_dict()
            optim_state['param_groups'][0]['lr'] = lr
            self.optimizer.load_state_dict(optim_state)
            print('Learning rate adjusted to: {lr:.6f}'.format(
                lr=optim_state['param_groups'][0]['lr']))

    def _reset(self):
        # Reset
        load = self.continue_from and os.path.exists(self.continue_from)
        self.start_epoch = 0
        self.val_no_impv = 0
        self.prev_acc = float("0.0")
        self.best_acc = float("0.0")
        if load: # if the checkpoint model exists
            print('Loading checkpoint model %s' % self.continue_from)
            package = torch.load(self.continue_from)
            self.model.module.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start_epoch = int(package.get('epoch', 1))
            self.tr_acc[:self.start_epoch] = package['tr_acc'][:self.start_epoch]
            self.cv_acc[:self.start_epoch] = package['cv_acc'][:self.start_epoch]
            print('best acc so far', max(package['cv_acc'][:self.start_epoch]))
            self.val_no_impv = package.get('val_no_impv', 0)
            if 'random_state' in package:
                torch.set_rng_state(package['random_state'])
            
            self.prev_acc = self.cv_acc[self.start_epoch-1]
            self.best_acc = min(self.cv_acc[:self.start_epoch])

        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.halving = False

    def train(self):
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            print("Training...")
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss, tr_avg_acc = self._run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)
            self.writer.add_scalar('Loss/per_epoch_tr', tr_avg_loss, epoch)
            self.writer.add_scalar('Accuracy/per_epoch_tr', tr_avg_acc, epoch)

            # Cross validation
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            val_avg_loss, val_avg_acc = self._run_one_epoch(epoch, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, val_avg_loss))
            print('-' * 85)
            self.writer.add_scalar('Loss/per_epoch_cv', val_avg_loss, epoch)
            self.writer.add_scalar('Accuracy/per_epoch_cv', val_avg_acc, epoch)

            # Adjust learning rate (halving)
            if val_avg_acc <= self.prev_acc:
                self.val_no_impv += 1
                if self.val_no_impv >= 10 and self.early_stop:
                    print("No improvement for 10 epochs, early stopping.")
                    break
            else:
                self.val_no_impv = 0

            self.scheduler.step()
            self.prev_acc = val_avg_acc

            # Save the best model
            self.tr_acc[epoch] = tr_avg_acc
            self.cv_acc[epoch] = val_avg_acc
            package = self.model.module.serialize(self.model.module,
                                                       self.optimizer, epoch + 1,
                                                       tr_acc=self.tr_acc,
                                                       cv_acc=self.cv_acc,
                                                       val_no_impv = self.val_no_impv,
                                                       random_state=torch.get_rng_state())
            if val_avg_acc > self.best_acc:
                self.best_acc = val_avg_acc
                file_path = os.path.join(self.save_folder, self.model_path)
                torch.save(package, file_path)
                print("Find better validated model, saving to %s" % file_path)

            # Save model each epoch, nd make a copy at last.pth
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save(package, file_path)
                print('Saving checkpoint model to %s' % file_path)

            # update config#.pth
            torch.save(package, os.path.join(self.save_folder, self.config + '.pth'))



    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0
        total_accuracy = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader
        empty = 0 # empty examples
        for i, (signal, target, mask) in enumerate(data_loader):
            if torch.sum(mask) == 0:
                empty += 1
                continue
            if not cross_valid:
                logits = self.model(signal)
            else:
                with torch.no_grad():
                    logits = self.model(signal)
            loss, accuracy = loss_func(logits, target, mask)
            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                            self.max_norm)
                self.optimizer.step()
            total_loss += loss.item()
            total_accuracy += accuracy.item()

            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} |  Average accuracy {4:.3f} | {5:.1f} ms/batch'.format(
                          epoch + 1, i + 1, total_loss / (i + 1 - empty),
                          loss.item(), total_accuracy / (i + 1 - empty), 
                          1000 * (time.time() - start) / (i + 1 - empty)),
                      flush=True)
        return total_loss / (i + 1 - empty), total_accuracy / (i + 1 - empty)
