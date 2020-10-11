import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from base_trainer import BaseTrainer
from utils import compute_STOI, compute_PESQ
from collections import defaultdict
plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            model,
            loss_function,
            optimizer,
            scheduler,
            metrics,
            train_dataloader,
            validation_dataloader,
            test=False,
    ):
        super(Trainer, self).__init__(config, resume, model, loss_function, optimizer, scheduler, test)
        self.train_data_loader = train_dataloader
        self.validation_data_loader = validation_dataloader
        self.metrics = metrics
        self.score = config["score"]

    def _train_epoch(self, epoch):
        loss_total = 0.0
        error_total, base_total = defaultdict(int), defaultdict(int)

        for i, (audio, target, mask) in enumerate(tqdm(self.train_data_loader)):
            audio = audio.to(self.gpu_ids[0])
            target = target.to(self.gpu_ids[0])
            mask = mask.to(self.gpu_ids[0])

            self.optimizer.zero_grad()
            output = self.model(audio, mask)
            loss = self.loss_function(output, target)
            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()
            for name, metric_fn in self.metrics.items():
                error, base = metric_fn(output, target)
                error_total[name] += error
                base_total[name] += base
        
        dl_len = len(self.train_data_loader)
        self.writer.add_scalar(f"Train/Loss", loss_total / dl_len, epoch)
        for name in self.metrics.items:
            self.writer.add_scalar(f"Train/{name}", error_total[name] / base_total[name], epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        loss_total = 0.0
        error_total, base_total = defaultdict(int), defaultdict(int)

        for i, (audio, target, mask) in enumerate(tqdm(self.validation_data_loader)):
            audio = audio.to(self.gpu_ids[0])
            target = target.to(self.gpu_ids[0])
            mask = mask.to(self.gpu_ids[0])

            output = self.model(audio, mask)
            loss = self.loss_function(output, target)
            loss_total += loss.item()
            for name, metric_fn in self.metrics.items():
                error, base = metric_fn(output, target)
                error_total[name] += error
                base_total[name] += base

        if not self.test:
            for name in self.metrics.items:
                self.writer.add_scalar(f"Train/{name}", error_total[name] / base_total[name], epoch)
        return error_total[self.score] / base_total[self.score]

    
