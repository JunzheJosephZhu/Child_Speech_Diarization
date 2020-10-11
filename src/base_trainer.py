import time
from pathlib import Path

import json5
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from utils import prepare_empty_dir, ExecutionTime, writer

class BaseTrainer:
    def __init__(self,
                 config,
                 resume: bool,
                 model,
                 loss_function,
                 optimizer,
                 scheduler,
                 test):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.test = test

        self.gpu_ids = config['gpu_ids']
        self.model = model.to(self.gpu_ids[0])
        if len(self.gpu_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)
        # Trainer
        self.epochs = config["trainer"]["epochs"]

        # The following args is not in the config file. We will update it if the resume is True in later.
        self.start_epoch = 0
        self.best_score = np.inf
        self.root_dir = Path(config["root"]).expanduser().absolute() / "experiments" / config["experiment_name"]
        self.checkpoints_dir = self.root_dir / "checkpoints"
        self.logs_dir = self.root_dir / "logs"
        prepare_empty_dir([self.checkpoints_dir, self.logs_dir], resume=resume)

        if not self.test:
            self.writer = writer(self.logs_dir.as_posix())
            self.writer.add_text(
                tag="Configuration",
                text_string=f"<pre>  \n{json5.dumps(config, indent=4, sort_keys=False)}  \n</pre>",
                global_step=1
            )
            print("Configurations are as follows: ")
            print(json5.dumps(config, indent=2, sort_keys=False))

            with open((self.root_dir / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}.json").as_posix(), "w") as handle:
                json5.dump(config, handle, indent=2, sort_keys=False)

        if resume or self.test: self._resume_checkpoint()

        self._print_networks([self.model])

    def _resume_checkpoint(self):
        """Resume experiment from the latest checkpoint.
        Notes:
            To be careful at the loading. if the model is an instance of DataParallel, we need to set model.module.*
        """
        if not self.test:
            latest_model_path = self.checkpoints_dir.expanduser().absolute() / "latest_model.pth"
        else:
            latest_model_path = self.checkpoints_dir.expanduser().absolute() / "best_model.pth"
        assert latest_model_path.exists(), f"{latest_model_path} does not exist, can not load latest checkpoint."

        checkpoint = torch.load(latest_model_path.as_posix(), map_location=torch.device(self.gpu_ids[0]))

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_score = checkpoint["best_score"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])

        print(f"Model checkpoint loaded. Training will begin in {self.start_epoch} epoch.")

    def _save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint to <root_dir>/checkpoints directory.
        It contains:
            - current epoch
            - best score in history
            - optimizer parameters
            - model parameters
        Args:
            is_best(bool): if current checkpoint got the best score, it also will be saved in <root_dir>/checkpoints/best_model.tar.
        """
        print(f"\t Saving {epoch} epoch model checkpoint...")

        # Construct checkpoint tar package
        state_dict = {
            "epoch": epoch,
            "best_score": self.best_score,
            "optimizer": self.optimizer.state_dict()
        }

        if isinstance(self.model, torch.nn.DataParallel):  # Parallel
            state_dict["model"] = self.model.module.cpu().state_dict()
        else:
            state_dict["model"] = self.model.cpu().state_dict()

        """
        Notes:
            - latest_model.tar:
                Contains all checkpoint information, including optimizer parameters, model parameters, etc. New checkpoint will overwrite old one.
            - model_<epoch>.pth: 
                The parameters of the model. Follow-up we can specify epoch to inference.
            - best_model.tar:
                Like latest_model, but only saved when <is_best> is True.
        """
        torch.save(state_dict, (self.checkpoints_dir / "latest_model.pth").as_posix())
        torch.save(state_dict["model"], (self.checkpoints_dir / f"model_{str(epoch).zfill(4)}.pth").as_posix())
        if is_best:
            print(f"\t Found best score in {epoch} epoch, saving...")
            torch.save(state_dict, (self.checkpoints_dir / "best_model.pth").as_posix())

        # Use model.cpu() or model.to("cpu") will migrate the model to CPU, at which point we need remigrate model back.
        # No matter tensor.cuda() or tensor.to("cuda"), if tensor in CPU, the tensor will not be migrated to GPU, but the model will.
        self.model.to(self.gpu_ids[0])

    def _is_best(self, score):
        """Check if the current model is the best.
        """
        if score <= self.best_score:
            self.best_score = score
            return True
        else:
            return False

    @staticmethod
    def _print_networks(nets: list):
        print(f"This project contains {len(nets)} networks, the number of the parameters: ")
        params_of_all_networks = 0
        for i, net in enumerate(nets, start=1):
            params_of_network = 0
            for param in net.parameters():
                params_of_network += param.numel()

            print(f"\tNetwork {i}: {params_of_network / 1e6} million.")
            params_of_all_networks += params_of_network

        print(f"The amount of parameters is {params_of_all_networks / 1e6} million.")

    def _set_models_to_train_mode(self):
        self.model.train()

    def _set_models_to_eval_mode(self):
        self.model.eval()

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            print(f"============== {epoch} epoch ==============")
            print("[0 seconds] Begin training...")
            timer = ExecutionTime()

            self._set_models_to_train_mode()
            self._train_epoch(epoch)
            self._save_checkpoint(epoch)

            self._set_models_to_eval_mode()
            score = self._validation_epoch(epoch)
            if self._is_best(score):
                self._save_checkpoint(epoch, is_best=True)

            self.scheduler.step()
            print(f"[{timer.duration()} seconds] End epoch {epoch}, best_score {self.best_score}.")

    def run_test(self):
        print(f"============== test time ==============\n{self.root_dir}")

        self._set_models_to_eval_mode()
        score = self._validation_epoch(self.start_epoch)
        print(f"loaded model at epoch {self.start_epoch}, \nval error {self.best_score}")
        print(f"test error rate is {score}")

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _validation_epoch(self, epoch):
        raise NotImplementedError