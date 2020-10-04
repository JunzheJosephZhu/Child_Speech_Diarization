import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from base_trainer import BaseTrainer
from utils import compute_STOI, compute_PESQ
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
            metric,
            train_dataloader,
            validation_dataloader,
    ):
        super(Trainer, self).__init__(config, resume, model, loss_function, optimizer, scheduler)
        self.train_data_loader = train_dataloader
        self.validation_data_loader = validation_dataloader
        self.metric = metric

    def _train_epoch(self, epoch):
        loss_total = 0.0
        error_total, base_total = 0.0, 0.0

        for i, (audio, target) in enumerate(tqdm(self.train_data_loader)):
            audio = audio.to(self.gpu_ids[0])
            target = target.to(self.gpu_ids[0])

            self.optimizer.zero_grad()
            output = self.model(audio)
            loss = self.loss_function(output, target)
            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()
            error, base = self.metric(output, target)
            error_total += error
            base_total += base

        dl_len = len(self.train_data_loader)
        self.writer.add_scalar(f"Train/Loss", loss_total / dl_len, epoch)
        self.writer.add_scalar(f"Train/Error", error_total / base_total, epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        loss_total = 0.0
        error_total, base_total = 0.0, 0.0

        for i, (audio, target) in enumerate(tqdm(self.validation_data_loader)):
            audio = audio.to(self.gpu_ids[0])
            target = target.to(self.gpu_ids[0])

            output = self.model(audio)

            loss = self.loss_function(output, target)
            loss_total += loss.item()
            error, base = self.metric(output, target)
            error_total += error
            base_total += base

        # dl_len = len(self.train_data_loader)
        # self.writer.add_scalar(f"Val/Loss", loss_total / dl_len, epoch)
        self.writer.add_scalar(f"Val/Error", error_total / base_total, epoch)
        return error_total / base_total
