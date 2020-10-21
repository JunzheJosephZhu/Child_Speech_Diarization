import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from metric import AudioAnalysis

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
            lr_override=False,
            test=False,
    ):
        super(Trainer, self).__init__(config, resume, model, loss_function, optimizer, scheduler, lr_override, test)
        self.train_data_loader = train_dataloader
        self.validation_data_loader = validation_dataloader
        self.metrics = metrics
        self.score = config["score"]
        assert self.score in self.metrics

    def _train_epoch(self, epoch):
        loss_total = 0.0
        error_total, base_total = defaultdict(float), defaultdict(float)

        pbar = tqdm(self.train_data_loader)
        for i, (audio, mask, target) in enumerate(pbar):
            audio = audio.to(self.gpu_ids[0])
            target = target.to(self.gpu_ids[0])
            mask = mask.to(self.gpu_ids[0]).bool()

            self.optimizer.zero_grad()
            output = self.model(audio, mask)
            loss = self.loss_function(output, target)
            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()

            des = f"Loss: {loss.item(): .3f}"
            for name, metric_fn in self.metrics.items():
                error, base = metric_fn(output, target)
                error_total[name] += error
                base_total[name] += base
                if type(error_total[name]) == float:
                    des += f" || {name}: {error / base: .2f}"
                else:
                    des += f" | {name}: {np.array2string(error / base, precision=2)}"
            pbar.set_description(des)

        dl_len = len(self.train_data_loader)
        self.writer.add_scalar(f"Train/Loss", loss_total / dl_len, epoch)
        for name in self.metrics:
            if type(error_total[name]) == float:
                self.writer.add_scalar(f"Train/{name}", error_total[name] / base_total[name], epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        loss_total = 0.0
        error_total, base_total = defaultdict(float), defaultdict(float)
        audio_analysis = AudioAnalysis()

        for i, (audio, mask, target) in enumerate(tqdm(self.validation_data_loader)):
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

            if self.test:
                if len(target.size()) > 1: # if not in MIL mode
                    audio_analysis(audio, output, target)

        if not self.test:
            for name in self.metrics:
                if type(error_total[name]) == float:
                    self.writer.add_scalar(f"Val/{name}", error_total[name] / base_total[name], epoch)
            return error_total[self.score] / base_total[self.score]
        else:
            audio_analysis.save()
            return {name: error_total[name] / base_total[name] for name in error_total}

