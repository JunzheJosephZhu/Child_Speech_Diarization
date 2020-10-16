import torch
import numpy as np
EPS = 1e-6

def get_zeros(tensor):
    zeros = torch.zeros(tensor.size(), dtype=tensor.dtype)
    zeros = zeros.cuda(tensor.get_device()) if tensor.is_cuda else zeros
    return zeros

class MedianFilter():
    def __init__(self, median_filter):
        self.pad = torch.nn.ReflectionPad1d(median_filter//2)
        self.median_filter = median_filter
    
    def __call__(self, logits_batch):
        '''
            logits_batch: [B, C, T]
        '''
        # [B, C, L + median_filter // 2 * 2]
        logits_batch = self.pad(logits_batch)
        # [B, C, L, median_filter]
        logits_batch = logits_batch.unfold(-1, self.median_filter, 1)
        # [B, C, L]
        logits_batch, _ = torch.median(logits_batch, dim=-1)
        return logits_batch

class DER():
    def __init__(self, threshold=0.5, median_filter=5): # each frame is 256ms, so median filter=256*5=1280ms
        self.threshold = threshold
        self.filter = MedianFilter(median_filter)

    def __call__(self, logits_batch, label_batch):
        '''
            logits_batch: torch.Tensor, [B, C, L]
            label_batch: torch.Tensor, [B, C, L], value = [0, 1]
        '''
        B, C, T = logits_batch.size()
        error, T_scored = 0.0, 0.0
        logits_batch = self.filter(logits_batch)

        for idx in range(B):
            # [L, C], [L, C]
            logits, label = logits_batch[idx].T, label_batch[idx].T
            decisions = (torch.sigmoid(logits) > self.threshold).long()
            label = label.long()
            # [L], total number of speakers in each frame
            n_ref = torch.sum(label, axis=-1)
            n_sys = torch.sum(decisions, axis=-1)
            res = {}
            res['speech_scored'] = torch.sum(n_ref > 0)
            res['speech_miss'] = torch.sum(
                torch.logical_and(n_ref > 0, n_sys == 0))
            res['speech_falarm'] = torch.sum(
                torch.logical_and(n_ref == 0, n_sys > 0))
            res['speaker_scored'] = torch.sum(n_ref)
            res['speaker_miss'] = torch.sum(torch.max(n_ref - n_sys, get_zeros(n_ref)))
            res['speaker_falarm'] = torch.sum(torch.max(n_sys - n_ref, get_zeros(n_ref)))
            # [L], total of agreed speakers in each frame
            n_map = torch.sum(
                torch.logical_and(label == 1, decisions == 1),
                axis=-1)
            # torch.minimum(n_ref, n_sys): lesser of total speakers in each frame
            # 'speaker_error': some speaker is not assigned. 
            # Maximum - minimum is fused into miss/falalarm, while minimum - agreed is fused into speaker_error
            res['speaker_error'] = torch.sum(torch.min(n_ref, n_sys) - n_map)
            res['correct'] = torch.sum(label == decisions).float() / label.shape[1]
            res['diarization_error'] = (
                res['speaker_miss'] + res['speaker_falarm'] + res['speaker_error'])
            res['frames'] = len(label)
            error += res['diarization_error']
            T_scored += res['speaker_scored']
        return error.item(), T_scored.item()

class ERR():
    def __call__(self, logits_batch, label_batch):
        '''
            logits_batch: torch.Tensor, [B, C]
            label_batch: torch.Tensor, [B]
        '''
        # [B, L]
        assert logits_batch.size(0) == label_batch.size(0)
        preds = logits_batch.argmax(1)
        return torch.sum(preds != label_batch).item(), label_batch.size(0)

class Class_ERR():
    def __call__(self, logits_batch, label_batch):
        '''
            logits_batch: torch.Tensor, [B, C]
            label_batch: torch.Tensor, [B]
        '''
        # [B, L]
        assert logits_batch.size(0) == label_batch.size(0)
        num_cls = logits_batch.size(1)
        preds = logits_batch.argmax(1).detach().cpu().numpy()
        labels = label_batch.detach().cpu().numpy()

        error, base = np.zeros(num_cls), np.zeros(num_cls)
        for i in range(num_cls):
            error[i] = np.sum((preds != labels)[labels == i])
            base[i] = np.sum(labels == i)
        return error, base

class Frame_ERR():
    def __init__(self, threshold=0.5, median_filter=5): # each frame is 256ms, so median filter=256*5=1280ms
        self.threshold = threshold
        self.filter = MedianFilter(median_filter)

    def __call__(self, logits_batch, label_batch):
        '''
            logits_batch: torch.Tensor, [B, C, L]
            label_batch: torch.Tensor, [B, C, L]
        '''
        assert logits_batch.size() == label_batch.size()
        logits_batch = self.filter(logits_batch)
        decisions = torch.sigmoid(logits_batch) > self.threshold
        # [B, L]
        error_frames = (decisions != label_batch).any(dim=1)
        return torch.sum(error_frames).item(), label_batch.size(0) * label_batch.size(2)

class DER_Tier():
    def __init__(self, threshold=0.5, median_filter=5): # each frame is 256ms, so median filter=256*5=1280ms
        self.threshold = threshold
        self.filter = MedianFilter(median_filter)

    def __call__(self, logits_batch, label_batch):
        '''
            logits_batch: torch.Tensor, [B, C, L]
            label_batch: torch.Tensor, [B, C, L]
        '''
        assert logits_batch.size() == label_batch.size()
        logits_batch = self.filter(logits_batch)
        decisions = torch.sigmoid(logits_batch) > self.threshold
        # [B, L]
        error_tiers = decisions != label_batch
        return error_tiers.sum(2).sum(0).detach().cpu().numpy(), label_batch.sum(2).sum(0).detach().cpu().numpy()

class ERR_Tier():
    def __init__(self, threshold=0.5, median_filter=5): # each frame is 256ms, so median filter=256*5=1280ms
        self.threshold = threshold
        self.filter = MedianFilter(median_filter)

    def __call__(self, logits_batch, label_batch):
        '''
            logits_batch: torch.Tensor, [B, C, L]
            label_batch: torch.Tensor, [B, C, L]
        '''
        assert logits_batch.size() == label_batch.size()
        logits_batch = self.filter(logits_batch)
        decisions = torch.sigmoid(logits_batch) > self.threshold
        # [B, L]
        error_tiers = decisions != label_batch
        return error_tiers.sum(2).sum(0).detach().cpu().numpy(), label_batch.size(0) * label_batch.size(2)

class Frame_Tier_ERR():
    def __init__(self, threshold=0.5, median_filter=5): # each frame is 256ms, so median filter=256*5=1280ms
        self.threshold = threshold
        self.filter = MedianFilter(median_filter)

    def __call__(self, logits_batch, label_batch):
        '''
            logits_batch: torch.Tensor, [B, C, L]
            label_batch: torch.Tensor, [B, C, L]
        '''
        assert logits_batch.size() == label_batch.size()
        logits_batch = self.filter(logits_batch)
        decisions = torch.sigmoid(logits_batch) > self.threshold
        # [B, L]
        error_frames = decisions != label_batch
        return torch.sum(error_frames).item(), torch.numel(label_batch)



if __name__ == "__main__":
    a = torch.randn(20)
    import os
    import json
    from pathlib import Path
    from data import LENADataSet
    root = "~/Desktop/MaxMin_Pytorch"
    config_filename = "configs/AE_RNN.json"
    with open(os.path.join(Path(root).expanduser(), config_filename)) as file:
        config = json.load(file)
    dataset_config = config["dataset"]
    trainset = LENADataSet(dataset_config["train"], **dataset_config["args"])
    testset = LENADataSet(dataset_config["val"], **dataset_config["args"])
    error_total, T_total = 0.0, 0.0
    der = DER(**config["metrics"][0]["args"])
    frame_err = Frame_ERR(**config["metrics"][1]["args"])
    frame_tier_err = Frame_Tier_ERR(**config["metrics"][2]["args"])

    for sound, mask, label in trainset:
        label = torch.Tensor(label).unsqueeze(0)
        error, T_scored = frame_tier_err(label - 0.5, torch.cat([label[:, :, 1:], label[:, :, :1]], dim=-1))
        # error, T_scored = der(torch.ones(label.shape), label[:, :, :])
        error_total += error
        T_total += T_scored
        # print(error / T_scored)
        print(error)
    print(error_total / T_total)

    # test ACC
    output = torch.Tensor([[0.3, -0.3, 0], [0, 0, -0.2]])
    truth = torch.Tensor([0, 2]).int()
    print(ERR()(output, truth))