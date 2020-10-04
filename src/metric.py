import torch
EPS = 1e-6

def get_zeros(tensor):
    zeros = torch.zeros(tensor.size(), dtype=tensor.dtype)
    zeros = zeros.cuda(tensor.get_device()) if tensor.is_cuda else zeros
    return zeros

class DER():
    def __init__(self, threshold=0.5, median_filter=5):
        self.threshold = threshold
        self.pad = torch.nn.ReflectionPad1d(median_filter//2)
        self.median_filter = median_filter

    def __call__(self, logits_batch, label_batch):
        '''
            logits_batch: torch.Tensor, [B, C, T]
            truth_batch: torch.Tensor, [B, C, T]
        '''
        B, C, T = logits_batch.size()
        error, T_scored = 0.0, 0.0
        # [B, C, T + median_filter // 2 * 2]
        logits_batch = self.pad(logits_batch)
        # [B, C, T, median_filter]
        logits_batch = logits_batch.unfold(-1, self.median_filter, 1)
        # [B, C, T]
        logits_batch, _ = torch.median(logits_batch, dim=-1)
        for idx in range(B):
            # [T, C], [T, C]
            logits, label = logits_batch[idx].T, label_batch[idx].T
            decisions = (torch.sigmoid(logits) > self.threshold).long()
            label = label.long()
            # [T], total number of speakers in each frame
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
            # [T], total of agreed speakers in each frame
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
        return error, T_scored

if __name__ == "__main__":
    a = torch.randn(20)
    import os
    import json
    from data import LENADataSet
    root = "/ws/ifp-10_3/hasegawa/junzhez2/MaxMin_Pytorch"
    config_filename = "configs/SAE_RNN.json"
    with open(os.path.join(root, config_filename)) as file:
        config = json.load(file)
    dataset_config = config["dataset"]
    trainset = LENADataSet(dataset_config["train"], **dataset_config["args"])
    testset = LENADataSet(dataset_config["test"], **dataset_config["args"])
    error_total, T_total = 0.0, 0.0
    der = DER(**config["metric"]["args"])
    for sound, label in trainset:
        label = torch.Tensor(label).unsqueeze(0)
        error, T_scored = der(label - 0.5, torch.cat([label[:, :, 1:], label[:, :, :1]], dim=-1))
        error, T_scored = der(label[:, :, 2:], label[:, :, 1 : -1])
        error_total += error
        T_total += T_scored
    print(error_total/T_total)