import torch
median_filter = 11

pad = torch.nn.ReflectionPad1d(median_filter//2)
logits_batch = torch.Tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]])
logits_batch = pad(logits_batch)
# [B, C, T, median_filter]
logits_batch = logits_batch.unfold(-1, median_filter, 1)
# [B, C, T]
logits_batch, _ = torch.median(logits_batch, dim=-1)
print(logits_batch)