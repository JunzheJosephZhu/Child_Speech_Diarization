import torch
import sys
def CCE_acc(logits, target, mask):
    '''
        logits: [B, C, T], float
        target: [B, T], int
        mask: [B, T], float
    '''
    loss = torch.nn.CrossEntropyLoss(reduce=False)(logits, target)*mask
    loss = loss.sum()/mask.sum()
    correct = (logits.argmax(1) == target).int()
    acc = (correct * mask).sum()/mask.sum()
    return loss, acc