import torch
import sys

class FocalBCE(torch.nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(self, alpha_pos=0.25, alpha_neg=0.75, gamma=2.0, ignore_index=None, reduction='mean'):
        super(FocalBCE, self).__init__()

        self.alpha = [alpha_pos, alpha_neg]
        self.gamma = gamma
        self.smooth = 1e-6
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_loss = -self.alpha[0] * torch.pow(torch.sub(1.0, prob), self.gamma) * torch.log(prob) * pos_mask
        neg_loss = -self.alpha[1] * torch.pow(prob, self.gamma) * \
                   torch.log(torch.sub(1.0, prob)) * neg_mask

        neg_loss = neg_loss.sum()
        pos_loss = pos_loss.sum()
        num_pos = pos_mask.view(pos_mask.size(0), -1).sum()
        num_neg = neg_mask.view(neg_mask.size(0), -1).sum()

        return (neg_loss + pos_loss) / (num_pos + num_neg)

def BCE(**args):
    return torch.nn.BCEWithLogitsLoss(**args)

def CCE(**args):
    return torch.nn.CrossEntropyLoss(**args)


if __name__ == '__main__':
    torch.manual_seed(0)
    B, C, T = 2, 3, 80
    pred = torch.randn(2, 3, 80)
    target1 = (pred > 0).float()
    target2 = (pred < 0).float()
    print(FocalBCE()(pred, target1), FocalBCE()(pred, target2))
    print(BCE()(pred, target1), BCE()(pred, target2))