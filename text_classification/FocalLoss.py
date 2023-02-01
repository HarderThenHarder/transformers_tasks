# !/usr/bin/env python3
"""
==== No Bugs in code, just some Random Unexpected FEATURES ====
┌─────────────────────────────────────────────────────────────┐
│┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
│├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
│├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
│├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
│└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
│      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
│      └───┴─────┴───────────────────────┴─────┴───┘          │
└─────────────────────────────────────────────────────────────┘

Focal Loss 用于缓解样本类别不均衡问题。

Author: pankeyu
Date: 2023/02/01
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    """
    Focal Loss Class.
    focal_loss = 
        -(1 - alpha) * softmax(x) ^ gamma * log(1 - softmax(x))   ->  when y=0
        -alpha * (1 - softmax(x)) ^ gamma * log_softmax(x)        ->  when y=1

    Args:
        nn (_type_): _description_
    """

    def __init__(self, gamma=2.0, alpha=0.25, device='cpu'):
        """
        Init Func.

        Args:
            gamma (int, optional): defaults to 0, equals to CrossEntropy Loss.
            alpha (_type_, optional): balance params. Defaults to None.
        """
        super().__init__()
        self.gamma = gamma
        self.device = device
        self.alpha = torch.Tensor([alpha, 1 - alpha]).to(device)

    def forward(self, logits: torch.tensor, target: torch.tensor):
        """
        forward func.

        Args:
            logits (torch.tensor): logtis, (batch, n_classes)
            target (torch.tensor): labels, (batch, )

        Returns:
            focal loss, (1,)
        """
        target = target.view(-1, 1)                             # (batch, 1)
        logpt = F.log_softmax(logits, dim=-1)                   # (batch, n_classes)
        logpt = logpt.gather(dim=-1, index=target).view(-1)     # (batch, )
        pt = Variable(logpt.data.exp(), requires_grad=True)     # (batch, )

        if self.alpha is not None:
            at = self.alpha.gather(0, target.data.view(-1))     # (batch, )
            logpt = logpt * Variable(at, requires_grad=True)    # (batch, )

        loss = -1 * (1 - pt) ** self.gamma * logpt              # (batch, )
        return loss.mean()


if __name__ == '__main__':
    from rich import print
    random_logits = torch.rand(8, 2)
    target = torch.LongTensor([1, 1, 1, 1, 0, 0, 0, 0])
    loss_func = FocalLoss()
    loss = loss_func(random_logits, target)
    print(loss)
    loss.backward()