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

R-Drop Loss, 由于Backbone中通常存在Dropout，因此可以通过减小同一个样本
经过两次backbone之后的logits分布差异，来增强模型的鲁棒性。

Paper Reference:
    https://arxiv.org/pdf/2106.14448.pdf

Code Reference:
    https://github.com/dropreg/R-Drop

Author: pankeyu
Date: 2022/11/15
"""
import torch
import torch.nn.functional as F


class RDropLoss(object):
    """
    RDrop Loss 类。

    Args:
        object (_type_): _description_
    """

    def __init__(self, reduction='none'):
        """
        init func.

        Args:
            reduction (str, optional): kl-divergence param. Defaults to 'none'.
        """
        super().__init__()
        if reduction not in ['sum', 'mean', 'none', 'batchmean']:
            raise ValueError(
                "@param reduction must in ['sum', 'mean', 'batchmean', 'none'], "
                "while received {}.".format(reduction))
        self.reduction = reduction

    def compute_kl_loss(
        self,
        logits: torch.tensor, 
        logtis2: torch.tensor,
        pad_mask=None,
        device='cpu'
        ) -> torch.tensor:
        """
        输入同一个样本经过两次backbone后的结果，计算KL-Divergence。

        Args:
            logits (torch.tensor): 第一次logits
            logtis2 (torch.tensor): 第二次logits
            pad_mask (torch.tensor): mask向量，用于去掉padding token的影响
            device (str): cpu or gpu

        Returns:
            torch.tensor: _description_
        """
        loss1 = F.kl_div(F.log_softmax(logits, dim=-1),
                        F.softmax(logtis2, dim=-1),
                        reduction=self.reduction)
        loss2 = F.kl_div(F.log_softmax(logtis2, dim=-1),
                        F.softmax(logits, dim=-1),
                        reduction=self.reduction)

        if pad_mask is not None:
            pad_mask = self.generate_mask_tensor(loss1, pad_mask).to(device)
            loss1 = torch.masked_select(loss1, pad_mask)
            loss2 = torch.masked_select(loss2, pad_mask)
        
        loss = (loss1.sum() + loss2.sum()) / 2
        return loss

    def generate_mask_tensor(
        self, 
        loss1: torch.tensor,
        pad_mask: torch.tensor
        ) -> torch.tensor:
        """
        根据二维的attention_mask生成三维的mask矩阵，用于过滤掉loss中
        的padding token的值。

        Args:
            loss1 (torch.tensor): (batch, seq_len, vocab_size)
            pad_mask (torch.tensor): (batch, seq_len)

        Returns:
            torch.tensor: (batch, seq_len, vocab_size)
        """
        mask_tensor = []
        batch, seq_len, vocab_size = loss1.size()
        for batch_idx in range(batch):
            for seq_idx in range(seq_len):
                if pad_mask[batch_idx][seq_idx]:
                    mask_tensor.append([True] * vocab_size)
                else:
                    mask_tensor.append([False] * vocab_size)
        mask_tensor = torch.tensor(mask_tensor).reshape(batch, seq_len, vocab_size)
        return mask_tensor


if __name__ == '__main__':
    rdrop = RDropLoss()
    loss = torch.randn(2, 5, 3)     # (2, 5, 3)
    pad_mask = torch.LongTensor([
        [1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0]
    ])                              # (2, 5)
    pad_mask = rdrop.generate_mask_tensor(loss, pad_mask)
    print(torch.masked_select(loss, pad_mask))
