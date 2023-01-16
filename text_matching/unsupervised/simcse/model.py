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

ESimCSE模型类实现。
ESimCSE是SimCSE的一种增强方式，通过重塑正/负例构建方法来提升模型性能。

Reference:
    Paper: https://arxiv.org/pdf/2109.04380.pdf
    Code : https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/text_matching/simcse/model.py

Author: pankeyu
Date: 2023/01/13
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCSE(nn.Module):
    """
    SimCSE模型，采用ESimCSE方式实现。
    
    Args:
        nn (_type_): _description_
    """

    def __init__(
        self, 
        encoder, 
        dropout=None,
        margin=0.0,
        scale=20,
        output_embedding_dim=256):
        """
        Init func.

        Args:
            encoder (_type_): pretrained model, 默认使用 ernie3.0。
            dropout (_type_, optional): hidden_state 的 dropout 比例。
            margin (float, optional): 为所有正例的余弦相似度降低的值. Defaults to 0.0.
            scale (int, optional): 缩放余弦相似度的值便于模型收敛. Defaults to 20.
            output_embedding_dim (_type_, optional): 输出维度（是否将默认的768维度压缩到更小的维度）. Defaults to 256.
        """
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.output_embedding_dim = output_embedding_dim
        if output_embedding_dim > 0:
            self.embedding_reduce_linear = nn.Linear(768, output_embedding_dim)
        self.margin = margin
        self.scale = scale
        self.criterion = torch.nn.CrossEntropyLoss()

    def get_pooled_embedding(
        self, 
        input_ids, 
        token_type_ids=None,
        attention_mask=None
        ) -> torch.tensor:
        """
        获得句子的embedding，如果有压缩，则返回压缩后的embedding。

        Args:
            input_ids (_type_): _description_
            token_type_ids (_type_, optional): _description_. Defaults to None.
            attention_mask (_type_, optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: (batch, self.output_embedding_dim)
        """
        pooled_embedding = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )["pooler_output"]

        if self.output_embedding_dim > 0:
            pooled_embedding = self.embedding_reduce_linear(pooled_embedding)       # 维度压缩
        pooled_embedding = self.dropout(pooled_embedding)                           # dropout
        pooled_embedding = F.normalize(pooled_embedding, p=2, dim=-1)
        
        return pooled_embedding

    def forward(
        self,
        query_input_ids: torch.tensor,
        query_token_type_ids: torch.tensor,
        doc_input_ids: torch.tensor,
        doc_token_type_ids: torch.tensor,
        device='cpu'
        ) -> torch.tensor:
        """
        传入query/doc对，构建正/负例并计算contrastive loss。

        Args:
            query_input_ids (torch.LongTensor): (batch, seq_len)
            query_token_type_ids (torch.LongTensor): (batch, seq_len)
            doc_input_ids (torch.LongTensor): (batch, seq_len)
            doc_token_type_ids (torch.LongTensor): (batch, seq_len)
            device (str): 使用设备

        Returns:
            torch.tensor: (1)
        """
        query_embedding = self.get_pooled_embedding(
            input_ids=query_input_ids,
            token_type_ids=query_token_type_ids
        )                                                           # (batch, self.output_embedding_dim)

        doc_embedding = self.get_pooled_embedding(
            input_ids=doc_input_ids,
            token_type_ids=doc_token_type_ids
        )                                                           # (batch, self.output_embedding_dim)
        
        cos_sim = torch.matmul(query_embedding, doc_embedding.T)    # (batch, batch)
        margin_diag = torch.diag(torch.full(                        # (batch, batch), 只有对角线等于margin值的对角矩阵
            size=[query_embedding.size()[0]], 
            fill_value=self.margin
        )).to(device)
        cos_sim = cos_sim - margin_diag                             # 主对角线（正例）的余弦相似度都减掉 margin
        cos_sim *= self.scale                                       # 缩放相似度，便于收敛

        labels = torch.arange(                                      # 只有对角上为正例，其余全是负例，所以这个batch样本标签为 -> [0, 1, 2, ...]
            0, 
            query_embedding.size()[0], 
            dtype=torch.int64
        ).to(device)
        loss = self.criterion(cos_sim, labels)

        return loss


if __name__ == '__main__':
    from rich import print
    from transformers import AutoTokenizer, AutoModel

    device = 'cpu'
    tokenizer = AutoTokenizer.from_pretrained('nghuyong/ernie-3.0-base-zh')
    encoder = AutoModel.from_pretrained('nghuyong/ernie-3.0-base-zh')
    model = SimCSE(encoder).to(device)

    sentences = ['一个男孩在打篮球', '他是蔡徐坤吗', '他怎么这么帅呢']
    query_inputs = tokenizer(
        sentences, 
        return_tensors='pt',
        max_length=20,
        padding='max_length'
    )
    doc_inputs = tokenizer(
        sentences, 
        return_tensors='pt',
        max_length=20,
        padding='max_length'
    )

    # * 测试SimCSE训练输出loss
    loss = model(query_input_ids=query_inputs['input_ids'],
                    query_token_type_ids=query_inputs['token_type_ids'],
                    query_attention_mask=query_inputs['attention_mask'],
                    doc_input_ids=doc_inputs['input_ids'],
                    doc_token_type_ids=doc_inputs['token_type_ids'],
                    doc_attention_mask=doc_inputs['attention_mask'])
    print('loss: ', loss)
