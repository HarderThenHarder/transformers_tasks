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

各种匹配模型的实现。

Author: pankeyu
Date: 2022/10/26
"""
from typing import List

import torch
import torch.nn as nn
import numpy as np


class PointwiseMatching(nn.Module):
    """
    PointWise 匹配实现, 高准度但匹配慢, 不适合用作大规模匹配且低时延的场景。
    Code Reference: https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/text_matching/ernie_matching/model.py
    
    Args:
        nn (_type_): _description_
    """

    def __init__(self, encoder, dropout=None):
        """
        init func.

        Args:
            encoder (transformers.AutoModel): backbone, 默认使用 ernie 3.0
            dropout (float): dropout 比例
        """
        super().__init__()
        self.encoder = encoder
        hidden_size = 768
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.classifier = nn.Linear(768, 2)

    def forward(self,
                input_ids,
                token_type_ids,
                position_ids=None,
                attention_mask=None) -> torch.tensor:
        """
        Foward 函数，输入匹配好的pair对，返回二维向量（相似/不相似）。

        Args:
            input_ids (torch.LongTensor): (batch, seq_len)
            token_type_ids (torch.LongTensor): (batch, seq_len)
            position_ids (torch.LongTensor): (batch, seq_len)
            attention_mask (torch.LongTensor): (batch, seq_len)

        Returns:
            torch.tensor: (batch, 2)
        """
        pooled_embedding = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask
        )["pooler_output"]                                  # (batch, hidden_size)
        pooled_embedding = self.dropout(pooled_embedding)   # (batch, hidden_size)
        logits = self.classifier(pooled_embedding)          # (batch, 2)
        
        return logits


class DSSM(nn.Module):
    """
    DSSM(Deep Structured Semantic Model) 模型实现, 采用cos值计算向量相似度, 精度稍低, 但计算速度快。
    Paper Reference: https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf

    Args:
        nn (_type_): _description_
    """

    def __init__(self, encoder, dropout=None):
        """
        init func.

        Args:
            encoder (transformers.AutoModel): backbone, 默认使用 ernie 3.0
            dropout (float): dropout.
        """
        super().__init__()
        self.encoder = encoder
        hidden_size = 768
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

    def forward(
        self,
        input_ids: torch.tensor,
        token_type_ids: torch.tensor,
        attention_mask: torch.tensor
    ) -> torch.tensor:
        """
        forward 函数，输入单句子，获得单句子的embedding。

        Args:
            input_ids (torch.LongTensor): (batch, seq_len)
            token_type_ids (torch.LongTensor): (batch, seq_len)
            attention_mask (torch.LongTensor): (batch, seq_len)

        Returns:
            torch.tensor: embedding -> (batch, hidden_size)
        """
        embedding = self.encoder(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )["pooler_output"]                                  # (batch, hidden_size)
        return embedding

    def get_similarity(
        self,
        query_input_ids: torch.tensor,
        query_token_type_ids: torch.tensor,
        query_attention_mask: torch.tensor,
        doc_input_ids: torch.tensor,
        doc_token_type_ids: torch.tensor,
        doc_attention_mask: torch.tensor
    ) -> torch.tensor:
        """
        输入query和doc的向量，返回query和doc两个向量的余弦相似度。

        Args:
            query_input_ids (torch.LongTensor): (batch, seq_len)
            query_token_type_ids (torch.LongTensor): (batch, seq_len)
            query_attention_mask (torch.LongTensor): (batch, seq_len)
            doc_input_ids (torch.LongTensor): (batch, seq_len)
            doc_token_type_ids (torch.LongTensor): (batch, seq_len)
            doc_attention_mask (torch.LongTensor): (batch, seq_len)

        Returns:
            torch.tensor: (batch, 1)
        """
        query_embedding = self.encoder(
            input_ids=query_input_ids,
            token_type_ids=query_token_type_ids,
            attention_mask=query_attention_mask
        )["pooler_output"]                                 # (batch, hidden_size)
        query_embedding = self.dropout(query_embedding)

        doc_embedding = self.encoder(
            input_ids=doc_input_ids,
            token_type_ids=doc_token_type_ids,
            attention_mask=doc_attention_mask
        )["pooler_output"]                                  # (batch, hidden_size)
        doc_embedding = self.dropout(doc_embedding)

        similarity = nn.functional.cosine_similarity(query_embedding, doc_embedding)
        return similarity


class SentenceTransformer(nn.Module):
    """
    Sentence Transomer实现, 双塔网络, 精度适中, 计算速度快。
    Paper Reference: https://arxiv.org/pdf/1908.10084.pdf
    Code Reference: https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/text_matching/sentence_transformers/model.py
    
    Args:
        nn (_type_): _description_
    """

    def __init__(self, encoder, dropout=0.1):
        """
        init func.

        Args:
            encoder (transformers.PretrainedModel): backbone, 默认使用 ernie 3.0
            dropout (float): dropout.
        """
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.classifier = nn.Linear(768 * 3, 2)         # concat(u, v, u - v) -> 2, 相似/不相似

    def forward(
        self,
        query_input_ids: torch.tensor,
        query_token_type_ids: torch.tensor,
        query_attention_mask: torch.tensor,
        doc_embeddings: torch.tensor,
    ) -> torch.tensor:
        """
        forward 函数，输入query句子和doc_embedding向量，将query句子过一遍模型得到
        query embedding再和doc_embedding做二分类。

        Args:
            input_ids (torch.LongTensor): (batch, seq_len)
            token_type_ids (torch.LongTensor): (batch, seq_len)
            attention_mask (torch.LongTensor): (batch, seq_len)
            doc_embedding (torch.LongTensor): 所有需要匹配的doc_embedding -> (batch, doc_embedding_numbers, hidden_size)

        Returns:
            torch.tensor: embedding_match_logits -> (batch, doc_embedding_numbers, 2)
        """
        query_embedding = self.encoder(
            input_ids=query_input_ids,
            token_type_ids=query_token_type_ids,
            attention_mask=query_attention_mask
        )["last_hidden_state"]                                                  # (batch, seq_len, hidden_size)
        
        query_attention_mask = torch.unsqueeze(query_attention_mask, dim=-1)    # (batch, seq_len, 1)
        query_embedding = query_embedding * query_attention_mask                # (batch, seq_len, hidden_size)
        query_sum_embedding = torch.sum(query_embedding, dim=1)                 # (batch, hidden_size)
        query_sum_mask = torch.sum(query_attention_mask, dim=1)                 # (batch, 1)
        query_mean = query_sum_embedding / query_sum_mask                       # (batch, hidden_size)

        query_mean = query_mean.unsqueeze(dim=1).repeat(1, doc_embeddings.size()[1], 1)  # (batch, doc_embedding_numbers, hidden_size)
        sub = torch.abs(torch.subtract(query_mean, doc_embeddings))                      # (batch, doc_embedding_numbers, hidden_size)
        concat = torch.cat([query_mean, doc_embeddings, sub], dim=-1)                    # (batch, doc_embedding_numbers, hidden_size * 3)
        logits = self.classifier(concat)                                                 # (batch, doc_embedding_numbers, 2)
        return logits

    def get_embedding(
        self,
        input_ids: torch.tensor,
        token_type_ids: torch.tensor,
        attention_mask: torch.tensor,
    ) -> torch.tensor:
        """
        输入句子，返回这个句子的embedding，用于事先计算doc embedding并存储。

        Args:
            input_ids (torch.LongTensor): (batch, seq_len)
            token_type_ids (torch.LongTensor): (batch, seq_len)
            attention_mask (torch.LongTensor): (batch, seq_len)

        Returns:
            torch.tensor: embedding向量 -> (batch, hidden_size)
        """
        embedding = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )["last_hidden_state"]                                                  # (batch, seq_len, hidden_size)
        
        attention_mask = torch.unsqueeze(attention_mask, dim=-1)                # (batch, seq_len, 1)
        embedding = embedding * attention_mask                                  # (batch, seq_len, hidden_size)
        sum_embedding = torch.sum(embedding, dim=1)                             # (batch, hidden_size)
        sum_mask = torch.sum(attention_mask, dim=1)                             # (batch, 1)
        mean = sum_embedding / sum_mask                                         # (batch, hidden_size)
        return mean
    
    def get_similarity_label(
        self,
        query_input_ids: torch.tensor,
        query_token_type_ids: torch.tensor,
        query_attention_mask: torch.tensor,
        doc_input_ids: torch.tensor,
        doc_token_type_ids: torch.tensor,
        doc_attention_mask: torch.tensor
    ) -> torch.tensor:
        """
        forward 函数，输入query和doc的向量，返回两个向量相似/不相似的二维向量。

        Args:
            query_input_ids (torch.LongTensor): (batch, seq_len)
            query_token_type_ids (torch.LongTensor): (batch, seq_len)
            query_attention_mask (torch.LongTensor): (batch, seq_len)
            doc_input_ids (torch.LongTensor): (batch, seq_len)
            doc_token_type_ids (torch.LongTensor): (batch, seq_len)
            doc_attention_mask (torch.LongTensor): (batch, seq_len)

        Returns:
            torch.tensor: (batch, 2)
        """
        query_embedding = self.encoder(
            input_ids=query_input_ids,
            token_type_ids=query_token_type_ids,
            attention_mask=query_attention_mask
        )["last_hidden_state"]                                                  # (batch, seq_len, hidden_size)
        query_embedding = self.dropout(query_embedding)                         # (batch, seq_len, hidden_size)
        query_attention_mask = torch.unsqueeze(query_attention_mask, dim=-1)    # (batch, seq_len, 1)
        query_embedding = query_embedding * query_attention_mask                # (batch, seq_len, hidden_size)
        query_sum_embedding = torch.sum(query_embedding, dim=1)                 # (batch, hidden_size)
        query_sum_mask = torch.sum(query_attention_mask, dim=1)                 # (batch, 1)
        query_mean = query_sum_embedding / query_sum_mask                       # (batch, hidden_size)

        doc_embedding = self.encoder(
            input_ids=doc_input_ids,
            token_type_ids=doc_token_type_ids,
            attention_mask=doc_attention_mask
        )["last_hidden_state"]                                                  # (batch, seq_len, hidden_size)
        doc_embedding = self.dropout(doc_embedding)                             # (batch, seq_len, hidden_size)
        doc_attention_mask = torch.unsqueeze(doc_attention_mask, dim=-1)        # (batch, seq_len, 1)
        doc_embedding = doc_embedding * doc_attention_mask                      # (batch, seq_len, hidden_size)
        doc_sum_embdding = torch.sum(doc_embedding, dim=1)                      # (batch, hidden_size)
        doc_sum_mask = torch.sum(doc_attention_mask, dim=1)                     # (batch, 1)
        doc_mean = doc_sum_embdding / doc_sum_mask                              # (batch, hidden_size)

        sub = torch.abs(torch.subtract(query_mean, doc_mean))                   # (batch, hidden_size)
        concat = torch.cat([query_mean, doc_mean, sub], dim=-1)                 # (batch, hidden_size * 3)
        logits = self.classifier(concat)                                        # (batch, 2)

        return logits


if __name__ == '__main__':
    from rich import print
    from transformers import AutoTokenizer, AutoModel
    from utils import convert_dssm_example

    device = 'cpu'
    tokenizer = AutoTokenizer.from_pretrained('nghuyong/ernie-3.0-base-zh')
    encoder = AutoModel.from_pretrained('nghuyong/ernie-3.0-base-zh')
    model = SentenceTransformer(encoder).to(device)

    example = {
        "text": [
            '今天天气好吗	今天天气怎样	1'
        ]
    }
    batch = convert_dssm_example(example, tokenizer, 10)
    print(batch)

    # * 测试sentence bert训练输出logits
    # output = model.get_similarity_label(query_input_ids=torch.LongTensor(batch['query_input_ids']),
    #                         query_token_type_ids=torch.LongTensor(batch['query_token_type_ids']),
    #                         query_attention_mask=torch.LongTensor(batch['query_attention_mask']),
    #                         doc_input_ids=torch.LongTensor(batch['doc_input_ids']),
    #                         doc_token_type_ids=torch.LongTensor(batch['doc_token_type_ids']),
    #                         doc_attention_mask=torch.LongTensor(batch['doc_attention_mask']))

    # * 测试sentence bert的inference功能
    # output = model(query_input_ids=torch.LongTensor(batch['query_input_ids']).to(device),
    #                 query_token_type_ids=torch.LongTensor(batch['query_token_type_ids']).to(device),
    #                 query_attention_mask=torch.LongTensor(batch['query_attention_mask']).to(device),
    #                 doc_embeddings=torch.randn(1, 10, 768).to(device))

    # * 测试sentence bert获取sentence embedding功能
    output = model.get_embedding(input_ids=torch.LongTensor(batch['query_input_ids']).to(device),
                                    token_type_ids=torch.LongTensor(batch['query_token_type_ids']).to(device),
                                    attention_mask=torch.LongTensor(batch['query_attention_mask']).to(device))
    print(output, output.size())