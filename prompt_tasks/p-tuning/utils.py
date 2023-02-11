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

工具类。

Author: pankeyu
Date: 2022/11/11
"""
import json
import traceback
from typing import List

import torch
import numpy as np


def convert_example(
    examples: dict, 
    tokenizer, 
    max_seq_len: int,
    max_label_len: int,
    p_embedding_num=6,
    train_mode=True,
    return_tensor=False
    ) -> dict:
    """
    将样本数据转换为模型接收的输入数据。

    Args:
        examples (dict): 训练数据样本, e.g. -> {
                                                "text": [
                                                            '娱乐	嗨放派怎么停播了',
                                                            '体育	世界杯为何迟迟不见宣传',
                                                            ...
                                                ]
                                            }
        max_label_len (int): 最大label长度，若没有达到最大长度，则padding为最大长度
        p_embedding_num (int): p-tuning token 的个数
        train_mode (bool): 训练阶段 or 推理阶段。
        return_tensor (bool): 是否返回tensor类型，如不是，则返回numpy类型。

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'input_ids': [[101, 3928, ...], [101, 4395, ...]], 
                            'token_type_ids': [[0, 0, ...], [0, 0, ...]],
                            'mask_positions': [[5, 6, ...], [3, 4, ...]],
                            'mask_labels': [[183, 234], [298, 322], ...]
                        }
    """
    tokenized_output = {
            'input_ids': [], 
            'attention_mask': [],
            'mask_positions': [],                                           # 记录label的位置（即MASK Token的位置）
            'mask_labels': []                                               # 记录MASK Token的原始值（即Label值）
        }

    for i, example in enumerate(examples['text']):
        try:
            start_mask_position = 1                                         # 将 prompt token(s) 插在 [CLS] 之后
            
            if train_mode:
                label, content = example.strip().split('\t')
            else:
                content = example.strip()

            encoded_inputs = tokenizer(
                text=content,
                truncation=True,
                max_length=max_seq_len,
                padding='max_length')
        except:
            print(f'Error Line {i+1}: "{example}" -> {traceback.format_exc()}')
            continue

        input_ids = encoded_inputs['input_ids']
        mask_tokens = ['[MASK]'] * max_label_len                                            # 1.生成 MASK Tokens, 和label长度一致
        mask_ids = tokenizer.convert_tokens_to_ids(mask_tokens)                             # token 转 id

        p_tokens = ["[unused{}]".format(i+1) for i in range(p_embedding_num)]               # 2.构建 prompt token(s)
        p_tokens_ids = tokenizer.convert_tokens_to_ids(p_tokens)                            # token 转 id

        tmp_input_ids = input_ids[:-1]
        tmp_input_ids = tmp_input_ids[:max_seq_len-len(mask_ids)-len(p_tokens_ids)-1]       # 根据最大长度-p_token长度-label长度，裁剪content的长度
        tmp_input_ids = tmp_input_ids[:start_mask_position] + mask_ids + tmp_input_ids[     # 3.插入 MASK -> [CLS][MASK][MASK]世界杯...[SEP]
                    start_mask_position:]
        input_ids = tmp_input_ids + [input_ids[-1]]                                         # 补上[SEP]
        input_ids = p_tokens_ids + input_ids                                                # 4.插入 prompt -> [unused1][unused2]...[CLS][MASK]...[SEP]
        mask_positions = [len(p_tokens_ids) + start_mask_position + i for                   # 将 Mask Tokens 的位置记录下来
                            i in range(max_label_len)]

        tokenized_output['input_ids'].append(input_ids)
        if 'token_type_ids' in tokenized_output:                                            # 兼容不需要 token_type_id 的模型, e.g. Roberta-Base
            if 'token_type_ids' not in tokenized_output:
                tokenized_output['token_type_ids'] = [encoded_inputs['token_type_ids']]
            else:
                tokenized_output['token_type_ids'].append(encoded_inputs['token_type_ids'])
        tokenized_output['attention_mask'].append(encoded_inputs['attention_mask'])
        tokenized_output['mask_positions'].append(mask_positions)

        if train_mode:
            mask_labels = tokenizer(text=label)                                                 # label token 转 id
            mask_labels = mask_labels['input_ids'][1:-1]                                        # 丢掉[CLS]和[SEP]
            mask_labels = mask_labels[:max_label_len]
            mask_labels +=  [tokenizer.pad_token_id] * (max_label_len - len(mask_labels))       # 将 label 补到最长
            tokenized_output['mask_labels'].append(mask_labels)

    for k, v in tokenized_output.items():
        if return_tensor:
            tokenized_output[k] = torch.LongTensor(v)
        else:
            tokenized_output[k] = np.array(v)

    return tokenized_output


def mlm_loss2(
    logits: torch.tensor,
    mask_positions: torch.tensor,
    mask_labels: torch.tensor,
    cross_entropy_criterion: torch.nn.CrossEntropyLoss,
    masked_lm_scale=1.0,
    device='cpu'
    ) -> torch.tensor:
    """
    计算指定位置的mask token的output与label之间的cross entropy loss。

    Args:
        logits (torch.tensor): 模型原始输出 -> (batch, seq_len, vocab_size)
        mask_positions (torch.tensor): mask token的位置  -> (batch, mask_label_num)
        mask_labels (torch.tensor): mask token的label -> (batch, mask_label_num)
        cross_entropy_criterion (CrossEntropyLoss): CE Loss计算器
        masked_lm_scale (float): scale 参数
        device (str): cpu还是gpu
    
    Returns:
        torch.tensor: CE Loss
    """
    batch_size, seq_len, vocab_size = logits.size()
    mask_positions_after_reshaped = []
    for batch, mask_pos in enumerate(mask_positions.detach().cpu().numpy().tolist()):
        for pos in mask_pos:
            mask_positions_after_reshaped.append(batch * seq_len + pos)
    
    logits = logits.reshape(batch_size * seq_len, -1)                           # (batch_size * seq_len, vocab_size)
    mask_logits = logits[mask_positions_after_reshaped]                         # (batch * label_num, vocab_size)
    mask_labels = mask_labels.reshape(-1, 1).squeeze()                          # (batch * label_num)
    loss = cross_entropy_criterion(mask_logits, mask_labels)
    
    return loss / masked_lm_scale


def mlm_loss(
    logits: torch.tensor,
    mask_positions: torch.tensor,
    sub_mask_labels: list,
    cross_entropy_criterion: torch.nn.CrossEntropyLoss,
    masked_lm_scale=1.0,
    device='cpu'
    ) -> torch.tensor:
    """
    计算指定位置的mask token的output与label之间的cross entropy loss。

    Args:
        logits (torch.tensor): 模型原始输出 -> (batch, seq_len, vocab_size)
        mask_positions (torch.tensor): mask token的位置  -> (batch, mask_label_num)
        sub_mask_labels (list): mask token的sub label, 由于每个label的sub_label数目不同，所以这里是个变长的list, 
                                    e.g. -> [
                                        [[2398, 3352]],
                                        [[2398, 3352], [3819, 3861]]
                                    ]
        cross_entropy_criterion (CrossEntropyLoss): CE Loss计算器
        masked_lm_scale (float): scale 参数
        device (str): cpu还是gpu
    
    Returns:
        torch.tensor: CE Loss
    """
    batch_size, seq_len, vocab_size = logits.size()
    loss = None
    for single_logits, single_sub_mask_labels, single_mask_positions in zip(logits, sub_mask_labels, mask_positions):
        single_mask_logits = single_logits[single_mask_positions]                           # (mask_label_num, vocab_size)
        single_mask_logits = single_mask_logits.repeat(len(single_sub_mask_labels), 1, 1)   # (sub_label_num, mask_label_num, vocab_size)
        single_mask_logits = single_mask_logits.reshape(-1, vocab_size)                     # (sub_label_num * mask_label_num, vocab_size)
        single_sub_mask_labels = torch.LongTensor(single_sub_mask_labels).to(device)        # (sub_label_num, mask_label_num)
        single_sub_mask_labels = single_sub_mask_labels.reshape(-1, 1).squeeze()            # (sub_label_num * mask_label_num)
        cur_loss = cross_entropy_criterion(single_mask_logits, single_sub_mask_labels)
        cur_loss = cur_loss / len(single_sub_mask_labels)
        if not loss:
            loss = cur_loss
        else:
            loss += cur_loss
    loss = loss / batch_size                                                                # (1,)
    return loss / masked_lm_scale


def convert_logits_to_ids(
    logits: torch.tensor, 
    mask_positions: torch.tensor
    ) -> torch.LongTensor:
    """
    输入Language Model的词表概率分布（LMModel的logits），将mask_position位置的
    token logits转换为token的id。

    Args:
        logits (torch.tensor): model output -> (batch, seq_len, vocab_size)
        mask_positions (torch.tensor): mask token的位置 -> (batch, mask_label_num)

    Returns:
        torch.LongTensor: 对应mask position上最大概率的推理token -> (batch, mask_label_num)
    """
    label_length = mask_positions.size()[1]                                     # 标签长度
    batch_size, seq_len, vocab_size = logits.size()
    mask_positions_after_reshaped = []
    for batch, mask_pos in enumerate(mask_positions.detach().cpu().numpy().tolist()):
        for pos in mask_pos:
            mask_positions_after_reshaped.append(batch * seq_len + pos)
    
    logits = logits.reshape(batch_size * seq_len, -1)                           # (batch_size * seq_len, vocab_size)
    mask_logits = logits[mask_positions_after_reshaped]                         # (batch * label_num, vocab_size)
    predicate_tokens = mask_logits.argmax(dim=-1)                               # (batch * label_num)
    predicate_tokens = predicate_tokens.reshape(-1, label_length)               # (batch, label_num)

    return predicate_tokens


if __name__ == '__main__':
    from rich import print

    logits = torch.randn(1, 20, 21193)
    mask_positions = torch.LongTensor([
        [3, 4]
    ])
    predicate_tokens = convert_logits_to_ids(logits, mask_positions)
    print(predicate_tokens)