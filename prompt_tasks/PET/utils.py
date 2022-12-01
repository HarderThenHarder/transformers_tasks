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

工具类，包含prompt自定义字段的填值。

Author: pankeyu
Date: 2022/11/28
"""
import json
import traceback
from typing import List

import torch
import numpy as np

from Template import HardTemplate


def convert_example(
    examples: dict, 
    tokenizer, 
    max_seq_len: int,
    max_label_len: int,
    template: HardTemplate,
    train_mode=True,
    return_tensor=False
    ) -> dict:
    """
    将样本数据转换为模型接收的输入数据。

    Args:
        examples (dict): 训练数据样本, e.g. -> {
                                                "text": [
                                                            '手机	这个手机也太卡了。',
                                                            '体育	世界杯为何迟迟不见宣传',
                                                            ...
                                                ]
                                            }
        max_seq_len (int): 句子的最大长度，若没有达到最大长度，则padding为最大长度
        max_label_len (int): 最大label长度，若没有达到最大长度，则padding为最大长度
        template (HardTemplate): 模板类。
        train_mode (bool): 训练阶段 or 推理阶段。
        return_tensor (bool): 是否返回tensor类型，如不是，则返回numpy类型。

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'input_ids': [[1, 47, 10, 7, 304, 3, 3, 3, 3, 47, 27, 247, 98, 105, 512, 777, 15, 12043, 2], ...],
                            'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ...],
                            'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], ...],
                            'mask_positions': [[5, 6, 7, 8], ...],
                            'mask_labels': [[2372, 3442, 0, 0], [2643, 4434, 2334, 0], ...]
                        }
    """
    tokenized_output = {
        'input_ids': [],
        'token_type_ids': [],
        'attention_mask': [],
        'mask_positions': [],
        'mask_labels': []
        }

    for i, example in enumerate(examples['text']):
        try:
            if train_mode:
                label, content = example.strip().split('\t')
            else:
                content = example.strip()

            inputs_dict={
                'textA': content,
                'MASK': '[MASK]'
            }
            encoded_inputs = template(
                inputs_dict=inputs_dict,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                mask_length=max_label_len
            )
        except:
            print(f'Error Line {i+1}: "{example}" -> {traceback.format_exc()}')
            exit()
        tokenized_output['input_ids'].append(encoded_inputs["input_ids"])
        tokenized_output['token_type_ids'].append(encoded_inputs["token_type_ids"])
        tokenized_output['attention_mask'].append(encoded_inputs["attention_mask"])
        tokenized_output['mask_positions'].append(encoded_inputs["mask_position"])
        
        if train_mode:
            label_encoded = tokenizer(text=[label])                                     # 将label补到最大长度
            label_encoded = label_encoded['input_ids'][0][1:-1]
            label_encoded = label_encoded[:max_label_len]
            label_encoded = label_encoded + [tokenizer.pad_token_id] * (max_label_len - len(label_encoded))
            tokenized_output['mask_labels'].append(label_encoded)
    
    for k, v in tokenized_output.items():
        if return_tensor:
            tokenized_output[k] = torch.LongTensor(v)
        else:
            tokenized_output[k] = np.array(v)

    return tokenized_output


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
        if not single_sub_mask_labels.size():                                               # 处理单token维度下维度缺失的问题
            single_sub_mask_labels = single_sub_mask_labels.unsqueeze(dim=0)
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
    predict_tokens = mask_logits.argmax(dim=-1)                                 # (batch * label_num)
    predict_tokens = predict_tokens.reshape(-1, label_length)                   # (batch, label_num)

    return predict_tokens


if __name__ == '__main__':
    from rich import print

    logits = torch.randn(1, 20, 21193)
    mask_positions = torch.LongTensor([
        [3, 4]
    ])
    predict_tokens = convert_logits_to_ids(logits, mask_positions)
    print(predict_tokens)