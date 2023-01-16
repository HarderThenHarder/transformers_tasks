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
Date: 2022/10/26
"""
import random
import traceback

import torch
import numpy as np


def convert_example(
    examples: dict, 
    tokenizer, 
    max_seq_len: int,
    mode='train'
    ):
    """
    将样本数据转换为模型接收的输入数据。

    Args:
        examples (dict): 数据样本（不同mode下的数据集不一样）, e.g. -> {
                                                "text": '蛋黄吃多了有什么坏处',                               # train mode
                                                        or '蛋黄吃多了有什么坏处	吃鸡蛋白过多有什么坏处	0',  # evaluate mode
                                                        or '蛋黄吃多了有什么坏处	吃鸡蛋白过多有什么坏处',     # inference mode
                                            }
        mode (bool): 数据集格式 -> 'train': （无监督）训练集模式，一行只有一句话；
                                'evaluate': 验证集训练集模式，两句话 + 标签
                                'inference': 推理集模式，两句话。

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'query_input_ids': [[101, 3928, ...], [101, 4395, ...]], 
                            'query_token_type_ids': [[0, 0, ...], [0, 0, ...]],
                            'doc_input_ids': [[101, 2648, ...], [101, 3342, ...]], 
                            'doc_token_type_ids': [[0, 0, ...], [0, 0, ...]],
                            'labels': [1, 0, ...]
                        }
    """
    tokenized_output = {
        'query_input_ids': [], 
        'query_token_type_ids': [],
        'doc_input_ids': [], 
        'doc_token_type_ids': []
    }

    for example in examples['text']:
        try:
            if mode == 'train':
                query = doc = example.strip()
            elif mode == 'evaluate':
                query, doc, label = example.strip().split('\t')
            elif mode == 'inference':
                query, doc = example.strip().split('\t')
            else:
                raise ValueError(f'No mode called {mode}, expected in ["train", "evaluate", "inference"].')
            
            query_encoded_inputs = tokenizer(
                text=query,
                truncation=True,
                max_length=max_seq_len,
                padding='max_length')
            doc_encoded_inputs = tokenizer(
                text=doc,
                truncation=True,
                max_length=max_seq_len,
                padding='max_length')
        except:
            print(f'{examples["text"]} -> {traceback.format_exc()}')
            exit()

        tokenized_output['query_input_ids'].append(query_encoded_inputs["input_ids"])
        tokenized_output['query_token_type_ids'].append(query_encoded_inputs["token_type_ids"])
        tokenized_output['doc_input_ids'].append(doc_encoded_inputs["input_ids"])
        tokenized_output['doc_token_type_ids'].append(doc_encoded_inputs["token_type_ids"])
        if mode == 'evaluate':
            if 'labels' not in tokenized_output:
                tokenized_output['labels'] = []
            tokenized_output['labels'].append(int(label))
    
    for k, v in tokenized_output.items():
        tokenized_output[k] = np.array(v)

    return tokenized_output


def word_repetition(
    input_ids, 
    token_type_ids, 
    dup_rate=0.32,
    min_dup_sentence_len_threshold=5,
    device='cpu'
    ) -> torch.tensor:
    """
    随机重复单词策略，用于在正例样本中添加噪声。

    Args:
        input_ids (_type_): y
        token_type_ids (_type_): _description_
        dup_rate (float, optional): 重复字数占总句子长度的比例. Defaults to 0.32.
        min_dup_sentence_len_threshold (int): 触发随机重复的最小句子长度
        device (str): 使用设备

    Returns:
        _type_: 随机重复后的 input_ids 和 token_type_ids.

    Reference:
        https://github.com/PaddlePaddle/PaddleNLP/blob/a337ced850ee9f4fabc8c3f304a2f3bf9055013e/examples/text_matching/simcse/data.py#L97
    """
    input_ids = input_ids.numpy().tolist()
    token_type_ids = token_type_ids.numpy().tolist()

    batch_size, seq_len = len(input_ids), len(input_ids[0])
    repetitied_input_ids = []
    repetitied_token_type_ids = []
    rep_seq_len = seq_len
    
    for batch_id in range(batch_size):
        cur_input_id = input_ids[batch_id]
        actual_len = np.count_nonzero(cur_input_id)                                     # 去掉padding token，求句子真实长度
        dup_word_index = []
        
        if actual_len > min_dup_sentence_len_threshold:                                 # 句子太短则不进行随机重复
            dup_len = random.randint(a=0, b=max(2, int(dup_rate * actual_len)))
            dup_word_index = random.sample(list(range(1, actual_len - 1)), k=dup_len)   # 不重复[CLS]和[SEP]

        r_input_id = []
        r_token_type_id = []
        for idx, word_id in enumerate(cur_input_id):                                    # 「今天很开心」 -> 「今今天很开开心」 
            if idx in dup_word_index:
                r_input_id.append(word_id)
                r_token_type_id.append(token_type_ids[batch_id][idx])
            r_input_id.append(word_id)
            r_token_type_id.append(token_type_ids[batch_id][idx])
        after_dup_len = len(r_input_id)
        repetitied_input_ids.append(r_input_id)
        repetitied_token_type_ids.append(r_token_type_id)

        if after_dup_len > rep_seq_len:
            rep_seq_len = after_dup_len

    for batch_id in range(batch_size):                                                   # padding到最大长度
        after_dup_len = len(repetitied_input_ids[batch_id])
        pad_len = rep_seq_len - after_dup_len
        repetitied_input_ids[batch_id] += [0] * pad_len
        repetitied_token_type_ids[batch_id] += [0] * pad_len

    return torch.tensor(repetitied_input_ids).to(device), torch.tensor(repetitied_token_type_ids).to(device)
