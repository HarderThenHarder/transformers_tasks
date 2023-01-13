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
import traceback
import numpy as np


def convert_pointwise_example(examples: dict, tokenizer, max_seq_len: int):
    """
    将样本数据转换为模型接收的输入数据。

    Args:
        examples (dict): 训练数据样本, e.g. -> {
                                                "text": [
                                                            '今天天气好吗	今天天气怎样	1',
                                                            '今天天气好吗	胡歌结婚了吗	0',
                                                            ...
                                                ]
                                            }

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'input_ids': [[101, 3928, ...], [101, 4395, ...]], 
                            'token_type_ids': [[0, 0, ...], [0, 0, ...]],
                            'position_ids': [[0, 1, 2, ...], [0, 1, 2, ...]],
                            'attention_mask': [[1, 1, ...], [1, 1, ...]],
                            'labels': [1, 0, ...]
                        }
    """
    tokenized_output = {
        'input_ids': [], 
        'token_type_ids': [],
        'position_ids': [],
        'attention_mask': [],
        'labels': []
    }

    for example in examples['text']:
        try:
            query, doc, label = example.split('\t')
            encoded_inputs = tokenizer(
                text=query,
                text_pair=doc,
                truncation=True,
                max_length=max_seq_len,
                padding='max_length')
        except:
            print(f'"{example}" -> {traceback.format_exc()}')
            exit()

        tokenized_output['input_ids'].append(encoded_inputs["input_ids"])
        tokenized_output['token_type_ids'].append(encoded_inputs["token_type_ids"])
        tokenized_output['position_ids'].append([i for i in range(len(encoded_inputs["input_ids"]))])
        tokenized_output['attention_mask'].append(encoded_inputs["attention_mask"])
        tokenized_output['labels'].append(int(label))
    
    for k, v in tokenized_output.items():
        tokenized_output[k] = np.array(v)

    return tokenized_output


def convert_dssm_example(examples: dict, tokenizer, max_seq_len: int):
    """
    将样本数据转换为模型接收的输入数据。

    Args:
        examples (dict): 训练数据样本, e.g. -> {
                                                "text": [
                                                            '今天天气好吗	今天天气怎样	1',
                                                            '今天天气好吗	胡歌结婚了吗	0',
                                                            ...
                                                ]
                                            }

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'query_input_ids': [[101, 3928, ...], [101, 4395, ...]], 
                            'query_token_type_ids': [[0, 0, ...], [0, 0, ...]],
                            'query_attention_mask': [[1, 1, ...], [1, 1, ...]],
                            'doc_input_ids': [[101, 2648, ...], [101, 3342, ...]], 
                            'doc_token_type_ids': [[0, 0, ...], [0, 0, ...]],
                            'doc_attention_mask': [[1, 1, ...], [1, 1, ...]],
                            'labels': [1, 0, ...]
                        }
    """
    tokenized_output = {
        'query_input_ids': [], 
        'query_token_type_ids': [],
        'query_attention_mask': [],
        'doc_input_ids': [], 
        'doc_token_type_ids': [],
        'doc_attention_mask': [],
        'labels': []
    }

    for example in examples['text']:
        try:
            query, doc, label = example.split('\t')
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
            print(f'"{example}" -> {traceback.format_exc()}')
            exit()

        tokenized_output['query_input_ids'].append(query_encoded_inputs["input_ids"])
        tokenized_output['query_token_type_ids'].append(query_encoded_inputs["token_type_ids"])
        tokenized_output['query_attention_mask'].append(query_encoded_inputs["attention_mask"])
        tokenized_output['doc_input_ids'].append(doc_encoded_inputs["input_ids"])
        tokenized_output['doc_token_type_ids'].append(doc_encoded_inputs["token_type_ids"])
        tokenized_output['doc_attention_mask'].append(doc_encoded_inputs["attention_mask"])
        tokenized_output['labels'].append(int(label))
    
    for k, v in tokenized_output.items():
        tokenized_output[k] = np.array(v)

    return tokenized_output