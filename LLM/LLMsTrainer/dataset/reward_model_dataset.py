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
dataset for reward model.


Author: pankeyu
Date: 2023/07/31
"""
import random
from functools import partial

import torch
from rich import print
from datasets import load_dataset

random.seed(42)                                 


def tokenize_inputs(
    prompt: str, 
    selected: str, 
    rejected: str, 
    tokenizer,
    max_seq_len: int
):
    """
    编码偏序对。

    Args:
        prompt (str): 原始 prompt
        selected (str): 优势回答
        rejected (str): 劣势回答
        tokenizer (_type_): _description_

    Returns:
        _type_: _description_
    """
    eos_token_id = tokenizer.encode(
        tokenizer.eos_token
    )[-1]
    
    selected_input_ids = tokenizer(
        prompt + selected, 
        truncation=True,
        padding='max_length',
        max_length=max_seq_len - 1
    ).input_ids

    if selected_input_ids[-1] != eos_token_id:
        selected_input_ids += [eos_token_id]

    rejected_input_ids = tokenizer(
        prompt + rejected, 
        truncation=True,
        padding='max_length',
        max_length=max_seq_len - 1
    ).input_ids

    if rejected_input_ids[-1] != eos_token_id:
        rejected_input_ids += [eos_token_id]

    return {
        "input_ids": [
            rejected_input_ids,                         # reject 在前
            selected_input_ids                          # select 在后
        ]
    }


def collate_fn(
    batch,
    tokenizer
):
    """
    batch 选择函数。
    当 batch_size > 1 时，将 batch 内所有的 pair 都 flatten 成同一个 batch。

    Args:
        batch (_type_): _description_

    Returns:
        当 batch_size=2, max_seq_len=10: {
            'input_ids': tensor(
                [[    1,   529, 29989, 14032, 29886,   357, 29989, 29958, 29902,     2],
                [    1,   529, 29989, 14032, 29886,   357, 29989, 29958, 29902,     2],
                [    1,   529, 29989, 14032, 29886,   357, 29989, 29958,  5618,     2],
                [    1,   529, 29989, 14032, 29886,   357, 29989, 29958,  5618,     2]]
            ),
            'attention_mask': tensor(
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
            )
        }
    """
    input_ids = sum(
        [x['input_ids'] for x in batch],                # 将 (batch, 2, seq_len) 合并为 (batch, seq_len)
        []
    )

    return tokenizer.pad(
        {"input_ids": input_ids}, 
        padding=True, 
        return_tensors="pt"
    )


def construct_dataloader(
    dataset_config, 
    tokenizer
):
    all_data_files = list(
        dataset_config['reward_model_datasets'].values()
    )

    dataset = load_dataset(
        "json", 
        split="train", 
        data_files=all_data_files
    )

    dataset = dataset.shuffle(seed=42)

    tokenized_dataset = dataset.map(
        tokenize_inputs, 
        input_columns=[
            "prompt", 
            "selected", 
            "rejected"
        ], 
        fn_kwargs={
            'tokenizer': tokenizer, 
            'max_seq_len':
            dataset_config['seq_length']
        },
        desc='Tokenizing',
        num_proc=dataset_config['dataset_map_num_proc']
    )

    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset, 
        batch_size=dataset_config['batch_size'],
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    return dataloader


if __name__ == '__main__':
    from transformers import LlamaTokenizer
    
    dataset_config = {
        "reward_model_datasets": {
            "rm_oa_hh": "/mnt/bn/pankeyu/mlx/users/pankeyu/playground/LLMsTrainer/data/reward_model_data/rm_oa_hh.jsonl",
        },
        "seq_length": 10,
        "batch_size": 2,
        "dataset_map_num_proc": 16,
    }

    tokenizer = LlamaTokenizer.from_pretrained(
        '/mlx/users/pankeyu/playground/backbones/llama7b-v2-avg-plus',
        trust_remote_code=True
    )

    train_loader = construct_dataloader(
        dataset_config,
        tokenizer
    )

    for batch in train_loader:
        print(batch)
        exit()
