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
Date: 2023/03/26
"""
import json
import traceback

import numpy as np
from tqdm import tqdm


def convert_example(
        examples: dict, 
        tokenizer,
        max_source_seq_len: int,
        max_target_seq_len: int,
    ):
    """
    将样本数据转换为Ptuning模型接收的输入数据。

    Args:
        examples (dict): 训练数据样本, e.g. -> {
                                                "text": [
                                                            '{"context": "年基准利率4.35%。从实际看...", "target": "2017年银行贷款基准利率"}',
                                                            ...
                                                ]
                                            }
        max_source_seq_len (int): prompt最大长度
        max_target_seq_len (int): 答案最大长度

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'input_ids': [[1525, 10, ...], [758, 2345, ...]], 
                            'labels': [[822, 10, ...], [125, 58...]]
                        }
    """
    tokenized_output = {
        'input_ids': [],
        'labels': []
    }

    max_seq_length = max_source_seq_len + max_target_seq_len

    for example in examples['text']:
        try:
            example = json.loads(example)
            context = example["context"]
            target = example["target"]

            prompts_ids = tokenizer.encode(
                text=context,
                add_special_tokens=False
            )

            target_ids = tokenizer.encode(
                text=target,
                add_special_tokens=False
            )                    

            if len(prompts_ids) >= max_source_seq_len:                                          # source 需要留一个 [gMASK] token 在结尾
                prompts_ids = prompts_ids[:max_source_seq_len - 1]

            if len(target_ids) >= max_target_seq_len - 1:                                       # target 需要留一个 <sop> 在开头和一个 <eop> token 在结尾
                target_ids = target_ids[:max_target_seq_len - 2]

            input_ids = tokenizer.build_inputs_with_special_tokens(prompts_ids, target_ids)     # source_ids + [gMASK] + <sop> + target_ids + <eop>
            context_length = input_ids.index(tokenizer.bos_token_id)                            # bos 在 target 的第一位
            mask_position = context_length - 1                                                  # [gMASK] 在 source 的最后一位
            labels = [-100] * context_length + input_ids[mask_position + 1:]                    # 从 bos 开始到后面所有的 target 到 eos 都为 label

            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

            tokenized_output['input_ids'].append(input_ids)
            tokenized_output['labels'].append(labels)
        except:
            print(f'"{example}" -> {traceback.format_exc()}')
            continue

    for k, v in tokenized_output.items():
        tokenized_output[k] = np.array(v)

    return tokenized_output


def check_max_length_of_datasets(
        tokenizer,
        dataset_file: str
    ):
    """
    测试数据集最大的输入/输出tokens是多少。

    Args:
        dataset_file (str): _description_
    """
    source_seq_len_list = []
    target_seq_len_list = []
    with open(dataset_file, 'r') as f:
        for line in tqdm(f.readlines()):
            line = json.loads(line)

            source_len = tokenizer.encode(line['context'])
            source_seq_len_list.append(len(source_len))

            target_len = tokenizer.encode(line['target'])
            target_seq_len_list.append(len(target_len))

    print(dataset_file)
    print(f"【Source Sequence】 Max: {max(source_seq_len_list)}, Avg: {int(sum(source_seq_len_list) / len(source_seq_len_list))}, Middle: {sorted(source_seq_len_list)[int(len(source_seq_len_list) / 2)]}.")
    print(f"【Target Sequence】 Max: {max(target_seq_len_list)}, Avg: {int(sum(target_seq_len_list) / len(target_seq_len_list))}, Middle: {sorted(target_seq_len_list)[int(len(target_seq_len_list) / 2)]}.")