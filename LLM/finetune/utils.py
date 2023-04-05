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
        config, 
        max_source_seq_len: int,
        max_target_seq_len: int,
    ):
    """
    将样本数据转换为模型接收的输入数据。

    Args:
        examples (dict): 训练数据样本, e.g. -> {
                                                "text": [
                                                            '{"context": "年基准利率4.35%。从实际看...", "answer": "年基准利率4.35%", "question": "2017年银行贷款基准利率", "id": 0}',
                                                            ...
                                                ]
                                            }
        max_source_seq_len (int): prompt最大长度
        max_target_seq_len (int): 答案最大长度

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'input_ids': [[1525, 10, ...], [758, 2345, ...]], 
                            'attention_mask': [[1, 1, ...], [1, 1, ...]],
                            'labels': [[822, 10, ...], [125, 58...]]
                        }
    """
    tokenized_output = {
            'input_ids': [],
            'labels': []
        }
    
    for example in examples['text']:
        try:
            example = json.loads(example)
            context = example["context"]
            target = example["target"]

            prompts_ids = tokenizer.encode(
                text=context,
                max_length=max_source_seq_len,
                truncation=True
            )

            target_ids = tokenizer.encode(
                text=target,
                max_length=max_target_seq_len,
                truncation=True,
                add_special_tokens=False                    # 设置为False，则会 -> [_, token1, token2, ...], 
            )                                               # 否则为 -> [_, token1, token2, [gMASK], <sop>]

            max_seq_len = max_source_seq_len + max_target_seq_len + 1
            input_ids = prompts_ids + target_ids + [config.eos_token_id]

            prompts_length = len(prompts_ids)
            labels = (
                [-100] * (prompts_length - 1) + 
                input_ids[prompts_length - 1:] + 
                [-100] * (max_seq_len - len(input_ids))
            )

            input_ids_with_padded = input_ids + [tokenizer.pad_token_id] * (max_seq_len - len(input_ids))

            """
            只有 target 部分的需要写 label，padding 和 prompts 不分不需要写，
            
            因此，需要进行错位，以 
                
                prompt = '测试输入'
                target = '测试输出'
                max_source_seq_len=5,
                max_target_seq_len=5

            
            为例，错位后的 input 和 label 如下：

                input       ->        label
                ▁           ->        <image_-100>
                测试         ->        <image_-100>
                输入         ->        <image_-100>
                [gMASK]     ->        <image_-100>
                <sop>       ->        _
                ▁           ->         测试
                测试         ->         输出
                输出         ->         </s>
                </s>        ->         <image_-100>
                <pad>       ->         <image_-100>
            
            可运行下面代码打出结果：  

            print(f'input_token -> label')
            for input_token, label_token in zip(tokenizer.convert_ids_to_tokens(input_ids_with_padded),
                                                tokenizer.convert_ids_to_tokens(labels)):
                print(f'{input_token} -> {label_token}')
            """
        except:
            print(f'"{example}" -> {traceback.format_exc()}')
            continue

        tokenized_output['input_ids'].append(input_ids_with_padded)
        tokenized_output['labels'].append(labels)
    
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