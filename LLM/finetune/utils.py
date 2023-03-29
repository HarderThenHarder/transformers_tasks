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


def get_masks_and_position_ids(
        input_ids_with_padded, 
        max_length, 
        gmask=False, 
        position_encoding_2d=True
    ):
    """
    生成 mask_attention 和 position_ids。

    Args:
        input_ids_with_padded (_type_): _description_
        prompt_len (_type_): _description_
        tokenizer (_type_): _description_
        max_length (_type_): _description_
        gmask (bool, optional): _description_. Defaults to False.
        position_encoding_2d (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    mask_position = input_ids_with_padded.index(150001)             # 150001 -> [gMask]
    attention_mask = np.ones((1, max_length, max_length))
    attention_mask = np.tril(attention_mask)
    attention_mask[..., :mask_position - 1] = 1
    attention_mask = attention_mask < 0.5                           # convert to True or False

    if position_encoding_2d:
        seq_length = input_ids_with_padded.index(150004)            # 150004 -> <sop>
        position_ids = np.arange(max_length)
        if not gmask:
            position_ids[seq_length:] = mask_position               # 从<sop>开始，后面所有的position id都变成[gMask]的position
        block_position_ids = np.concatenate(
            (
                np.zeros(seq_length),
                np.arange(
                    max_length - seq_length
                ) + 1,
            )                                                       # 从<sop>开始（等于1），后面字符位置编码依次 +1
        )
        position_ids = np.stack((position_ids, block_position_ids), axis=0)
    else:
        position_ids = np.arange(max_length)
        if not gmask:
            position_ids[max_length - 1:] = mask_position
    
    return attention_mask, position_ids


def convert_example(
        examples: dict, 
        tokenizer, 
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
            'attention_mask': [],
            'position_ids': [],
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
            input_ids = prompts_ids + target_ids + [tokenizer.eos_token_id] * 2

            prompts_length = len(prompts_ids)
            labels = (
                [-100] * (prompts_length - 1) + 
                input_ids[prompts_length - 1:] + 
                [tokenizer.eos_token_id] + 
                [-100] * (max_seq_len - len(input_ids) - 1)
            )

            input_ids_with_padded = input_ids + [tokenizer.eos_token_id] * (max_seq_len - len(input_ids))

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
            attention_mask, position_ids = get_masks_and_position_ids(
                input_ids_with_padded=input_ids_with_padded,
                max_length=max_seq_len,
                gmask=False,
                position_encoding_2d=True
            )
        except:
            print(f'"{example}" -> {traceback.format_exc()}')
            continue

        tokenized_output['input_ids'].append(input_ids_with_padded)
        tokenized_output['attention_mask'].append(attention_mask)
        tokenized_output['position_ids'].append(position_ids)
        tokenized_output['labels'].append(labels)
    
    for k, v in tokenized_output.items():
        tokenized_output[k] = np.array(v)

    return tokenized_output


def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids) + 1
    
    print('longest: ', longest)
    
    input_ids = []
    attention_mask_list = []
    position_ids_list = []
    labels_list = []

    for ids_l, feature in zip(len_ids, features):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        
        labels = (
            [-100] * (seq_len - 1)
            + ids[(seq_len - 1) :]
            + [tokenizer.eos_token_id]
            + [-100] * (longest - ids_l - 1)
        )

        # print('input_ids: ', tokenizer.convert_ids_to_tokens(ids))
        print('labels: ', tokenizer.convert_ids_to_tokens(labels))

        ids = ids + [tokenizer.eos_token_id] * (longest - ids_l)
        _ids = ids

        print('padding input_ids: ', tokenizer.convert_ids_to_tokens(_ids))

        attention_mask, position_ids = get_masks_and_position_ids(
            ids, 
            longest, 
            gmask=False
        )
        labels_list.append(labels)
        input_ids.append(_ids)
        attention_mask_list.append(attention_mask)
        position_ids_list.append(position_ids)
    input_ids = input_ids
    labels = labels_list
    attention_mask = attention_mask_list
    position_ids = position_ids_list
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


if __name__ == '__main__':
    from rich import print
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    context = "测试一个输入"
    target = "测试输出"

    print('-' * 100)
    examples = {
        "text": [
                    '{"context": "测试一个输入", "target": "测试输出"}',
        ]
    }

    tokenized_output = convert_example(
        examples=examples,
        tokenizer=tokenizer,
        max_source_seq_len=5,
        max_target_seq_len=5
    )

    print(tokenized_output)
    print('inputs: ')
    print(tokenizer.convert_ids_to_tokens(tokenized_output['input_ids'][0]))
    print('labels: ')
    print(tokenizer.convert_ids_to_tokens(tokenized_output['labels'][0]))

    print('-' * 100)

    prompt_ids = tokenizer.encode(
            context,
            max_length=5,
            truncation=True
        )
    target_ids = tokenizer.encode(
        target, 
        max_length=5, 
        truncation=True, 
        add_special_tokens=False
    )
    featrues = [{
        'input_ids': prompt_ids + target_ids + [tokenizer.eos_token_id] * 2,
        'seq_len': len(prompt_ids)
    }]
    res = data_collator(features=featrues)
    print('data collator res: ', res)