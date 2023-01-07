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
Date: 2023/01/04
"""
import json
import traceback

import numpy as np


def convert_example(examples: dict, tokenizer, max_source_seq_len: int, max_target_seq_len: int):
    """
    将样本数据转换为模型接收的输入数据。

    Args:
        examples (dict): 训练数据样本, e.g. -> {
                                                "text": [
                                                            '{"context": "年基准利率4.35%。从实际看...", "answer": "年基准利率4.35%", "question": "2017年银行贷款基准利率", "id": 0}',
                                                            ...
                                                ]
                                            }
        max_source_seq_len (int): encoder 最大输入长度
        max_target_seq_len (int): decoder 最大输入长度

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'input_ids': [[1525, 10, ...], [758, 2345, ...]], 
                            'attention_mask': [[1, 1, ...], [1, 1, ...]],
                            'decoder_input_ids': [[0, 822, ...], [0, 10, ...]],
                            'labels': [[822, 10, ...], [125, 58...]]
                        }
    """
    tokenized_output = {
            'input_ids': [],                                # encoder 输入
            'attention_mask': [],                           # encoder attention mask
            'decoder_input_ids': [],                        # decoder 输入（right shift）
            'labels': []                                    # decoder 标签
        }

    for example in examples['text']:
        try:
            example = json.loads(example)
            context = example["context"]
            question = example["question"]
            answer = example["answer"]

            input_seq = f'问题：{question}{tokenizer.sep_token}原文：{context}'
            output_seq = f'答案：{answer}{tokenizer.eos_token}'
            
            output_ids = tokenizer.encode(                                             # 处理 decoder 输入
                text=output_seq,
                truncation=True,
                max_length=max_target_seq_len
            )
            decoder_input_ids = output_ids[:-2]                                         # bert-tokenizer 会加一个[CLS]，这个就当成<eos>了，因为是 right-shift
                                                                                        # 所以要-1，又因为 bert-tokenizer会加一个[SEP]，所以要-2
            decoder_input_ids = decoder_input_ids + [tokenizer.pad_token_id] * (max_target_seq_len - len(decoder_input_ids))
            lables = output_ids[1:-1]                                                   # 去掉 [CLS] 和 [SEP]
            lables = lables + [-100] * (max_target_seq_len - len(lables))                # -100 用于忽略在计算 label loss 时忽略 padding token
            
            inputs = tokenizer(                                                         # 处理 encoder 输入
                text=input_seq,
                truncation=True,
                max_length=max_source_seq_len,
                padding='max_length'
            )
        except:
            print(f'"{example}" -> {traceback.format_exc()}')
            continue

        tokenized_output['input_ids'].append(inputs["input_ids"])
        tokenized_output['attention_mask'].append(inputs["attention_mask"])
        tokenized_output['decoder_input_ids'].append(decoder_input_ids)
        tokenized_output['labels'].append(lables)
    
    for k, v in tokenized_output.items():
        tokenized_output[k] = np.array(v)

    return tokenized_output


if __name__ == '__main__':
    from rich import print
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("uer/t5-small-chinese-cluecorpussmall")
    tokenizer.eos_token = tokenizer.sep_token
    tokenizer.bos_token = tokenizer.cls_token

    res = convert_example({
                "text": [
                    "{\"context\": \"年基准利率4.35%。从实际看\", \"answer\": \"年基准利率4.35%\", \"question\": \"2017年银行贷款基准利率\"}"
                ]
            },
            tokenizer=tokenizer,
            max_source_seq_len=50,
            max_target_seq_len=20
    )
    print(res)
    print('input_ids: ', tokenizer.convert_ids_to_tokens(res['input_ids'][0]))
    print('decoder_input_ids: ', tokenizer.convert_ids_to_tokens(res['decoder_input_ids'][0]))
    print('labels: ', tokenizer.convert_ids_to_tokens(res['labels'][0]))
