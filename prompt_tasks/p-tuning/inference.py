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

加载预训练好的模型并测试效果。

Author: pankeyu
Date: 2022/11/13
"""
import time
from typing import List

import torch
import numpy as np
from rich import print

from transformers import AutoTokenizer, AutoModelForMaskedLM
from utils import convert_example, convert_logits_to_ids

from verbalizer import Verbalizer

device = 'cuda:1'
model_path = 'checkpoints/predicate_generate/model_best'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)
model.to(device).eval()

max_label_len = 6                               # 标签最大长度
p_embedding_num = 6                             # p_token个数
verbalizer = Verbalizer(
        verbalizer_file='data/predicate_generate/verbalizer.txt',
        tokenizer=tokenizer,
        max_label_len=max_label_len
    )


def inference(contents: List[str]):
    """
    推理函数，输入原始句子，输出mask label的预测值。

    Args:
        contents (List[str]): 描原始句子列表。
    """
    with torch.no_grad():
        start_time = time.time()
        examples = {'text': contents}

        tokenized_output = convert_example(
            examples, 
            tokenizer, 
            max_seq_len=128,
            max_label_len=max_label_len,
            p_embedding_num=p_embedding_num,
            train_mode=False,
            return_tensor=True
        )

        logits = model(
            input_ids=tokenized_output['input_ids'].to(device),
            token_type_ids=tokenized_output['token_type_ids'].to(device)).logits
        
        predictions = convert_logits_to_ids(logits, tokenized_output['mask_positions']).cpu().numpy().tolist()  # (batch, label_num)
        predictions = verbalizer.batch_find_main_label(predictions)                                             # 找到子label属于的主label
        predictions = [ele['label'] for ele in predictions]
        used = time.time() - start_time
        print(f'Used {used}s.')
        return predictions


if __name__ == "__main__":
    contents = [
        "苹果卖相很好，而且很甜，很喜欢这个苹果，下次还会支持的", 
        "这破笔记本速度太慢了，卡的不要不要的"
    ]
    res = inference(contents)
    for c, r in zip(contents, res):
        print(f"{c} -> {r}")
