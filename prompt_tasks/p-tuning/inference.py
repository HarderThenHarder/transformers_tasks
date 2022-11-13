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

from transformers import AutoTokenizer, AutoModelForMaskedLM
from utils import convert_inputs, convert_logits_to_ids

from rich import print

device = "cuda:1"
model_path = "checkpoints/comment_classify/model_best"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)
model.to(device).eval()


def inference(contents: List[str]):
    """
    推理函数，输入原始句子，输出mask label的预测值。

    Args:
        contents (List[str]): 描原始句子列表。
    """
    with torch.no_grad():
        start_time = time.time()
        tokenized_output = convert_inputs(
            contents, tokenizer, max_seq_len=128, label_length=2
        )
        logits = model(
            input_ids=tokenized_output["input_ids"].to(device),
            token_type_ids=tokenized_output["token_type_ids"].to(device),
        ).logits
        predictions = convert_logits_to_ids(logits, tokenized_output["mask_positions"])
        label_tokens = []

        for prediction in predictions.detach().cpu().numpy().tolist():
            print(prediction)
            label_tokens.append("".join(tokenizer.convert_ids_to_tokens(prediction)))
        used = time.time() - start_time

        print(f"Used {used}s.")
        return label_tokens


if __name__ == "__main__":
    contents = ["苹果卖相很好，而且很甜，很喜欢这个苹果，下次还会支持的", "这破笔记本速度太慢了，卡的不要不要的"]
    res = inference(contents)
    print("inference label(s):", res)
