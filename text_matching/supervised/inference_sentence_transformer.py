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

利用训练好的模型做inference。

Author: pankeyu
Date: 2022/11/08
"""
import os
import json
import time

import torch
from transformers import AutoTokenizer

from rich import print

device = 'cuda:0'                                                        # 指定GPU设备
saved_model_path = './checkpoints/comment_classify/sentence_transformer/model_best/'     # 训练模型存放地址
tokenizer = AutoTokenizer.from_pretrained(saved_model_path) 
model = torch.load(os.path.join(saved_model_path, 'model.pt'))
model.to(device).eval()
type_desc_dict = json.load(open('embeddings/comment_classify/sentence_transformer_type_embeddings.json', 'r', encoding='utf8'))


def inference(desc: str):
    """
    类型推理函数，输出判断为「匹配」的结果，以及对应结果的匹配值（越大越匹配）。

    Args:
        desc (str): 实体长描述。
    """
    start_time = time.time()

    encoded_inputs = tokenizer(
                    text=[desc],
                    truncation=True,
                    max_length=256,
                    return_tensors='pt',
                    padding='max_length')

    doc_type, doc_embeddings = [], []
    for _type, _type_value in type_desc_dict.items():
        doc_type.append(_type_value['label'])
        doc_embeddings.append(_type_value['embedding'])
    doc_embeddings = torch.tensor(doc_embeddings).unsqueeze(dim=0)

    logits = model(query_input_ids=encoded_inputs['input_ids'].to(device),
                    query_token_type_ids=encoded_inputs['token_type_ids'].to(device),
                    query_attention_mask=encoded_inputs['attention_mask'].to(device),
                    doc_embeddings=doc_embeddings.to(device))                           # (1, doc_embeddings_num, 2)
    logits = logits.squeeze()                                                           # (doc_embeddings_num, 2)
    infrence_types = []
    for i, logit in enumerate(logits):
        if logit[1] > logit[0]:
            infrence_types.append((doc_type[i], logit[1].detach().cpu().item()))
    infrence_types.sort(key=lambda x: x[1], reverse=True)

    used = time.time() - start_time
    print(f'Used {used}s.')

    return infrence_types


if __name__ == '__main__':
    types = inference('这个破笔记本卡的不要不要的，差评')
    print(types)