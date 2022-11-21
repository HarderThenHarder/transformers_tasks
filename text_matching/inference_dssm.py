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
Date: 2022/10/26
"""
import os
import json
import time

import torch
from rich import print
from transformers import AutoTokenizer

device = 'cuda:0'                                                        # 指定GPU设备
saved_model_path = './checkpoints/comment_classify/dssm/model_best/'     # 训练模型存放地址
tokenizer = AutoTokenizer.from_pretrained(saved_model_path) 
model = torch.load(os.path.join(saved_model_path, 'model.pt'))
model.to(device).eval()
type_desc_dict = json.load(open('embeddings/comment_classify/dssm_type_embeddings.json', \
                            'r', encoding='utf8'))                       # type embedding 存放路径


def inference(desc: str):
    """
    类型推理函数，输出每个类别的匹配概率。

    Args:
        desc (str): 评论文本。
    """
    start_time = time.time()

    encoded_inputs = tokenizer(
                    text=desc,
                    truncation=True,
                    max_length=256,
                    return_tensors='pt',
                    padding='max_length')
    desc_embedding = model(input_ids=encoded_inputs['input_ids'].to(device),
                            token_type_ids=encoded_inputs['token_type_ids'].to(device),
                            attention_mask=encoded_inputs['attention_mask'].to(device))
    types_similarity = []
    for _type, _type_value in type_desc_dict.items():
        type_embedding = torch.tensor(_type_value['embedding']).unsqueeze(dim=0).to(device)
        similarity = torch.nn.functional.cosine_similarity(type_embedding, desc_embedding)
        types_similarity.append((_type_value['label'], similarity.detach().cpu().item()))
    types_similarity.sort(key=lambda x: x[1], reverse=True)
    
    used = time.time() - start_time
    print(f'Used {used}s.')
    return types_similarity


if __name__ == '__main__':
    types = inference('这个破笔记本卡的不要不要的，差评')
    print(types)