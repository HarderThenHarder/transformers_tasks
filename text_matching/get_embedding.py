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

将需要事先计算的embedding计算好存放到文件中。

Author: pankeyu
Date: 2022/10/26
"""
import os
import json

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

text_file = 'data/comment_classify/types_desc.txt'                       # 候选文本存放地址
output_file = 'embeddings/comment_classify/dssm_type_embeddings.json'    # embedding存放地址

device = 'cuda:0'                                                        # 指定GPU设备
model_type = 'dssm'                                                      # 使用DSSM还是Sentence Transformer
saved_model_path = './checkpoints/comment_classify/dssm/model_best/'     # 训练模型存放地址
tokenizer = AutoTokenizer.from_pretrained(saved_model_path) 
model = torch.load(os.path.join(saved_model_path, 'model.pt'))
model.to(device).eval()


def forward_embedding(type_desc: str) -> torch.tensor:
    """
    将输入喂给encoder并得到对应的embedding向量。

    Args:
        type_desc (_type_): 输入文本

    Returns:
        torch.tensor: (768,)
    """
    encoded_inputs = tokenizer(
                    text=type_desc,
                    truncation=True,
                    max_length=256,
                    return_tensors='pt',
                    padding='max_length')
    if model_type == 'dssm':
        embedding = model(input_ids=encoded_inputs['input_ids'].to(device),
                            token_type_ids=encoded_inputs['token_type_ids'].to(device),
                            attention_mask=encoded_inputs['attention_mask'].to(device))
    elif model_type == 'sentence_transformer':
        embedding = model.get_embedding(input_ids=encoded_inputs['input_ids'].to(device),
                            token_type_ids=encoded_inputs['token_type_ids'].to(device),
                            attention_mask=encoded_inputs['attention_mask'].to(device))
    else:
        raise ValueError('@param model_type must in ["dssm", "sentence_transformer"].')
    return embedding.detach().cpu().numpy()[0].tolist()


def extract_embedding(use_embedding=True):
    """
    获得type_embedding文件中存放的所有文本的embedding并存放到本地。

    Args:
        use_embedding (bool, optional): _description_. Defaults to True.
        model_type (str): 使用哪种模型结构
    """
    type_embedding_dict = {}
    with open(text_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            res = line.strip().split('\t')
            if len(res) == 3:
                _type, label, desc = res
            elif len(res) == 2:
                _type, label = res
                desc = ''
            type_desc = f'{label}：{desc}'
            
            if use_embedding:
                type_embedding = forward_embedding(type_desc)
            else:
                type_embedding = []
            
            type_embedding_dict[_type] = {
                'label': label,
                'text': type_desc,
                'embedding': type_embedding
            }
    json.dump(type_embedding_dict, open(output_file, 'w', encoding='utf8'), ensure_ascii=False)


if __name__ == '__main__':
    extract_embedding(use_embedding=True)