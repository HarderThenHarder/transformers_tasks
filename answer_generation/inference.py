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

测试训练好的模型效果。

Author: pankeyu
Date: 2023/01/05
"""
from rich import print
from transformers import AutoTokenizer, T5ForConditionalGeneration

device = 'cuda:0'
max_source_seq_len = 256
tokenizer = AutoTokenizer.from_pretrained('./checkpoints/DuReaderQG/model_best/')
model = T5ForConditionalGeneration.from_pretrained('./checkpoints/DuReaderQG/model_best/')
model.to(device).eval()


def inference(qustion: str, context: str):
    """
    inference函数。

    Args:
        qustion (str): 问题
        context (str): 原文
    """
    input_seq = f'问题：{question}{tokenizer.sep_token}原文：{context}'
    inputs = tokenizer(
        text=input_seq,
        truncation=True,
        max_length=max_source_seq_len,
        padding='max_length',
        return_tensors='pt'
    )
    outputs = model.generate(input_ids=inputs["input_ids"].to(device))
    output = tokenizer.decode(outputs[0].cpu().numpy(), skip_special_tokens=True).replace(" ", "")
    print(f'Q: "{qustion}"')
    print(f'C: "{context}"')
    print(f'A: "{output}"')


if __name__ == '__main__':
    question = '治疗宫颈糜烂的最佳时间'
    context = '专家指出，宫颈糜烂治疗时间应选在月经干净后3-7日，因为治疗之后宫颈有一定的创面，如赶上月经期易发生感染。因此患者应在月经干净后3天尽快来医院治疗。同时应该注意，术前3天禁同房，有生殖道急性炎症者应治好后才可进行。'
    inference(qustion=question, context=context)