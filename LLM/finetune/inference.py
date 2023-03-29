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

inference 训练好的模型。

Author: pankeyu
Date: 2023/03/17
"""
import torch
from peft import PeftModel, PeftConfig

from transformers import AutoTokenizer
from modeling_chatglm import ChatGLMForConditionalGeneration

torch.set_default_tensor_type(torch.cuda.HalfTensor)


device = 'cuda:0'
max_new_tokens = 300
peft_model_path = "checkpoints/model_1000"
config = PeftConfig.from_pretrained(peft_model_path)

model = ChatGLMForConditionalGeneration.from_pretrained(
    "THUDM/chatglm-6b", 
    trust_remote_code=True
).to(device)

model = PeftModel.from_pretrained(
    model, 
    peft_model_path
)

tokenizer = AutoTokenizer.from_pretrained(
    config.base_model_name_or_path, 
    trust_remote_code=True
)


def inference(
    instuction: str,
    sentence: str
    ):
    """
    模型 inference 函数。

    Args:
        instuction (str): _description_
        sentence (str): _description_

    Returns:
        _type_: _description_
    """
    with torch.no_grad():
        input_text = f"Instruction: {instuction}\n"
        if sentence:
            input_text += f"Input: {sentence}\n"
        input_text += f"Answer: "
        batch = tokenizer(input_text, return_tensors="pt")
        out = model.generate(
            input_ids=batch["input_ids"].to(device),
            attention_mask=torch.ones_like(batch["input_ids"]).bool().to(device),
            max_new_tokens=max_new_tokens,
            temperature=0
        )
        out_text = tokenizer.decode(out[0])
        answer = out_text.split('Answer: ')[-1]
        return answer


if __name__ == '__main__':
    from rich import print

    samples = [
        {
            'instruction': "你现在是一个很厉害的阅读理解器，严格按照人类指令进行回答。",
            "input": "下面句子包含了哪些三元组，只用json的格式回答：\n\n《文织召唤帖·九尾狐篇》是连载于晋江文学城的近代现代小说，作者是神隐黑猫。",
        },
        {
            'instruction': "你现在是一个很厉害的阅读理解器，严格按照人类指令进行回答。",
            "input": "下面子中的主语是什么类别，输出成列表形式。\n\n为什么不认真的检查一下， 发这么一本脏脏的书给顾客呢！\n",
        }
    ]

    for sample in samples:
        res = inference(
            sample['instruction'],
            sample['input']
        )
        print(res)