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

Test MBTI for LLMs.

Author: pankeyu
Date: 2023/07/19
"""
import json

import torch
import numpy as np
from tqdm import tqdm


SAVE_PATH = 'llms_mbti.json'

mbti_questions = json.load(
    open('mbti_questions.json', 'r', encoding='utf8')
)

few_shot_examples = [
    # "你是一个很内向的人，你倾向于一些务实的工作，并且很喜欢进行思考与制定计划。"      # explicit prompt
    # "你更倾向于？\nA.一个人呆着\nB.和朋友们一起\n答案：A",                      # inexplicit prompt
    # "你做事时更倾向于？\nA.靠逻辑\nB.靠感觉\n答案：A",
    # "当你准备做一件事的时候，你会选择？\nA.提前计划\nB.边做边计划\n答案：A",
    "以下哪种灯亮起之后代表可以通行？\nA.红灯\nB.绿灯\n答案：B",
    "下列哪个是人类居住的星球？\nA.地球\nB.月球\n答案：A",
    "人工智能可以拥有情感吗？\nA.可以\nB.不可以\n答案：A",
]


def encode_without_bos_eos_token(
        sentence: str, 
        tokenizer
    ):
    """
    去掉 encode 结果中的 bos 和 eos token。
    """
    token_ids = tokenizer.encode(sentence)
    if tokenizer.bos_token_id is not None:
        token_ids = [token_id for token_id in token_ids if token_id != tokenizer.bos_token_id]
    if tokenizer.eos_token_id is not None:
        token_ids = [token_id for token_id in token_ids if token_id != tokenizer.eos_token_id]
    return token_ids


def get_model_answer(
        model,
        tokenizer,
        question: str,
        options: list
    ):
    """
    输入题目，解析出模型最大概率的答案。

    Args:
        options (list[str]): 题目的所有候选项, e.g. -> ['A', 'B', 'C', 'D']
    """
    full_question = '\n\n'.join(few_shot_examples) + '\n\n' + question

    inputs = tokenizer(full_question, return_tensors='pt')['input_ids']
    if inputs[0][-1] == tokenizer.eos_token_id:
        raise ValueError('Need to set `add_eos_token` in tokenizer to false.')
    
    inputs = inputs.cuda()

    with torch.no_grad():
        logits = model(inputs).logits
        assert logits.shape[0] == 1
        logits = logits[0][-1].flatten()

        choices = [logits[encode_without_bos_eos_token(option, tokenizer)[0]] for option in options]
        probs = (
            torch.nn.functional.softmax(
                torch.tensor(choices, dtype=torch.float32), 
                dim=-1
            ).detach().cpu().numpy()
        )
        
        answer = dict([
            (i, option) for i, option in enumerate(options)
        ])[np.argmax(probs)]
        
        return answer


def get_model_examing_result(
    model,
    tokenizer
):
    """
    get all answers of LLMs.
    """
    cur_model_score = {
        'E': 0,
        'I': 0,
        'S': 0,
        'N': 0,
        'T': 0,
        'F': 0,
        'J': 0,
        'P': 0
    }

    for question in tqdm(mbti_questions.values()):
        res = get_model_answer(
            model,
            tokenizer,
            question['question'],
            ['A', 'B']
        )
        mbti_choice = question[res]
        cur_model_score[mbti_choice] += 1

    e_or_i = 'E' if cur_model_score['E'] > cur_model_score['I'] else 'I'
    s_or_n = 'S' if cur_model_score['S'] > cur_model_score['N'] else 'N'
    t_or_f = 'T' if cur_model_score['T'] > cur_model_score['F'] else 'F'
    j_or_p = 'J' if cur_model_score['J'] > cur_model_score['P'] else 'P'

    return {
        'details': cur_model_score,
        'res': ''.join([e_or_i, s_or_n, t_or_f, j_or_p])
    }


if __name__ == '__main__':
    from rich import print
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        LlamaTokenizer
    )

    # * 设定待测试的模型 & tokenizer
    models = [
        'baichuan-inc/Baichuan-7B',
        'bigscience/bloom-7b1',
    ]

    tokenizers = [
        'baichuan-inc/Baichuan-7B',
        'bigscience/bloom-7b1',
    ]

    model, tokenizer, llms_mbti = None, None, {}
    for model_path, tokenizer_path in zip(models, tokenizers):
        
        print('Model: ', model_path)
        print('Tokenizer: ', tokenizer_path)

        if model: 
            del model
        
        if tokenizer: 
            del tokenizer

        if 'llama' in tokenizer_path:
            tokenizer = LlamaTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            trust_remote_code=True
        )

        mbti_res = get_model_examing_result(
            model,
            tokenizer
        )

        llms_mbti[model_path.split('/')[-1]] = mbti_res
        json.dump(llms_mbti, open(SAVE_PATH, 'w', encoding='utf8'))
    
    print(f'[Done] Result has saved at {SAVE_PATH}.')