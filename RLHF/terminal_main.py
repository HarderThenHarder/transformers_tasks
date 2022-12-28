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

Reinforcemnet Learning from Human Feedback 终端版。

Author: pankeyu
Date: 2022/12/27
"""
import random
from rich import print

import torch
import torch.nn.functional as F
from trl.ppo import PPOTrainer
from trl.gpt2 import GPT2HeadWithValueModel
from transformers import AutoTokenizer
from transformers import top_k_top_p_filtering


MODEL_CONFIG = {
    'model_name': 'uer/gpt2-chinese-cluecorpussmall',
    'device': 'cuda:0'
}


def respond_to_batch(model, queries, txt_len=20, top_k=0, top_p=1.0):
    """
    根据prompt生成答案。

    Args:
        model (_type_): _description_
        queries (_type_): _description_
        txt_len (int, optional): _description_. Defaults to 20.
        top_k (int, optional): _description_. Defaults to 0.
        top_p (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    input_ids = queries
    device = MODEL_CONFIG['device']
    for _ in range(txt_len):
        outputs = model(input_ids.to(device))
        next_token_logits = outputs[0][:, -1, :]
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        input_ids = torch.cat([input_ids.to(device), next_token.unsqueeze(-1)], dim=-1).cpu()
    return input_ids[:, -txt_len:]


def main():
    """
    主函数。
    """
    print('[blue]' + '#' * 53)
    print('[blue]| Reinforcemnet Learning from Human Feedback 终端版 |')
    print('[blue]' + '#' * 53)

    print('[green]Model Config: ')
    print(MODEL_CONFIG)

    print('[red][*] Initializing Model & Env...')
    model = GPT2HeadWithValueModel.from_pretrained(MODEL_CONFIG['model_name']).to(MODEL_CONFIG['device'])
    ref_model = GPT2HeadWithValueModel.from_pretrained(MODEL_CONFIG['model_name']).to(MODEL_CONFIG['device'])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
    ppo_config = {'batch_size': 1, 'forward_batch_size': 1}
    ppo_trainer = PPOTrainer(
        model,
        ref_model,
        tokenizer,
        **ppo_config
    )
    print('[green][+] Initializing Model & Env...Done.')

    prompts = [
            '刚收到货，感觉',
            '这部电影很',
            '说实话，真的很',
            '这次购物总的来说体验很'
        ]
    print(f'[yellow][+] Prompt池: {prompts}.')

    step = 1
    while True:
        current_prompt = random.choice(prompts)
        print(f'\n[Step {step}]')
        print(f'[blue]current prompt>>>[/blue] {current_prompt}')
        print('generating results...', end='\r')
        query_tensor = tokenizer.encode(current_prompt, return_tensors="pt").to(MODEL_CONFIG['device'])
        response_tensor = respond_to_batch(model, query_tensor)
        response_txt = tokenizer.decode(response_tensor[0, :].to('cpu'))

        print(f'[green]current result>>>[/green] {response_txt}')
        reward_txt = input('Reward (0.0 ~ 5.0): ')
        reward = [torch.tensor(float(reward_txt)).to(MODEL_CONFIG['device'])]

        print('Updating Model...')
        train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0].to(MODEL_CONFIG['device'])], reward)
        step += 1


if __name__ == '__main__':
    main()
