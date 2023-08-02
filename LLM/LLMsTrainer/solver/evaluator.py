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

LLM 评估器，包括 NLL、单选题、生成任务等测试。

Authors: pankeyu
Date: 2023/05/25
"""
import json

import torch
import numpy as np


def eval_generation(
        model,
        tokenizer,
        device,
        test_datasets
    ):
    """
    测试当前模型的生成效果。
    """
    res_list = []
    with torch.no_grad():
        for name, path in test_datasets.items():
            with open(path, 'r', encoding='utf8') as f:
                for line in f:
                    line = json.loads(line)
                    raw_inputs = line['content']
                    inputs = tokenizer(
                        raw_inputs,
                        return_tensors="pt",
                        add_special_tokens=False,
                        return_attention_mask=False,
                        return_token_type_ids=False
                    )
                    input_length = inputs["input_ids"].shape[1]
                    
                    for k, v in inputs.items():
                        inputs[k] = v.to(device)
                    
                    pred = model.generate(
                        **inputs, 
                        max_new_tokens=100, 
                        repetition_penalty=1.0,
                        do_sample=False,
                    )
                    
                    pred = pred[0, input_length:]
                    pred = tokenizer.decode(pred.cpu(), skip_special_tokens=True)
                    res_list.append({
                        'test_input': raw_inputs,
                        'test_ouput': pred,
                        'source_file': name
                    })
    return res_list


def eval_nll(
        model,
        tokenizer,
        device,
        nll_files,
        seq_length
    ):
    """
    在评测集上评测 NLL loss。
    """
    nll_loss_dict = {}
    for name, path in nll_files.items():
        with open(path, 'r', encoding='utf8') as f:
            total_nll, sample_count = 0, 0
            for line in f:
                line = json.loads(line)
                
                mask_text = ""
                if "title" in line and "content" in line:
                    text = f'{line["title"]}\n{line["content"]}'
                
                elif "content" in line and "target" in line:
                    text = f'{line["content"]}\n{line["target"]}'
                    mask_text = f'{line["content"]}\n'

                elif "content_starcoder" in line and "target" in line:
                    # text = f'{line["content_starcoder"]}\n{line["target"]}'
                    text = f'{tokenizer.user_token}\n{line["content_starcoder"]}\n{tokenizer.chat_end_token}\n{tokenizer.assistant_token}{line["target"]}\n{tokenizer.chat_end_token}'
                    mask_text = f'{tokenizer.user_token}\n{line["content_starcoder"]}\n{tokenizer.chat_end_token}\n{tokenizer.assistant_token}'
                
                elif "user" in line and "assistant" in line:
                    text = f'{line["user"]}\n{line["assistant"]}'
                    mask_text = f'{line["user"]}\n'
                
                elif "content" in line:
                    text = line['content']
                
                else:
                    raise ValueError(f'Not valid eval dataset format: {line}')
                
                input_ids = tokenizer(
                    text, 
                    return_tensors = "pt",
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    truncation=True,
                    max_length=seq_length
                )["input_ids"]
                
                target_ids = input_ids.clone()

                if mask_text:
                    mask_text_len = len(tokenizer(mask_text)['input_ids'])
                    target_ids[:, :mask_text_len] = -100
                input_ids, target_ids = input_ids.to(device), target_ids.to(device)

                with torch.no_grad():
                    outputs = model(input_ids, labels=target_ids)
                    current_nll = outputs.loss.item() * (target_ids.shape[1] - 1)

                total_nll += current_nll
                sample_count += 1
        
        nll_loss_dict[name] = total_nll / sample_count
    
    return nll_loss_dict


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


def eval_single_choice(
        model,
        tokenizer,
        device,
        single_choice_files
    ):
    """
    选择题库测试 Acc。

    Returns:
        {
            '题库1': {
                'total_question_num': 10,
                'correct_question_num': 8,
                'acc': 0.8,
                'question_details': [...]
            },
            ...
        }
    """
    def get_model_answer(
            model,
            tokenizer,
            question: str,
            options: list,
            device: str
        ):
        """
        输入题目，解析出模型最大概率的答案。

        Args:
            options (list[str]): 题目的所有候选项, e.g. -> ['A', 'B', 'C', 'D']
        """
        few_shot_examples = [
            "以下选项中，哪一个是人类居住的星球？\nA. 地球\nB. 月球\nC. 金星\nD. 木星\n答案：A",
            "当交通灯处于什么颜色时代表可以通行？\nA. 红色\nB. 绿色\n答案：B",
            "人拥有____眼睛？\nA. 一只\nB. 六只\nC. 三只\nD. 两只\n答案：D",
            "小明给小红推荐了一本书，因为他实在是太喜欢这本书了。上述句子中的“他”指的是小红吗？\nA. 是\nB. 不是\n答案：B",
            "大壮和大美结婚不久后就生下了小明，那么大壮是小明的____？\nA. 哥哥\nB. 姐姐\nC. 妈妈\nD. 爸爸\n答案：D"
        ]
        full_question = '\n\n'.join(few_shot_examples) + '\n\n' + question

        inputs = tokenizer(full_question, return_tensors='pt')['input_ids']
        if inputs[0][-1] == tokenizer.eos_token_id:
            print('Drop EOS token!')
            inputs[0] = inputs[0][:-1]
        
        inputs = inputs.to(device)

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
            answer = dict([(i, option) for i, option in enumerate(options)])[np.argmax(probs)]
            return answer

    metrics_dict = {}
    for name, path in single_choice_files.items():
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                data = json.loads(line)
                
                if data['source'] in ['winograd_wsc']:
                    options = ["A", "B"]
                else:
                    options = ["A", "B", "C", "D"]

                answer = get_model_answer(
                    model,
                    tokenizer,
                    data['question'],
                    options,
                    device
                )

                if data['source'] not in metrics_dict:
                    metrics_dict[data['source']] = {
                        'total_question_num': 0,
                        'correct_question_num': 0,
                        'acc': 0,
                        'question_details': []
                    }
                
                metrics_dict[data['source']]['total_question_num'] += 1
                if answer == data['answer']:
                    metrics_dict[data['source']]['correct_question_num'] += 1
                metrics_dict[data['source']]['acc'] = round(metrics_dict[data['source']]['correct_question_num']/metrics_dict[data['source']]['total_question_num'], 2)
                
                question_str = data["question"].replace("\n", "\\n")
                metrics_dict[data['source']]['question_details'].append(
                    f'{question_str}\t{answer}\t{data["answer"]}'
                )

    return metrics_dict


def eval_reward_model_pair_acc(
    model,
    tokenizer,
    device,
    test_reward_model_files,
    max_seq_len
):
    """
    测试reward model在验证集上的正确率。
    """

    def tokenize_inputs(
        prompt: str, 
        selected: str, 
        rejected: str, 
        tokenizer,
        max_seq_len: int
    ):
        """
        编码偏序对。

        Args:
            prompt (str): 原始 prompt
            selected (str): 优势回答
            rejected (str): 劣势回答
            tokenizer (_type_): _description_

        Returns:
            _type_: _description_
        """
        eos_token_id = tokenizer.encode(
            tokenizer.eos_token
        )[-1]
        
        selected_input_ids = tokenizer(
            prompt + selected, 
            truncation=True,
            padding='max_length',
            max_length=max_seq_len - 1
        ).input_ids

        if selected_input_ids[-1] != eos_token_id:
            selected_input_ids += [eos_token_id]

        rejected_input_ids = tokenizer(
            prompt + rejected, 
            truncation=True,
            padding='max_length',
            max_length=max_seq_len - 1
        ).input_ids

        if rejected_input_ids[-1] != eos_token_id:
            rejected_input_ids += [eos_token_id]

        return [
            rejected_input_ids,                         # reject 在前
            selected_input_ids                          # select 在后
        ]

    metrics_dict, all_delta_scores, all_scores, all_tokens = {}, [], [], []
    for name, path in test_reward_model_files.items():
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                data = json.loads(line)
                
                inputs = tokenize_inputs(
                    data['prompt'],
                    data['selected'],
                    data['rejected'],
                    tokenizer,
                    max_seq_len
                )

                input_ids = tokenizer.pad(
                    {"input_ids": inputs}, 
                    padding=True, 
                    return_tensors="pt"
                )['input_ids']

                input_ids = input_ids.to(device)
                with torch.no_grad():
                    scores = model(input_ids)[0]

                delta_scores = scores.reshape(-1, 2).diff().view(-1)
                all_delta_scores.extend(delta_scores.tolist())
                all_scores.extend(scores.view(-1).tolist())
                all_tokens.extend(input_ids.tolist())

    delta_scores = np.hstack(all_delta_scores)
    metrics_dict['delta_mean'] = delta_scores.mean()
    metrics_dict['delta_std'] = delta_scores.std()
    metrics_dict['acc'] = (delta_scores > 0).mean()
    metrics_dict['delta_scores'] = list(delta_scores)
    
    return metrics_dict