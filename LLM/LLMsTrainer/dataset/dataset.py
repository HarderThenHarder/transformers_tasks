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

dataset 加载，包含数据 sample、truncation、concat 等策略。

Authors: pankeyu
Date: 2023/05/30

Reference Authors: s-JoL(sl12160010@gmail.com)
Reference repo: Open-Llama
"""
import os
import math
import json
import random
from glob import glob

from tqdm import tqdm

import torch
from datasets import load_dataset

random.seed(42)                                 


def pretrain_transform_gen(tokenizer):
    def pretrain_transform(batch):
        """
        处理pretrain数据。

        batch: {
            'title': ['这是一条测试标题'],
            'content': ['这是一条测试内容'],
        } 或 {
            'text': ['这是一条合并后的数据']
        } 或 {
            'question': ['这是一个测试问题'],
            'answer': ['这是一个测试回答']
        } 或 {
            'code': ['import numpy as np']
        } 或 {
            'user': ['你好，介绍一下你自己'],
            'assistant': ['您好，我是一个人工智能助手']
        }
        """
        if "text" in batch and batch['text'][0] is not None:
            pass

        elif "content" in batch and batch['content'][0] is not None:
            if "title" in batch and batch['title'][0] is not None:                                                                           # 添加「标题」「内容」的 special token
                assert tokenizer.title_token and tokenizer.content_token, \
                    'Detect special token: `content_token` or `title_token` is None, you should specify these special tokens in tokenizer config,' \
                    ' or you should combine `title` and `content` to `text` in your jsonl dataset.'
                text = f'{tokenizer.title_token}{batch["title"][0]}{tokenizer.content_token}{batch["content"][0]}'
                batch["text"] = [text]
            else:
                batch["text"] = [batch["content"][0]]
        
        elif ("question" in batch and batch['question'][0] is not None) or ("answer" in batch and batch["answer"][0] is not None):           # 添加「问题」「答案」的 special token
            assert "question" in batch and batch['question'][0] is not None and "answer" in batch and batch["answer"][0] is not None, 'key `answer` and `question` should both in sample, '\
                'while received{}, or you can switch `question` and `answer` to another key in your dataset.'.format(batch)
            
            assert tokenizer.question_token and tokenizer.answer_token, \
                'Detect special token: `question_token` or `answer_token` is None, you should specify these special tokens in tokenizer config,' \
                ' or you should combine `question` and `answer` to `text` in your jsonl dataset.'
            
            text = f'{tokenizer.question_token}{batch["question"][0]}{tokenizer.answer_token}{batch["answer"][0]}'
            batch["text"] = [text]
        
        elif "code" in batch and batch["code"][0] is not None:                                                                               # 添加「代码」的 special token
            assert tokenizer.code_token, \
                'Detect special token: `code_token` is None, you should specify these special tokens in tokenizer config,' \
                ' or you should switch key `code` to `text` in your jsonl dataset.'
            text = f'{tokenizer.code_token}{batch["code"][0]}'
            batch["text"] = [text]
        
        elif ("user" in batch and batch["user"][0] is not None) or ("assistant" in batch and batch["assistant"][0] is not None):
            assert "user" in batch and batch["user"][0] is not None and "assistant" in batch and batch["assistant"][0] is not None,  'key `user` and `assistant` should both in sample, '\
                'while received{}, or you can switch `user` and `assistant` to another key in your dataset.'.format(batch)
            
            assert tokenizer.user_token and tokenizer.assistant_token, \
                'Detect special token: `user_token` or `assistant_token` is None, you should specify these special tokens in tokenizer config, ' \
                'or you should combine `user` and `assistant` to `text` in your jsonl dataset.'
            
            text = f'{tokenizer.user_token}{batch["user"][0]}{tokenizer.assistant_token}{batch["assistant"][0]}'
            batch["text"] = [text]
        
        else:
            raise Exception("Unrecognized pretrain dataset format.")
        
        return batch
    return pretrain_transform


def sft_transform_gen(tokenizer):
    def sft_transform(batch):
        """
        处理指令微调的数据。

        batch: {
            'context': ['你好世界的英文是什么？'],
            'target': ['Hello Word']
        }
        """
        if "context" in batch and batch["context"][0] is not None and "target" in batch and batch["target"][0] is not None:
            text = [batch["context"][0] + batch["target"][0]]
            return {
                "text": text, 
                "prompt_contents": [
                    [batch["context"][0]]
                ]
            }
        
        elif ("user" in batch and batch["user"][0] is not None) or ("assistant" in batch and batch["assistant"][0] is not None):
            assert "user" in batch and batch["user"][0] is not None and "assistant" in batch and batch["assistant"][0] is not None,  'key `user` and `assistant` should both in sample, '\
                'or you can switch `user` and `assistant` key to `context` and `content` key in your dataset.'
            
            assert tokenizer.user_token and tokenizer.assistant_token and tokenizer.chat_end_token, \
                'Detect special token: `user_token` or `assistant_token` or `chat_end_token` is None, you should specify these special tokens in tokenizer config, ' \
                'while received{}, or you can switch `user` and `assistant` key to `context` and `content` key in your dataset.'.format(batch)
            text = f'{tokenizer.user_token}\n{batch["user"][0]}\n{tokenizer.chat_end_token}\n{tokenizer.assistant_token}\n{batch["assistant"][0]}\n{tokenizer.chat_end_token}'
            return {
                "text": [text], 
                "prompt_contents": [
                    [f'{tokenizer.user_token}\n{batch["user"][0]}\n{tokenizer.chat_end_token}\n{tokenizer.assistant_token}\n']
                ]
            }
        
        elif "dialog" in batch and batch['dialog'][0] is not None:
            assert tokenizer.user_token and tokenizer.assistant_token and tokenizer.chat_end_token and tokenizer.system_token, \
                'Detect special token: `user_token` or `assistant_token` or `chat_end_token` or `system_token` is None, you should specify these special tokens in tokenizer config, ' \
                'while received{}, or you can switch `user` and `assistant` key to `context` and `content` key in your dataset.'.format(batch)
            
            assert type(batch['dialog'][0]) == list, \
                '`dialog` key in your dataset should be a list, but got {}'.format(type(batch['dialog'][0]))

            text = ''
            prompt_contents = []
            dialog = batch['dialog'][0]
            for i, message in enumerate(dialog):
                assert type(message) == dict, \
                    'value of `dialog` shoud be List[dict], it should be like {"dialog": [{"content": "you are a chatbot.", "role": "system"}, {"content": "Hello", "role": "user"}, {"content": "Hi, how are you?", "role": "assistant"}]}. But got {}'.format(message)
                
                if message['role'] == 'user':
                    cur_text = f'{tokenizer.user_token}\n{message["content"]}\n{tokenizer.chat_end_token}\n'
                    text += cur_text
                    if i < len(dialog) - 1 and dialog[i+1]['role'] == 'assistant':
                        prompt_contents.append(cur_text + tokenizer.assistant_token)
                    else:
                        prompt_contents.append(cur_text)                                            # user 部分的 loss 需要被 mask
                elif message['role'] == 'assistant':
                    cur_text = f'{tokenizer.assistant_token}\n{message["content"]}\n{tokenizer.chat_end_token}\n'
                    text += cur_text
                elif message['role'] == 'system':
                    cur_text = f'{tokenizer.system_token}\n{message["content"]}\n{tokenizer.chat_end_token}\n'
                    text += cur_text
                    if i < len(dialog) - 1 and dialog[i+1]['role'] == 'assistant':
                        prompt_contents.append(cur_text + tokenizer.assistant_token)
                    else:
                        prompt_contents.append(cur_text)                                            # system 部分的 loss 需要被 mask
                else:
                    raise ValueError(f"`role` should in ['user', 'assistant', 'system'], while received: {message['role']}.")

                text += tokenizer.chat_end_token
            
            return {
                "text": [text], 
                "prompt_contents": [prompt_contents]
            }
            
        else:
            raise Exception("Unrecognized sft dataset format.")

    return sft_transform


def sample_sequence_gen(
        seq_length, 
        eos_token_id
    ):
    """
    对超过最大长度的句子进行sample，选出其中的一段作为训练数据。
    * 优点: 能够保证短句子的完整性，增加长句子的多样性（而不是所有长文章都选择最前面的一段）
    * 缺点: 随机sample文章中的一段并不一定能保证截取的片段一定是通顺的，有可能从句子的中间处截断

    Args:
        seq_length (_type_): _description_
        eos_token_id (_type_): _description_
    """
    def sample_sequence(line):
        """
        sample出句子中的一小段。

        Args:
            line (dict): 单个句子 (batch_size=1), e.g. -> {
                'input_ids': tensor([37560, 24535, ...])
            }

        Returns:
            _type_: _description_
        """
        doc_length = line["input_ids"].shape[0]                     # len of current sentence, e.g. -> 378
        if doc_length <= seq_length:
            start = 0
        else:
            if random.random() < 1 / 4:                             # 25% 的概率选择最开始的一段
                start = 0
            else:
                start = random.randint(0, doc_length - seq_length)  # 75% 的概率在整个句子中随机 sample 一段
        input_ids = line["input_ids"][start : start + seq_length]
        
        if input_ids[-1] != eos_token_id:
            input_ids[-1] = eos_token_id
        
        return {"input_ids": input_ids}
    return sample_sequence


def split_sequence_gen(seq_length):
    """
    将长文章切成n段。
    * 优点: 保证了所有文章内容都被模型训练到
    * 缺点: 数据中可能会存在许多 padding token, 降低训练速度

    Args:
        seq_length (_type_): _description_
    """
    def split_sequence(batch):
        """
        单句切分。

        Args:
            batch (dict): 单个句子 (batch_size=1), e.g. -> {
                'input_ids': tensor([37560, 24535, ...])
            }

        Returns:
            _type_: _description_
        """
        input_ids = batch["input_ids"][0]
        if len(input_ids) < seq_length:
            out = input_ids
        else:
            out = []
            while len(input_ids) >= (1 + len(out)) * seq_length:
                out.append(input_ids[len(out) * seq_length : (1 + len(out)) * seq_length])
        return {"input_ids": out}
    return split_sequence


def check_eos_token_gen(
        eos_token_id,
        pad_token_id
    ):
    """
    用于检测句子末尾是否是以 eos_token_id 结尾。
    """
    def check_eos_token(batch):
        """
        check func.

        Args:
            line (dict): 单个句子 (batch_size=1), e.g. -> {
                'input_ids': tensor([37560, 24535, ...])
            }

        Returns:
            _type_: _description_
        """
        input_ids = batch["input_ids"]
        if input_ids[-1] != pad_token_id and input_ids[-1] != eos_token_id:
            input_ids[-1] = eos_token_id
        return {"input_ids": input_ids}
    return check_eos_token


def concat_multiple_sequence_gen(
        seq_length: int, 
        pad_token_id: int
    ):
    """
    先将多个不同的sequence都并成一整个序列，再均分成n条。

    Args:
        seq_length (int): 最大长度, e.g. -> 2048
        pad_token_id (int): _description_
    """
    def concat_multiple_sequence(batch):
        """
        将一个batch内的数据做拼接（长度平均化）。
        * 优点: 减少训练数据中的 padding token, 提升训练速度。
        * 缺点: 一个 sample 中可能混入来自多段不同文章的句子, 最终训练的模型也将续写出不同段落的结果, e.g. ->
                "白日依山尽," -> "黄河入海流。2012年中国本土文化得到了极大的提升, ..."

        Args:
            batch (batch_size, seq_len(每个句子长度不固定)): 按照指定 batch_size 生成的 batch 数据, 
                    e.g. -> {
                        'input_ids': [
                            tensor([37580, 35618, ...]),    // shape = (368,)
                            tensor([59644, 30346, ...]),    // shape = (289,)
                            ...
                        ]
                    }

        Returns:
            dict: 平均化后的 batch, e.g, -> {
                'input_ids': [
                    tensor([37580, 35618, ...]),    // shape = (2048,)
                    tensor([39482, 18346, ...]),    // shape = (2048,)
                ]
            }
        """
        concat_input_ids = torch.cat(batch["input_ids"], dim=0)                         # (sum_of_all_input_ids,) e.g. -> (8198,)
        length = concat_input_ids.shape[0]                                              # sum_of_all_input_ids, e.g. -> 8198
        chunks = math.ceil(length / seq_length)                                         # 最少用多少个chunks可以装完所有的句子(一个chunk长度为max_seq_len), e.g. -> ceil(8198 / 2048) = 5
        pad_length = chunks * seq_length - length                                       # 装完所有的chunk需要padding多少token, e.g. -> 5 * 2048 - 8198 = 2042
        pad = torch.ones(pad_length, dtype=concat_input_ids.dtype) * pad_token_id       # e.g. -> (2042,)
        concat_input_ids = torch.cat([concat_input_ids, pad], dim=0)                    # e.g. -> (10240,)
        input_ids = torch.chunk(concat_input_ids, chunks)                               # 均分为n个sample, e.g. -> [tensor(2048,), tensor(2048,), ...]
        return {"input_ids": input_ids}
    return concat_multiple_sequence


def get_labels_gen(
        pad_token_id,
        concat_multiple_sequence
    ):
    """
    添加label字段。

    Args:
        pad_token_id (_type_): _description_
        concat_multiple_sequence (_type_): _description_
    """
    def get_labels(line):
        input_ids = line["input_ids"]
        labels = input_ids.clone()
        labels[labels == pad_token_id] = -100

        if (
            not concat_multiple_sequence 
            and 
            "prompt_span_index_list" in line
        ):
            for prompt_span in line["prompt_span_index_list"]:
                labels[prompt_span[0]:prompt_span[1]] = -100

        return {"labels": labels}

    return get_labels


def sample_files_by_size(
        sampled_size: float, 
        source_files_dict: dict
    ):
    """
    根据每个 source file 的文件大小，以及目标采样大小，挑选训练数据集文件。
    """
    current_sampled_size = 0
    current_sampled_files = []
    all_files = list(source_files_dict.keys())

    while current_sampled_size < sampled_size:
        random_file = random.choice(all_files)
        current_sampled_files.append(random_file)
        current_sampled_size += source_files_dict[random_file]
    
    return current_sampled_size, current_sampled_files


def sample_dataset_by_source(
        dataset_config
    ):
    """
    按照不同比例在不同数据源之间 sample 数据集。
    由于每个文件的大小可能不一样（尽管文件中的文档数一样，但不同的文档长度可能差异很大），
    我们将使用「文件大小」作为相对公平的 sample 指标。
    """
    sample_policy_file = dataset_config.get('sample_policy_file', '')
    
    if sample_policy_file:
        sample_policy = json.load(open(sample_policy_file, 'r'))
        print(f'[Sample Policy] Used Sample Policy Config {sample_policy}.')
    else:
        sample_policy = {}

    all_data_files, total_file_size = {}, 0
    print('[Sample Policy] Calculating all dataset size...')
    for source, pattern in tqdm(dataset_config["data"].items()):
        data_files = glob(pattern)
        assert len(data_files) > 0, f'Source `{source}` has no files by pattern `{pattern}`.'
        
        all_data_files[source] = {}
        for data_file in data_files:
            fsize = os.path.getsize(data_file)
            fsize = fsize / float(1024 * 1024 * 1024)               # GB
            all_data_files[source][data_file] = fsize               # {'part-1': 0.142, 'part-2': 0.6723, ...}
            total_file_size += fsize
        
    print(f'[Sample Policy] Total file size (compressed): {round(total_file_size, 2)} GB.', )
    
    sampled_data_files = []
    if sample_policy:
        sampled_result, total_sampled_size = {}, 0
        for source, source_files_dict in all_data_files.items():
            assert source in sample_policy, f'source name: "{source}" is not in config "{sample_policy_file}".'
            
            sample_ratio = sample_policy[source]
            assert 0 < sample_ratio < 1, f'Each ratio should in (0, 1), while received {sample_ratio}.'
            
            sample_file_size = total_file_size * sample_ratio
            sampled_size, sampled_files = sample_files_by_size(
                sample_file_size, 
                source_files_dict
            )

            if not sampled_files:
                print(f'Source `{source}` has sampled NO files, please check sample ratio in config file.')
                print('source_files_dict: ', source_files_dict)
                print('sample_file_size: ', sample_file_size)
                exit()

            sampled_data_files.extend(sampled_files)

            sampled_size = round(sampled_size, 2)
            all_size = round(sum(source_files_dict.values()), 2)
            learned_times = round(sampled_size / all_size, 2)
            sampled_result[source] = f"{len(sampled_files)}/{len(source_files_dict)} - {sampled_size}GB/{all_size}GB ({learned_times} epochs)"
            
            total_sampled_size += sampled_size
        print(f'[Sample Policy] Sample results (compressed {round(total_sampled_size, 2)} GB): {sampled_result}.')
    else:
        for files in all_data_files.values():
            sampled_data_files.extend(files)

    random.shuffle(sampled_data_files)
    return sampled_data_files


def find_sub_list_index(
        main_list: list,
        sub_list: list
    ):
    """
    寻找一段 token span 在整个 input_ids 中的起、始位置。

    Args:
        main_list (list): inputs_ids, e.g. -> [1, 2, 3, 4, 5]
        sub_list (list): sub_span_ids, e.g. -> [3, 4, 5]
    
    Returns:
        (start, end), e.g. -> (2, 5)
    """
    start, end = -1, -1
    for idx in range(len(main_list) - len(sub_list) + 1):
        if main_list[idx: idx + len(sub_list)] == sub_list:
            start, end = idx, idx + len(sub_list)
            return start, end
    return start, end


def tokenize_inputs(
        tokenizer,
        pad_to_max: bool,
        added_eos_token: bool,
        seq_length,
        truncation
    ):
    """
    将文本转换为token。

    Args:
        pad_to_max (_type_): 句子是否要 padding。
        added_eos_token (_type_): 是否需要手动添加 eos token。
    """
    def tokenize(line):
        """
        将句子给 token 化。

        Args:
            line (dict): 单个样本, 如果是 SFT 则有 prompt_content 字段, 如果是 Pretrain 则没有, e.g. -> {
                'text': '将下面句子翻译为中文：Hello World!答：你好世界！',
                'prompt_content': '将下面句子翻译为中文：Hello World!答：'
            }

        Returns:
            _type_: _description_
        """
        if pad_to_max:
            if added_eos_token:
                input_ids = tokenizer(
                    line['text'],
                    return_tensors="pt",
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    padding="max_length",
                    max_length=seq_length,
                    truncation=truncation,
                )
            else:
                input_ids = tokenizer(
                    line['text'] + tokenizer.eos_token,
                    return_tensors="pt",
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    padding="max_length",
                    max_length=seq_length,
                    truncation=truncation,
                )
        else:
            if added_eos_token:
                input_ids = tokenizer(
                    line['text'],
                    return_tensors="pt",
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    truncation=truncation,
                )
            else:
                input_ids = tokenizer(
                    line['text'] + tokenizer.eos_token,
                    return_tensors="pt",
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    truncation=truncation,
                )
        
        prompt_span_index_list = []
        input_ids = input_ids['input_ids'][0]

        if "prompt_contents" in line:
            for prompt_content in line['prompt_contents']:
                if added_eos_token:
                    prompt_ids = tokenizer(prompt_content)['input_ids'][:-1]
                else:
                    prompt_ids = tokenizer(prompt_content)['input_ids']

                start, end = find_sub_list_index(
                    input_ids.tolist(), 
                    prompt_ids
                )

                if start != -1 and start < end:
                    prompt_span_index_list.append((start, end))

            return {
                'input_ids': input_ids,
                'prompt_span_index_list': prompt_span_index_list    # 如果是 SFT 需要 mask 掉 prompt 的 loss，
            }                                                       # 所以需要记录 prompt span 的起始位置
        
        return {"input_ids": input_ids}
    return tokenize


def construct_dataset(
        dataset_config, 
        tokenizer, 
        world_size=None
    ):
    all_data_files = sample_dataset_by_source(
        dataset_config
    )

    # 当shard可以被world_size整除时 split_dataset_by_node 会直接按shard进行划分，否则会读所有数据然后跳过一部分，可能会慢一点
    # https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.distributed.split_dataset_by_node
    if world_size is not None:
        num_shards = len(all_data_files)
        all_data_files = all_data_files[: num_shards // world_size * world_size]
    
    dataset = load_dataset(
        "json", 
        data_files=all_data_files, 
        split="train", 
        streaming=True
    )
    
    # shuffle
    dataset = dataset.shuffle(seed=42)
    
    # 文本预处理转换为统一格式
    if dataset_config["mode"] == "pretrain":
        dataset = dataset.map(
            pretrain_transform_gen(tokenizer), 
            batched=True, 
            batch_size=1
        )
        dataset = dataset.select_columns('text')
    elif dataset_config["mode"] == "sft":
        dataset = dataset.map(
            sft_transform_gen(tokenizer), 
            batched=True, 
            batch_size=1
        )
        dataset = dataset.select_columns(['text', 'prompt_contents'])
    else:
        raise Exception("Dataset mode: {} not found.".format(dataset_config["mode"]))

    seq_length = dataset_config["seq_length"]
    pad_to_max = dataset_config.get("pad_to_max", True)
    added_eos_token = dataset_config["added_eos_token"]
    sequence_sample_mode = dataset_config.get("sequence_sample_mode", "truncation")
    truncation = sequence_sample_mode == "truncation"
    concat_multiple_sequence = dataset_config.get("concat_multiple_sequence", False)
    
    # tokenize
    dataset = dataset.map(
        tokenize_inputs(tokenizer, pad_to_max, added_eos_token, seq_length, truncation)
    )

    # sequence_sample
    if sequence_sample_mode == "none":
        pass
    elif sequence_sample_mode == "truncation":
        dataset = dataset.map(
            check_eos_token_gen(tokenizer.eos_token_id, tokenizer.pad_token_id)
        )
    elif sequence_sample_mode == "sample":
        assert pad_to_max or concat_multiple_sequence
        dataset = dataset.map(
            sample_sequence_gen(seq_length, tokenizer.eos_token_id)
        )
    elif sequence_sample_mode == "split":
        assert not concat_multiple_sequence, \
            "You can't use both `concat_multiple_sequence` and `split`."
        dataset = dataset.map(
            split_sequence_gen(seq_length), 
            batched=True, 
            batch_size=1
        )
    else:
        raise Exception(
            "Unknown sequence_sample mode: {}.".format(sequence_sample_mode)
        )

    # filter unsed columns
    if dataset_config["mode"] == "pretrain":
        dataset = dataset.select_columns('input_ids')
    elif dataset_config["mode"] == "sft":
        dataset = dataset.select_columns(['input_ids', 'prompt_span_index_list'])

    # concat multiple sequence
    if concat_multiple_sequence:
        num_sequences = dataset_config["num_sequences"]
        dataset = dataset.map(
            concat_multiple_sequence_gen(seq_length, tokenizer.pad_token_id),
            batched=True,
            batch_size=num_sequences,
            drop_last_batch=True,
        )

    # add label
    dataset = dataset.map(
        get_labels_gen(tokenizer.pad_token_id, concat_multiple_sequence)
    )

    # filter unsed columns
    dataset = dataset.select_columns(["input_ids", "labels"])

    # shuffle
    dataset = dataset.shuffle(seed=42)
    return dataset


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from rich import print
    from transformers import AutoTokenizer, LlamaTokenizer

    data_config = {
        "mode": "sft",
        "data": {
            "sharegpt": "/mnt/bn/pankeyu/mlx/users/pankeyu/playground/LLMsTrainer/data/sft_pky_data/sharegpt/*.jsonl.zst"
        },
        "pad_to_max": True,
        "sequence_sample_mode": "truncation",
        "concat_multiple_sequence": False,
        "num_sequences": 10,
        "seq_length": 2048
    }

    # tokenizer_path = '/mnt/bn/pankeyu/mlx/users/pankeyu/playground/LLMsTrainer/configs/tokenizer_configs/openllama'
    tokenizer_path = '/mnt/bn/pankeyu/mlx/users/pankeyu/playground/LLMsTrainer/configs/tokenizer_configs/falcon_plus_v2'
    
    if 'llama' in tokenizer_path or 'Llama' in tokenizer_path:
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    test_tokens = tokenizer('this is a test sentence')['input_ids']                    # 测试tokenizer是否会自动添加eos token，用于后续判断是否自动为句子添加eos token
    data_config['added_eos_token'] = test_tokens[-1] == tokenizer.eos_token_id

    pretrain_dataset = construct_dataset(
        data_config, 
        tokenizer
    )
    
    print('n_shards: ', pretrain_dataset.n_shards)
    pretrain_loader = DataLoader(
        pretrain_dataset, 
        batch_size=1, 
        num_workers=1
    )
    
    sample_count, log_interval = 0, 10000
    for batch in pretrain_loader:
        sample_count += 1
        if not sample_count % log_interval:
            print(f'current_sample: {sample_count}...')
        for k, ids_list in batch.items():
            print(k, ids_list.shape)
            # print(k)
            print(ids_list)
            
            v_list = []
            for ids in ids_list:
                v_list.append([v for v in ids if v != -100])
            print([tokenizer.decode(v) for v in v_list])
        input('Next...')
    print(sample_count)
    print('Done.')