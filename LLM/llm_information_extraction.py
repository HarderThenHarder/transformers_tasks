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

利用 LLM 进行信息抽取任务。

Author: pankeyu
Date: 2023/03/19
"""
import re
import json

from rich import print
from rich.console import Console
from transformers import AutoTokenizer, AutoModel


# 定义不同实体下的具备属性
schema = {
    '人物': ['姓名', '性别', '出生日期', '出生地点', '职业', '获得奖项', '实体类型'],
    '书籍': ['书名', '作者', '类型', '发行时间', '定价', '实体类型'],
    '电视剧': ['电视剧名称', '导演', '演员', '题材', '出品方', '实体类型']
}


# prompt 模板
schema_str_list = []
for _type, properties in schema.items():
    properties_str = ', '.join(properties)
    schema_str_list.append(f'“{_type}”({properties_str})')
schema_str = '，'.join(schema_str_list)

IE_PATTERN = f'{{}}\n\n提取上述句子中{schema_str}类型的实体，并按照JSON格式输出，不允许出现上述句子中没有的信息。'


# 提供一些例子供模型参考
examples = [
    {
        'content': '岳云鹏，本名岳龙刚，1985年4月15日出生于河南省濮阳市南乐县，中国内地相声、影视男演员。',
        'answers': {
                        '姓名': '岳云鹏',
                        '性别': '男',
                        '出生日期': '1985年4月15日',
                        '出生地点': '河南省濮阳市南乐县',
                        '职业': '相声演员',
                        '获得奖项': '原文中未提及',
                        '实体类型': '人物'
            }
    },
    {
        'content': '《三体》是刘慈欣创作的长篇科幻小说系列，由《三体》《三体2：黑暗森林》《三体3：死神永生》组成，第一部于2006年5月起在《科幻世界》杂志上连载。',
        'answers': {
                        '书名': '《三体》',
                        '作者': '刘慈欣',
                        '类型': '长篇科幻小说',
                        '发行时间': '2006年5月',
                        '定价': '原文中未提及',
                        '实体类型': '书籍'
            }
    }
]


def init_prompts():
    """
    初始化前置prompt，便于模型做 incontext learning。
    """
    pre_history = [
        (
            '现在你需要帮助我完成信息抽取任务，当我给你一个句子时，你需要帮我抽取出句子中三元组，并按照JSON的格式输出，上述句子中没有的信息用“原文中未提及”来表示。',
            '好的，我将帮您完成信息抽取任务，并按照JSON的形式输出，原文中未出现的内容我会用“原文中未提及”来表示。'
        )
    ]

    for example in examples:
        sentence = example['content']
        sentence_with_prompt = IE_PATTERN.format(sentence)
        pre_history.append((
            f'{sentence_with_prompt}',
            json.dumps(example['answers'], ensure_ascii=False)
        ))

    return {'pre_history': pre_history}


def clean_response(response: str):
    """
    后处理模型输出。

    Args:
        response (str): _description_
    """
    if '```json' in response:
        res = re.findall(r'```json(.*?)```', response)
        if len(res):
            response = res[0]
    try:
        return json.loads(response)
    except:
        return response


def inference(
        sentences: list,
        custom_settings: dict
    ):
    """
    推理函数。

    Args:
        sentences (List[str]): 待抽取的句子。
        custom_settings (dict): 初始设定，包含人为给定的 few-shot example。
    """
    for sentence in sentences:
        with console.status("[bold bright_green] Model Inference..."):
            sentence_with_prompt = IE_PATTERN.format(sentence)
            response, history = model.chat(tokenizer, sentence_with_prompt, history=custom_settings['pre_history'])
            response = clean_response(response)
        print(f'>>> [bold bright_red]sentence: {sentence}')
        print(f'>>> [bold bright_green]inference answer: ')
        print(response)
        # print(history)


if __name__ == '__main__':
    console = Console()

    device = 'cuda:0'
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half()
    model.to(device)

    sentences = [
        '小惠玲，1968年2月21日出生于江苏省盐城市大丰区，中国内地影视女演员。',
        '《十五度的地下世界》是北京作家沈大星创作的科幻小说，该书于2004年在全国发售。',
    ]

    custom_settings = init_prompts()
    inference(
        sentences,
        custom_settings
    )