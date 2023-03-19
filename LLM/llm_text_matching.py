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

利用 LLM 进行文本匹配任务。

Author: pankeyu
Date: 2023/03/17
"""
from rich import print
from rich.console import Console
from transformers import AutoTokenizer, AutoModel


# 提供相似，不相似的语义匹配例子
examples = {
    '是': [
        ('如何找回账号', '账号丢了怎么办'),
    ],
    '不是': [
        ('如何找回账号', '附近最近的饭店'),
        ('李白技能讲解', '静夜思作者是李白吗')
    ]
}



def init_prompts():
    """
    初始化前置prompt，便于模型做 incontext learning。
    """
    pre_history = [
        (
            '现在你需要帮助我完成文本匹配任务，当我给你两个句子时，你需要回答我这两句话语义是否相似。只需要回答是否相似，不要做多余的回答。',
            '好的，我将只回答”是“或”不是“。'
        )
    ]

    for key, sentence_pairs in examples.items():
        for sentence_pair in sentence_pairs:
            sentence1, sentence2 = sentence_pair
            pre_history.append((
                f'句子一: {sentence1}\n句子二: {sentence2}\n上面两句话是相似的语义吗？',
                key
            ))

    return {'pre_history': pre_history}


def inference(
        sentence_pairs: list,
        custom_settings: dict
    ):
    """
    推理函数。

    Args:
        model (transformers.AutoModel): Language Model 模型。
        sentence_pairs (List[str]): 待推理的句子对。
        custom_settings (dict): 初始设定，包含人为给定的 few-shot example。
    """
    for sentence_pair in sentence_pairs:
        sentence1, sentence2 = sentence_pair
        sentence_with_prompt = f'句子一: {sentence1}\n句子二: {sentence2}\n上面两句话是相似的语义吗？'
        with console.status("[bold bright_green] Model Inference..."):
            response, history = model.chat(tokenizer, sentence_with_prompt, history=custom_settings['pre_history'])
        print(f'>>> [bold bright_red]sentence: {sentence_pair}')
        print(f'>>> [bold bright_green]inference answer: {response}')
        # print(history)


if __name__ == '__main__':
    console = Console()

    device = 'cuda:0'
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half()
    model.to(device)

    sentence_pairs = [
        ('如何修改头像', '可以通过上传图片修改头像吗'),
        ('王者荣耀司马懿连招', '王者荣耀司马懿有什么技巧'),
        ('王者荣耀司马懿连招', '历史上司马懿真的被诸葛亮空城计骗了吗'),
    ]    

    custom_settings = init_prompts()
    inference(
        sentence_pairs,
        custom_settings
    )