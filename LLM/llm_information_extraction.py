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

利用 LLM 进行信息抽取任务，先对句子做分类，再进行信息提取。

Author: pankeyu
Date: 2023/03/19
"""
import re
import json

from rich import print
from rich.console import Console
from transformers import AutoTokenizer, AutoModel


# 分类 example
class_examples = {
        '人物': '岳云鹏，本名岳龙刚，1985年4月15日出生于河南省濮阳市南乐县，中国内地相声、影视男演员。2005年，首次登台演出。2012年，主演卢卫国执导的喜剧电影《就是闹着玩的》。2013年在北京举办相声专场。',
        '书籍': '《三体》是刘慈欣创作的长篇科幻小说系列，由《三体》《三体2：黑暗森林》《三体3：死神永生》组成，第一部于2006年5月起在《科幻世界》杂志上连载，第二部于2008年5月首次出版，第三部则于2010年11月出版。',
        '电视剧': '《狂飙》是由中央电视台、爱奇艺出品，留白影视、中国长安出版传媒联合出品，中央政法委宣传教育局、中央政法委政法综治信息中心指导拍摄，徐纪周执导，张译、张颂文、李一桐、张志坚、吴刚领衔主演，倪大红、韩童生、李建义、石兆琪特邀主演，李健、高叶、王骁等主演的反黑刑侦剧。',
    }
class_list = list(class_examples.keys())

CLS_PATTERN = f"“{{}}”是 {class_list} 里的什么类别？"


# 定义不同实体下的具备属性
schema = {
    '人物': ['姓名', '性别', '出生日期', '出生地点', '职业', '获得奖项'],
    '书籍': ['书名', '作者', '类型', '发行时间', '定价'],
    '电视剧': ['电视剧名称', '导演', '演员', '题材', '出品方']
}

IE_PATTERN = "{}\n\n提取上述句子中{}类型的实体，并按照JSON格式输出，上述句子中不存在的信息用['原文中未提及']来表示，多个值之间用','分隔。"


# 提供一些例子供模型参考
ie_examples = {
        '人物': [
                    {
                        'content': '岳云鹏，本名岳龙刚，1985年4月15日出生于河南省濮阳市南乐县，中国内地相声、影视男演员。',
                        'answers': {
                                        '姓名': ['岳云鹏'],
                                        '性别': ['男'],
                                        '出生日期': ['1985年4月15日'],
                                        '出生地点': ['河南省濮阳市南乐县'],
                                        '职业': ['相声演员', '影视演员'],
                                        '获得奖项': ['原文中未提及']
                            }
                    }
        ],
        '书籍': [
                    {
                        'content': '《三体》是刘慈欣创作的长篇科幻小说系列，由《三体》《三体2：黑暗森林》《三体3：死神永生》组成，第一部于2006年5月起在《科幻世界》杂志上连载，第二部于2008年5月首次出版，第三部则于2010年11月出版。',
                        'answers': {
                                        '书名': ['《三体》'],
                                        '作者': ['刘慈欣'],
                                        '类型': ['长篇科幻小说'],
                                        '发行时间': ['2006年5月', '2008年5月', '2010年11月'],
                                        '定价': ['原文中未提及']
                            }
                    }
        ]
}


def init_prompts():
    """
    初始化前置prompt，便于模型做 incontext learning。
    """
    class_list = list(class_examples.keys())
    cls_pre_history = [
        (
            f'现在你是一个文本分类器，你需要按照要求将我给你的句子分类到：{class_list}类别中。',
            f'好的。'
        )
    ]

    for _type, exmpale in class_examples.items():
        cls_pre_history.append((f'“{exmpale}”是 {class_list} 里的什么类别？', _type))

    ie_pre_history = [
        (
            "现在你需要帮助我完成信息抽取任务，当我给你一个句子时，你需要帮我抽取出句子中三元组，并按照JSON的格式输出，上述句子中没有的信息用['原文中未提及']来表示，多个值之间用','分隔。",
            '好的，请输入您的句子。'
        )
    ]

    for _type, example_list in ie_examples.items():
        for example in example_list:
            sentence = example['content']
            properties_str = ', '.join(schema[_type])
            schema_str_list = f'“{_type}”({properties_str})'
            sentence_with_prompt = IE_PATTERN.format(sentence, schema_str_list)
            ie_pre_history.append((
                f'{sentence_with_prompt}',
                f"{json.dumps(example['answers'], ensure_ascii=False)}"
            ))

    return {'ie_pre_history': ie_pre_history, 'cls_pre_history': cls_pre_history}


def clean_response(response: str):
    """
    后处理模型输出。

    Args:
        response (str): _description_
    """
    if '```json' in response:
        res = re.findall(r'```json(.*?)```', response)
        if len(res) and res[0]:
            response = res[0]
        response.replace('、', ',')
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
            sentence_with_cls_prompt = CLS_PATTERN.format(sentence)
            cls_res, _ = model.chat(tokenizer, sentence_with_cls_prompt, history=custom_settings['cls_pre_history'])

            if cls_res not in schema:
                print(f'The type model inferenced {cls_res} which is not in schema dict, exited.')
                exit()

            properties_str = ', '.join(schema[cls_res])
            schema_str_list = f'“{cls_res}”({properties_str})'
            sentence_with_ie_prompt = IE_PATTERN.format(sentence, schema_str_list)
            ie_res, _ = model.chat(tokenizer, sentence_with_ie_prompt, history=custom_settings['ie_pre_history'])
            ie_res = clean_response(ie_res)
        print(f'>>> [bold bright_red]sentence: {sentence}')
        print(f'>>> [bold bright_green]inference answer: ')
        print(ie_res)


if __name__ == '__main__':
    console = Console()

    device = 'cuda:0'
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half()
    model.to(device)

    sentences = [
        '张译（原名张毅），1978年2月17日出生于黑龙江省哈尔滨市，中国内地男演员。1997年至2006年服役于北京军区政治部战友话剧团。2006年，主演军事励志题材电视剧《士兵突击》。',
        '《琅琊榜》是由山东影视传媒集团、山东影视制作有限公司、北京儒意欣欣影业投资有限公司、北京和颂天地影视文化有限公司、北京圣基影业有限公司、东阳正午阳光影视有限公司联合出品，由孔笙、李雪执导，胡歌、刘涛、王凯、黄维德、陈龙、吴磊、高鑫等主演的古装剧。',
    ]

    custom_settings = init_prompts()
    inference(
        sentences,
        custom_settings
    )