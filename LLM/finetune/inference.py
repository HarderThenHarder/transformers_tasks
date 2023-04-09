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
import time
import torch

from transformers import AutoTokenizer, AutoModel
torch.set_default_tensor_type(torch.cuda.HalfTensor)


def inference(
        model,
        tokenizer,
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
            max_new_tokens=max_new_tokens,
            temperature=0
        )
        out_text = tokenizer.decode(out[0])
        answer = out_text.split('Answer: ')[-1]
        return answer


if __name__ == '__main__':
    from rich import print

    device = 'cuda:0'
    max_new_tokens = 300
    model_path = "checkpoints/model_1000"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True
    ).half().to(device)

    samples = [
        {
            'instruction': "给你一个句子，同时给你一个句子主语和关系列表，你需要找出句子中主语包含的所有关系以及关系值，并输出为SPO列表。",
            "input": "给定主语：绵阳电影院\n给定句子：影院简介绵阳电影院位于绵阳市区最繁华的黄金地段，紧邻好又多，东面梅西百货，南接大观园，西面绵阳文化广场，北攘肯德基麦当劳，交通便利，是绵阳商业娱乐文化中心。绵阳电影院占地1000多平米，建筑面积3000多平米。影院采用了国际标准设计，即高空间、大坡度、超亮屏幕，视觉无遮挡。配备有SR.D、DTS顶级数码立体声还音系统，音响效果极佳。再配以高级航空座椅，日本三菱空调，以及优质的人文服务，堪称川西北一流影院。绵阳电影院拥有数字大厅一个，豪华数字厅两个（2、4厅），纯数字电影两个（3、5厅），影厅内安装有世界顶级的英国杜比CP650(EX)数字处理器、美国JBL音响、德国ISCO一体化镜头、美国QSC数字功放（DCA）、5.1声道杜比数码立体声系统！精彩电影、适中价位、舒适享受，尽在绵阳电影院。同时兼营小卖，水吧，服装等，年收入500多万。其中票房收入位于四川省单厅（大厅）之首。绵阳电影院还被四川省人民政府授予“文明电影院”称号。绵阳电影院隶属于绵阳市川涪影业股份有限责任公司，是四川省太平洋电影院线旗下的旗舰影院，绵阳电影院拥有中影进口大片首轮放映权。绵阳电影院成立50多年来，凭借优质服务、一流的硬件设施、优秀的地理位置赢得了广大绵阳人民的厚爱。\n给定关系列表：['所属国家', '建造者', '面积', '设计者', '高度']",
        },
        {
            'instruction': "给你一个句子，同时给你一个句子主语和关系列表，你需要找出句子中主语包含的所有关系以及关系值，并输出为SPO列表。",
            "input": "句子主语：我是一片云\n输入句子:《我是一片云》是根据琼瑶同名小说改编，由辜朗辉、赵俊宏执导，张晓敏、王伟平、孙启新、秦怡、严丽秋等主演的电视剧。该剧共5集，于1985年播出。\n输入关系:['主演', '演员', '作品类型', '首发时间', '导演', '播出平台', '首播平台', '集数']",
        }
    ]

    start = time.time()
    for i, sample in enumerate(samples):
        res = inference(
            model,
            tokenizer,
            sample['instruction'],
            sample['input']
        )
        print(f'res {i}: ')
        print(res)
    print(f'Used {round(time.time() - start, 2)}s.')