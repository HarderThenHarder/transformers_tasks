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

模板定义类，按照人为定义的模板进行encoding。

Author: pankeyu
Date: 2022/11/28
"""
import numpy as np


class HardTemplate(object):
    """
    硬模板，人工定义句子和[MASK]之间的位置关系。
    """

    def __init__(
        self,
        prompt: str
        ):
        """
        init func.

        Args:
            prompt (str): prompt格式定义字符串, e.g. -> "这是一条{MASK}评论：{textA}。"
        """
        self.prompt = prompt
        self.inputs_list = []                       # 根据文字prompt拆分为各part的列表
        self.custom_tokens = set(['MASK'])          # 从prompt中解析出的自定义token集合
        self.parse_prompt()                         # 解析prompt模板

    def parse_prompt(self):
        """
        将prompt文字模板拆解为可映射的数据结构。

        Examples:
            prompt -> "这是一条{MASK}评论：{textA}。"
            inputs_list -> ['这', '是', '一', '条', 'MASK', '评', '论', '：', 'textA', '。']
            custom_tokens -> {'textA', 'MASK'}
        """
        idx = 0
        while idx < len(self.prompt):
            part = ''
            if self.prompt[idx] not in ['{', '}']:
                self.inputs_list.append(self.prompt[idx])
            if self.prompt[idx] == '{':                  # 进入自定义字段
                idx += 1
                while self.prompt[idx] != '}':
                    part += self.prompt[idx]             # 拼接该自定义字段的值
                    idx += 1
            elif self.prompt[idx] == '}':
                raise ValueError("Unmatched bracket '}', check your prompt.")
            if part:
                self.inputs_list.append(part)
                self.custom_tokens.add(part)             # 将所有自定义字段存储，后续会检测输入信息是否完整
            idx += 1

    def __call__(
        self,
        inputs_dict: dict,
        tokenizer,
        mask_length,
        max_seq_len=512,
        ) -> dict:
        """
        输入一个样本，转换为符合模板的格式。

        Args:
            inputs_dict (dict): prompt中的参数字典, e.g. -> {
                                                            "textA": "这个手机也太卡了", 
                                                            "MASK": "[MASK]"
                                                        }
            tokenizer: 用于encoding文本
            mask_length (int): MASK token 的长度

        Returns:
            dict -> {
                'text': '[CLS]这是一条[MASK]评论：这个手机也太卡了。[SEP]',
                'input_ids': [1, 47, 10, 7, 304, 3, 480, 279, 74, 47, 27, 247, 98, 105, 512, 777, 15, 12043, 2],
                'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                'mask_position': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        """
        assert self.custom_tokens == set(inputs_dict), \
            f"@params inputs_dict doesn't match @param prompt, @prompt needs: {self.custom_tokens}, while @inputs_dict keys are: {set(inputs_dict)}."
        
        outputs = {
            'text': '', 
            'input_ids': [],
            'token_type_ids': [],
            'attention_mask': [],
            'mask_position': []
        }

        formated_str = ''
        for ele in self.inputs_list:
            if ele in self.custom_tokens:
                if ele == 'MASK':
                    formated_str += inputs_dict[ele] * mask_length
                else:
                    formated_str += inputs_dict[ele]
            else:
                formated_str += ele

        encoded = tokenizer(
            text=formated_str,
            truncation=True,
            max_length=max_seq_len,
            padding='max_length')
        outputs['input_ids'] = encoded['input_ids']
        outputs['token_type_ids'] = encoded['token_type_ids']
        outputs['attention_mask'] = encoded['attention_mask']
        outputs['text'] = ''.join(tokenizer.convert_ids_to_tokens(encoded['input_ids']))
        mask_token_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        mask_position = np.where(np.array(outputs['input_ids']) == mask_token_id)[0].tolist()
        outputs['mask_position'] = mask_position
        return outputs


if __name__ == '__main__':
    from rich import print
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    template = HardTemplate(
        prompt='这是一条{MASK}评论：{textA}'
    )
    tep = template(
                inputs_dict={'textA': '包装不错，苹果挺甜的，个头也大。', 'MASK': '[MASK]'},
                tokenizer=tokenizer,
                max_seq_len=30,
                mask_length=2
            )
    print(tep)

    # print(tokenizer.convert_ids_to_tokens([3819, 3352, 3819, 3352]))
    # print(tokenizer.convert_tokens_to_ids(['水', '果']))