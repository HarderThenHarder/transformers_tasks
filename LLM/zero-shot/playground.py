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

web端测试LLM效果。

Author: pankeyu
Date: 2023/03/17
"""
import streamlit as st
from transformers import AutoTokenizer, AutoModel

st.set_page_config(
    page_title="LLM Playground",
)

device = 'cpu'
if 'model' not in st.session_state:
    with st.spinner('Loading Model...'):
        tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
        if device == 'cpu':
            model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).float()
        else:
            model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
            model.to(device)
        st.session_state['model'] = model
        st.session_state['tokenizer'] = tokenizer


def main():
    """
    主函数流程。
    """
    logo = '[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&duration=500&pause=500&color=00E455&center=true&vCenter=true&multiline=true&repeat=false&width=700&height=80&lines=LLM%EF%BC%88Large+Language+Model%EF%BC%89Playground+-+Enjoy++(%EF%BC%BE%EF%BC%B5%EF%BC%BE)%E3%83%8E)](https://github.com/HarderThenHarder/transformers_tasks)'
    st.markdown(logo)
    c1, c2 = st.columns([5, 5])
    with c1:
        with st.expander('历史对话设定（In-Context Learning）', expanded=True):
            pre_history_count = st.number_input('历史对话轮数', 0, 10, 2)
            pre_history = []
            for i in range(pre_history_count):
                user_input = st.text_area(
                    f'轮数 {i + 1} - User'
                )
                bot_resp = st.text_area(
                    f'轮数 {i + 1} - Bot'
                )
                pre_history.append((user_input, bot_resp))
    
    with c2:
        with st.expander('对话输入框', expanded=True):
            current_input = st.text_area(
                '当前用户输入',
                value='Say Something...'
            )
            bt = st.button('Generate')
        if bt:
            with st.spinner('生成中...'):
                response, _ = st.session_state['model'].chat(st.session_state['tokenizer'], current_input, history=pre_history)
            with st.expander('当前模型输出', expanded=True):
                st.markdown(f':green[{response}]')


if __name__ == '__main__':
    main()