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

可视化Sampler策略，用于初期对多个source进行数据sample比例的制定。

Author: pankeyu
Date: 2023/06/01
"""
import copy

import pandas as pd
import streamlit as st
import plotly.express as px


st.set_page_config(
    page_title="Sampler Viewer",
    layout='wide'
)

if 'current_data_dis' not in st.session_state:
    st.session_state['current_data_dis_dict'] = {}
    st.session_state['current_data_dis_df'] = pd.DataFrame.from_dict({})
    st.session_state['current_data_percent_dict'] = {}
    st.session_state['modified_data_dis_dict'] = {}
    st.session_state['modified_data_dis_df'] = {}
    st.session_state['modified_data_percent_dict'] = {}
    st.session_state['modified_ratio_dict'] = {}


def main():
    """
    主函数流程。
    """
    st.sidebar.markdown(
        "<h1>📊 Sampler Viewer</h1>", 
        unsafe_allow_html=True
    )

    with st.expander('数据分布信息', expanded=True):
        data_distribution = st.text_area(
            '输入数据源名 & 数据量级（csv or json）',
            placeholder="""中文维基,英文维基\n1.62,14.37\n\nor\n\n{"中文维基": 1.62, "英文维基": 14.37}\n\n...""",
            height=200
        )
    
    if not data_distribution:
        st.stop()

    try:
        data_dis = eval(data_distribution)
    except Exception as e:
        if ',' in data_distribution:
            sources, nums = data_distribution.split('\n')
            sources = sources.split(',')
            nums = nums.split(',')
            data_dis = {}
            for s, n in zip(sources, nums):
                data_dis[s] = float(n)
        else:
            raise ValueError('Not invalid input.')
    
    st.session_state['current_data_dis_dict'] = copy.deepcopy(data_dis)
    st.session_state['modified_data_dis_dict'] = copy.deepcopy(data_dis)
    total = sum(data_dis.values())
    
    st.session_state['current_data_percent_dict'] = dict(
        [(k, v / total * 100) for k, v in data_dis.items()]
    )
    st.session_state['modified_data_percent_dict'] = copy.deepcopy(st.session_state['current_data_percent_dict'])

    current_dis = {
        'sources': list(data_dis.keys()),
        'nums': list(data_dis.values())
    }
    st.session_state['current_data_dis_df'] = pd.DataFrame.from_dict(current_dis)
    st.session_state['modified_data_dis_df'] = copy.deepcopy(st.session_state['current_data_dis_df'])
    
    if not st.session_state['current_data_dis_df'].empty:                                           # 原始分布绘制
        fig = px.pie(
            st.session_state['current_data_dis_df'], 
            values='nums', 
            names='sources', 
            hole=0.5,
            title='原始数据分布')
        st.sidebar.plotly_chart(fig, use_container_width=True)

        st.sidebar.markdown('---')

    with st.expander('采样概览', expanded=True):
        sample_result_area = st.empty()

    total_sample = 0
    for key, value in st.session_state['modified_data_percent_dict'].items():
        c1, c2, c3, c4 = st.columns([6, 6, 6, 6])
        with c1:
            modify_percent = st.number_input(
                f'「{key}」采样比例（%）',
                min_value=0.,
                max_value=100.,
                value=value,
                step=0.01
            )
        with c2:
            origin_size = st.session_state['current_data_dis_dict'][key]
            st.text_input(f'「{key}」原始大小：', value=f'{origin_size}GB', disabled=True)
        with c3:
            sample_size = (
                modify_percent / st.session_state['current_data_percent_dict'][key] 
                * st.session_state['current_data_dis_dict'][key]
            )
            st.text_input(f'「{key}」采样大小：', value=f'{sample_size}GB', disabled=True)
            st.session_state['modified_data_dis_dict'][key] = sample_size
            total_sample += sample_size
        with c4:
            ratio = round(modify_percent / st.session_state['current_data_percent_dict'][key], 2)
            st.text_input(f'「{key}」学习倍率：', value=f'{ratio} epoch(s)', disabled=True)
            st.session_state['modified_data_percent_dict'][key] = modify_percent

        modified_dis = {
            'sources': list(st.session_state['modified_data_dis_dict'].keys()),
            'nums': list(st.session_state['modified_data_dis_dict'].values())
        }
        st.session_state['modified_data_dis_df'] = pd.DataFrame.from_dict(modified_dis)

    percentage_left = 100 - sum(st.session_state['modified_data_percent_dict'].values())
    percentage_left = 0. if abs(percentage_left) < 1e-8 else percentage_left
    sample_result_area.markdown(f'当前剩余待分配的比例： :red[{round(percentage_left, 2)}] %. <br> 原始数据集大小： :green[{total}] GB. <br> 采样后数据集大小： :green[{round(total_sample, 2)}] GB.', unsafe_allow_html=True)

    if not st.session_state['modified_data_dis_df'].empty:                                          # 采样分布布绘制
        fig = px.pie(
            st.session_state['modified_data_dis_df'], 
            values='nums', 
            names='sources', 
            hole=0.5,
            title='采样数据分布')
        st.sidebar.plotly_chart(fig, use_container_width=True)


    c1, c2 = st.columns([9, 9])
    with c1:
        origin_percentage_to_float = dict(
            [(k, round(v / 100, 4)) for k, v in st.session_state['current_data_percent_dict'].items()]
        )
        st.write('原始数据分布：')
        st.write(origin_percentage_to_float)
        st.code(origin_percentage_to_float)

    with c2:
        modified_percentage_to_float = dict(
            [(k, round(v / 100, 4)) for k, v in st.session_state['modified_data_percent_dict'].items()]
        )
        st.write('采样数据分布：')
        st.write(modified_percentage_to_float)
        st.code(modified_percentage_to_float)


if __name__ == '__main__':
    main()