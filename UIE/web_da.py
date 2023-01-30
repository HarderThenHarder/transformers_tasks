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

自动数据增强 web server。

Author: pankeyu
Date: 2023/01/19
"""
import os
import json
import time
from io import StringIO
from typing import List

import torch
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, T5ForConditionalGeneration

from Augmenter import Augmenter
from inference import inference


st.set_page_config(
    page_title="Auto Data Augment For UIE",
    page_icon="🧸"
)

device = 'cpu'                                                 # 指定设备
generated_dataset_height = 800                                 # 生成样本展示高度                
max_show_num = 500                                             # 生成样本最大保存行数
max_seq_len = 128                                              # 数据集单句最大长度
batch_size = 128                                               # 负例生成时的batch_size

filling_model_path = 'transformers_tasks/data_augment/mask_then_fill/checkpoints/t5/model_best'


def get_prompt_samples(name: str) -> List[str]:
    """
    解析服务器/用户上传的数据集。
    """
    st.markdown('选择 **:green[.txt]** 数据集用于数据增强，支持接收多个文件。')
    st.code('⚠️ 注意：若传入多个文件，多个文件的数据将被融合。')
    st.markdown('* 数据集示例：')
    st.code("""[
        {
            "content": "一杠三星什么军衔？上尉", 
            "prompt": "一杠三星的军衔", 
            "result_list": [{"text": "上尉", "start": 9, "end": 11}]
        },
        ...
    ]
    """, language='json')
    
    with st.expander('', expanded=True):
        all_samples, files_count = [], 0
        upload_file = st.checkbox(f'上传本地文件用于：{name}', help='上传本地的数据集文件。')
        if upload_file:
            uploaded_files = st.file_uploader(f"选择文件", accept_multiple_files=True)
            for uploaded_file in uploaded_files:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                string_data = stringio.read()
                cur_data = string_data.split('\n')
                all_samples.extend([json.loads(line) for line in cur_data])
                files_count += 1
                df_dict = {
                    'text': cur_data[:5] + ['...']
                }
                df = pd.DataFrame.from_dict(df_dict)
                st.dataframe(df)
        else:
            dataset_path_str = st.text_input(f'「训练数据集」的服务器路径（多个文件用英文逗号分隔）用于：{name}', 
                help='支持访问 /mnt/nas-search-nlp/XXX 路径下的数据集，如: /mnt/nas-search-nlp/test/train.txt。'
            )
            if dataset_path_str:
                dataset_path_list = dataset_path_str.split(',')
                for dataset_path in dataset_path_list:
                    if not dataset_path:
                        continue
                    if not os.path.exists(dataset_path):
                        st.error(f'未找到文件 "{dataset_path}"，请确认文件路径。')
                        st.stop()
                    else:
                        lines = [json.loads(line.strip()) for line in open(dataset_path, 'r', encoding='utf8').readlines()]
                        df_dict = {
                            'text': lines[:5] + ['...']
                        }
                        df = pd.DataFrame.from_dict(df_dict)
                        st.dataframe(df)
                        all_samples.extend(lines)
                        files_count += 1
        st.markdown(f'总文件数：**:blue[{files_count}]**')
        st.markdown(f'总样本数：**:blue[{len(all_samples)}]**')
    return all_samples


def get_doccano_samples():
    """
    解析doccano导出的数据集，并数据增强。
    """
    st.markdown('选择 **:green[.jsonl]** 数据集（从doccano导出）用于正负例生成，支持接收多个文件。')
    st.code('⚠️ 注意：若传入多个文件，多个文件的数据将被融合。')
    st.markdown('* 数据集示例：')
    st.code(
        """
        [
            {
                "text": "《股份制企业资产评估》是1999年中国人民大学出版社出版的图书，作者是刘角膜真菌病玉平", 
                "entities": [
                        {"id": 53636, "label": "书籍", "start_offset": 1, "end_offset": 10, "text": "股份制企业资产评估"}, 
                        {"id": 53637, "label": "出版社", "start_offset": 17, "end_offset": 26, "text": "中国人民大学出版社"}
                    ], 
                "relations": [
                        {"id": 0, "from_id": 53636, "to_id": 53637, "type": "出版社"}
                    ]
            },
            ...
        ]
        """, language='json'
    )
    
    with st.expander('', expanded=True):
        all_samples, files_count = [], 0
        upload_file = st.checkbox(f'上传本地.jsonl文件', help='上传本地的数据集文件。')
        if upload_file:
            uploaded_files = st.file_uploader("选择文件", accept_multiple_files=True)
            for uploaded_file in uploaded_files:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                string_data = stringio.read()
                cur_data = string_data.split('\n')
                all_samples.extend([json.loads(line) for line in cur_data])
                files_count += 1
                df_dict = {
                    'text': cur_data[:5] + ['...']
                }
                df = pd.DataFrame.from_dict(df_dict)
                st.dataframe(df)
        else:
            dataset_path_str = st.text_input(f'「训练数据集」的服务器路径，jsonl格式（多个文件用英文逗号分隔）', 
                help='支持访问 /mnt/nas-search-nlp/XXX 路径下的数据集，如: /mnt/nas-search-nlp/test/doccano.jsonl。'
            )
            if dataset_path_str:
                dataset_path_list = dataset_path_str.split(',')
                for dataset_path in dataset_path_list:
                    if not dataset_path:
                        continue
                    if not os.path.exists(dataset_path):
                        st.error(f'未找到文件 "{dataset_path}"，请确认文件路径。')
                        st.stop()
                    else:
                        lines = [json.loads(line.strip()) for line in open(dataset_path, 'r', encoding='utf8').readlines()]
                        df_dict = {
                            'text': lines[:5] + ['...']
                        }
                        df = pd.DataFrame.from_dict(df_dict)
                        st.dataframe(df)
                        all_samples.extend(lines)
                        files_count += 1
        st.markdown(f'总文件数：**:blue[{files_count}]**')
        st.markdown(f'总样本数：**:blue[{len(all_samples)}]**')
    return all_samples


def get_model():
    """
    加载模型/tokenizer。
    """
    with st.expander('', expanded=True):
        upload_model = st.checkbox('是否上传本地模型', help='上传本地模型文件，模型结构需和[这里](https://github.com/HarderThenHarder/transformers_tasks/blob/main/UIE/model.py)保持一致。')
        model, tokenizer = None, None
        if upload_model:
            uploaded_model = st.file_uploader("选择文件", disabled=True)
            st.warning('暂不支持上传本地文件。')
            st.stop()
        else:
            model_path = st.text_input('「训练模型」的服务器路径', 
                help='支持访问 /mnt/nas-search-nlp/XXX 路径下的模型文件，如: /mnt/nas-search-nlp/model_best/model.pt。'
            )
            tokenizer_path = st.text_input('「tokenizer」的服务器路径', 
                help='支持访问 /mnt/nas-search-nlp/XXX 路径下的目录，如: /mnt/nas-search-nlp/model_best/'
            )

            if model_path and tokenizer_path:
                if not os.path.exists(model_path):
                    st.error(f'未找到文件 "{model_path}"，请确认文件路径。')
                    st.stop()
                elif not os.path.exists(tokenizer_path):
                    st.error(f'未找到目录 "{tokenizer_path}"，请确认文件路径。')
                    st.stop()
                else:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                    with st.spinner('模型加载中，请稍等...'):
                        if device == 'cpu':
                            model = torch.load(model_path, map_location=torch.device('cpu'))
                        else:
                            model = torch.load(model_path).to(device)
                    model.eval()
                    st.success('🎉 模型加载成功~')
        return model, tokenizer


def main():
    main_external_css = """
        <style>
            h1, h2, h3, h4, h5, h6 {color: #b59a6d !important}
        </style>
    """
    st.markdown(main_external_css, unsafe_allow_html=True)
    st.markdown("[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&duration=600&pause=500&color=38F77FFF&vCenter=true&multiline=true&width=606&height=130&lines=import+torch;from+model+import+Model;Model.fit()_;Epoch+1+[=======================================];Get+Off+Work%EF%BC%81)](https://github.com/HarderThenHarder/transformers_tasks)")

    da_page, model_analysis_page = st.tabs(['UIE-数据增强', 'UIE-自分析负例生成'])
    with da_page:
        da_mode = st.radio('数据增强模式', ['基于规则（Swap SPO）', '基于模型（Mask Then Fill）'])
        if da_mode == '基于规则（Swap SPO）':
            with st.expander('Swap SPO 策略介绍', expanded=False):
                st.markdown("""
                    Swap SPO 是一种基于规则的简单数据增强策略。<br>
                    
                    将同一数据集中 相同P的句子分成一组，并随机交换这些句子中的S和O。<br>
                    
                    ---

                    * 输入：<br>
                    
                    1. :blue[《夜曲》]是:green[周杰伦]:violet[作曲]的一首歌。<br>

                    2. :blue[《那些你很冒险的梦》]是当下非常火热的一首歌，:violet[作曲]为:green[林俊杰]。<br>
                    
                    ---

                    * 输出：<br>

                    :blue[《夜曲》]是当下非常火热的一首歌，:violet[作曲]为:green[周杰伦]。
                """, unsafe_allow_html=True)
            all_samples = get_prompt_samples(name='正例增强')
        elif da_mode == '基于模型（Mask Then Fill）':
            with st.expander('Mask Then Fill 策略介绍', expanded=False):
                st.markdown("""
                    [Mask Then Fill](https://arxiv.org/pdf/2301.02427.pdf) 是一种基于生成模型的信息抽取数据增强策略。
                    
                    对于一段文本，我们其分为「关键信息段」和「非关键信息段」，包含关键词片段称为「关键信息段」。

                    > `Note:` 例子中带颜色的代表关键信息段，其余的为非关键信息段。
                    
                    [e.g.] :blue[大年三十]我从:violet[北京]的大兴机场:orange[飞回]了:green[成都]。

                    ---

                    1. 我们随机 [MASK] 住一部分「非关键片段」，使其变为：

                    [e.g.] :blue[大年三十]我从:violet[北京]**[MASK]**:orange[飞回]了:green[成都]。

                    2. 随后喂给 filling 模型（T5-Fine Tuned）还原句子，得到新生成的句子：

                    [e.g.] :blue[大年三十]我从:violet[北京]首都机场作为起点，:orange[飞回]了:green[成都]。

                """, unsafe_allow_html=True)
            all_samples = get_doccano_samples()
            aug_num = st.number_input('数据增强倍数', min_value=1, max_value=4, value=1)
        finish_button = st.button('开始生成正例')
        if finish_button:
            start = time.time()
            with st.spinner('正例生成中，请稍等...'):
                if da_mode == '基于规则（Swap SPO）':
                    positive_samples, predicates_sentence_dict = Augmenter.auto_add_uie_relation_positive_samples(
                        all_samples
                    )
                    st.session_state['positive_samples'] = positive_samples
                    st.session_state['predicates_sentence_dict'] = predicates_sentence_dict
                elif da_mode == '基于模型（Mask Then Fill）':
                    if 'filling_model' not in st.session_state:
                        st.session_state['filling_model'] = T5ForConditionalGeneration.from_pretrained(filling_model_path).to(device)
                        st.session_state['filling_tokenizer'] = AutoTokenizer.from_pretrained(filling_model_path)
                    positive_samples = Augmenter.auto_add_uie_relation_positive_samples(
                        all_samples,
                        mode='mask_then_fill',
                        filling_model=st.session_state['filling_model'],
                        filling_tokenizer=st.session_state['filling_tokenizer'],
                        batch_size=64,
                        max_seq_len=128,
                        device=device,
                        aug_num=aug_num
                    )
                    st.session_state['positive_samples'] = positive_samples
            time_used = time.time() - start
            st.balloons()
            st.markdown('耗时：:green[{:.2f}s]'.format(time_used))
        
        if 'positive_samples' in st.session_state:
            if da_mode == '基于规则（Swap SPO）':
                df_dict = {
                    'content': [],
                    'prompt': [],
                    'result_list': []
                }
            elif da_mode == '基于模型（Mask Then Fill）':
                df_dict = {
                    'text': [],
                    'entities': [],
                    'relations': []
                }
            for sample in st.session_state['positive_samples']:
                sample = json.loads(sample)
                for k, v in sample.items():
                    df_dict[k].append(str(v))

            st.markdown('---')
            st.markdown(f'生成正例数：:green[{len(st.session_state["positive_samples"])}]')
            
            if da_mode == '基于规则（Swap SPO）':
                st.markdown(f'共包含 predicate 个数：:green[{len(st.session_state["predicates_sentence_dict"])}]')
                with st.expander('同 Predicate 样本详情（若数据集较大建议先「下载生成结果」再展开，以防卡死）', expanded=False):
                    st.write('数据集下包含的所有Predicate(s)：')
                    st.code(f'{list(st.session_state["predicates_sentence_dict"].keys())}')
                    search_p = st.selectbox('查看数据集中指定的同P样本：', list(st.session_state["predicates_sentence_dict"].keys()))
                    st.write(st.session_state['predicates_sentence_dict'][search_p])
            
            df = pd.DataFrame.from_dict(df_dict)
            st.dataframe(df, height=generated_dataset_height)
            text_file = '\n'.join(st.session_state['positive_samples'])
            st.download_button('下载正例数据集', text_file)
    
    with model_analysis_page:
        all_samples = get_prompt_samples(name='自分析负例生成')
        model, tokenizer = get_model()
        finish_button = st.button('开始生成负例')
        if finish_button:
            start = time.time()
            with st.spinner('负例生成中，请稍等...'):
                predicate_error_dict, summary_dict, negative_samples = Augmenter.auto_add_uie_relation_negative_samples(
                    model,
                    tokenizer,
                    all_samples,
                    inference,
                    device=device,
                    max_seq_len=max_seq_len,
                    batch_size=batch_size
                )
                st.session_state['predicate_error_dict'] = predicate_error_dict
                st.session_state['summary_dict'] = summary_dict
                st.session_state['negative_samples'] = negative_samples
            time_used = time.time() - start
            st.balloons()
            st.markdown('耗时：:green[{:.2f}s]'.format(time_used))
        
        if 'negative_samples' in st.session_state:
            df_dict = {
                'content': [],
                'prompt': [],
                'result_list': []
            }
            for sample in st.session_state['negative_samples']:
                if len(df_dict['content']) > max_show_num:
                    break
                sample = json.loads(sample)
                for k, v in sample.items():
                    df_dict[k].append(str(v))

            st.markdown('---')
            st.markdown(f'生成负例数：:green[{len(st.session_state["negative_samples"])}]')
            with st.expander('易混淆Predicate详情', expanded=True):
                p_df_dict = {
                    '原始P': [],
                    '易混淆P': []
                }
                for k, v in st.session_state['summary_dict'].items():
                    p_df_dict['原始P'].append(k)
                    p_df_dict['易混淆P'].append(','.join(v))
                pdf = pd.DataFrame.from_dict(p_df_dict)
                st.table(pdf)
                search_p = st.selectbox('查看指定P的混淆详情', p_df_dict['原始P'])
                if search_p:
                    st.write(st.session_state['predicate_error_dict'][search_p])
            df = pd.DataFrame.from_dict(df_dict)
            st.dataframe(df, height=generated_dataset_height)
            text_file = '\n'.join(st.session_state["negative_samples"])
            st.download_button('下载负例数据集', text_file)


if __name__ == '__main__':
    main()