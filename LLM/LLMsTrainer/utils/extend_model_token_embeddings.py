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
扩充模型中的 token_embedding 和 lm_head。

Author: pankeyu
Date: 2023/07/24
"""
import torch
from rich import print


def extend_token_embeddings(
    embed_tokens,
    lm_head,
    origin_tokenizer,
    extend_tokenizer
):
    """
    为了尽可能多复用原始 BPE 词表中的中文子 token 信息，
    使用中文的所有 sub token embedding 的平均值作为初始化 embedding。

    Args:
        embed_tokens: 原始模型中的 token embedding layer
        lm_head: 原始模型中的 lm_head layer
        origin_tokenizer: 原始tokenizer
        extend_tokenizer: 扩词表后的tokenizer（当前只支持新扩展tokens拼接在原始tokens之后的tokenizer）。
    """    
    embed_dim = embed_tokens.embedding_dim
    
    print('Token Embedding Hidden Dim: ', embed_dim)
    print('Language Model Head: ', lm_head)

    origin_tokenizer_tokens = list(origin_tokenizer.get_vocab().keys())
    extend_tokenizer_tokens = list(extend_tokenizer.get_vocab().keys())
    assert len(extend_tokenizer_tokens) > len(origin_tokenizer), \
        'Extend vocab size should larger than Origin vocab.'

    new_embeddings = torch.nn.Embedding(len(extend_tokenizer_tokens), embed_dim)              # allocate new matrix, shape: torch.Size([40419, 4096])
    new_embeddings.to(embed_tokens.weight.device)

    new_embeddings.weight.data.normal_(mean=0.0, std=1.0)                                     # initialize weights
    if new_embeddings.padding_idx is not None:
        new_embeddings.weight.data[new_embeddings.padding_idx].zero_()

    num_tokens_to_copy = min(
        len(origin_tokenizer_tokens), 
        len(extend_tokenizer_tokens)
    )

    new_embeddings.weight.data[:num_tokens_to_copy, :] = embed_tokens.weight.data[:num_tokens_to_copy, :]

    extend_token_start_index = len(origin_tokenizer_tokens)                                   # get the first extend token index
    print('Extend token start index: ', extend_token_start_index)

    for i in range(extend_token_start_index, len(extend_tokenizer_tokens)):
        ext_token = extend_tokenizer_tokens[i]
        
        # print('Before: ', new_embeddings.weight.data[i, :])
        if '<' in ext_token and '>' in ext_token:                                             # don't initial special token like: <pad>, <title>, ...
            pass
        else:
            sub_tokens = origin_tokenizer.encode(ext_token)                                   # e.g. [1, 259, 13]
            drop_tokens = [origin_tokenizer.bos_token_id, origin_tokenizer.eos_token_id]      # drop bos/eos token 
            sub_tokens = [
                sub_token for sub_token in sub_tokens if sub_token not in drop_tokens         # e.g. [259, 13]
            ]
            sub_tokens_embeddings = embed_tokens(torch.LongTensor(sub_tokens))                # all sub-token embedding in origin tokenizer, shape: (2, 4096)
            avg_sub_token_embedding = torch.mean(sub_tokens_embeddings, dim=0)                # average sub-token embedding as new extend token embedding, shpae: (4096,)
            new_embeddings.weight.data[i, :] = avg_sub_token_embedding                        # update average embedding in new embedding matrix
        # print('Before: ', new_embeddings.weight.data[i, :])

    print('New token embedding size: ', new_embeddings.weight.size())                         # torch.Size([40419, 4096])

    old_lm_head_output_features, old_lm_head_input_features = lm_head.weight.size()
    new_lm_head = torch.nn.Linear(
        old_lm_head_input_features, 
        len(extend_tokenizer_tokens),
        bias=False
    ).to(lm_head.weight.device)

    new_lm_head.weight.data[:num_tokens_to_copy, :] = lm_head.weight.data[:num_tokens_to_copy, :]

    for i in range(extend_token_start_index, len(extend_tokenizer_tokens)):
        ext_token = extend_tokenizer_tokens[i]

        # print('Before: ', new_lm_head.weight.data[i, :])
        if '<' in ext_token and '>' in ext_token:                                             # don't initial special token like: <pad>, <title>, ...
            pass
        else:
            sub_tokens = origin_tokenizer.encode(ext_token)                                   # e.g. [1, 259, 13]
            drop_tokens = [origin_tokenizer.bos_token_id, origin_tokenizer.eos_token_id]      # drop bos/eos token 
            sub_tokens = [
                sub_token for sub_token in sub_tokens if sub_token not in drop_tokens         # e.g. [259, 13]
            ]
            sub_lm_embeddings = lm_head.weight.data[sub_tokens, :]                            # all sub-token embedding in origin tokenizer, shape: (2, 4096)
            avg_sub_lm_embedding = torch.mean(sub_lm_embeddings, dim=0)                       # averaage sub-toke lm embedding as new extend token embedding, shpae: (4096,)
            new_lm_head.weight.data[i, :] = avg_sub_lm_embedding                              # update average embedding in new embedding matrix
        # print('After: ', new_lm_head.weight.data[i, :])

    print('New lm_head size: ', new_lm_head.weight.size())                                    # torch.Size([40419, 4096])

    return new_embeddings, new_lm_head


if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, LlamaTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        'openlm-research/open_llama_7b_v2',
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map='auto'
    )

    origin_tokenizer = LlamaTokenizer.from_pretrained(
        'openlm-research/open_llama_7b_v2',
        trust_remote_code=True,
    )

    extend_tokenizer = LlamaTokenizer.from_pretrained(
        'test_tokenizer',
        trust_remote_code=True,
    )

    new_embeddings, new_lm_head = extend_token_embeddings(
        model.model.embed_tokens,
        model.lm_head,
        origin_tokenizer,
        extend_tokenizer
    )

    model.model.embed_tokens = new_embeddings
    model.lm_head = new_lm_head
    model.config.vocab_size = len(extend_tokenizer)

    model.save_pretrained(
        'open-llama7b-v2-avg-plus'
    )