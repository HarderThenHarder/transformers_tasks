"""
模型参数冻结。

Code Source: https://www.zhihu.com/question/311095447/answer/589307812
"""
from collections.abc import Iterable


def set_freeze_by_names(model, layer_names, freeze=True):
    """
    core func.
    """
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        
        for param in child.parameters():
            param.requires_grad = not freeze
            

def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)


def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)


def set_freeze_by_idxs(model, idxs, freeze=True):
    if not isinstance(idxs, Iterable):
        idxs = [idxs]
    
    num_child = len(list(model.children()))
    idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
    
    for idx, child in enumerate(model.children()):
        if idx not in idxs:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze


def freeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, True)


def unfreeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, False)


def freeze_model_exclude_token_embeddings(
        model,
        token_embedding_layer_name
    ):
    """
    只训练模型的 token_embeddings, 将剩余参数全都 freeze。
    """
    found_word_embedding, freezed_modules = False, 0
    
    for name, param in model.named_parameters():
        if token_embedding_layer_name in name:
            param.requires_grad = True
            found_word_embedding = True
        else:
            param.requires_grad = False
            freezed_modules += 1
    
    assert found_word_embedding, f"Can not found `word embedding layer` witch named {token_embedding_layer_name}, check again."
    print(f'{freezed_modules} modules have been FREEZED.')


if __name__ == '__main__':
    from rich import print
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(
        '/mnt/bn/pankeyu/mlx/users/pankeyu/playground/backbones/falcon7b', 
        trust_remote_code=True
    )