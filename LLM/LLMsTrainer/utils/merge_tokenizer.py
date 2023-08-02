from tqdm import tqdm
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as model
from transformers import AutoTokenizer


def merge_llama():
    raw_model = model.ModelProto()
    raw_model.ParseFromString(
        open("llama_tokenizer.model", "rb").read()         # 需要自己下载：https://huggingface.co/openlm-research/open_llama_7b_v2/tree/main
    )

    exist_pieces = set([p.piece for p in raw_model.pieces])
    cn_model = model.ModelProto()
    cn_model.ParseFromString(
        open("test_tokenizer.model", "rb").read()          # 训练好的模型
    )

    for p in tqdm(cn_model.pieces, total=len(cn_model.pieces)):
        if p.piece not in exist_pieces:
            raw_model.pieces.append(p)

    with open("llama_tokenizer_extended.model", "wb") as f:
        f.write(raw_model.SerializeToString())

    sp_model = spm.SentencePieceProcessor(
        model_file="llama_tokenizer_extended.model"
    )

    print("merged vocab size: {}".format(sp_model.vocab_size()))


def merge_auto_tokenizer(
        origin_hf_tokenizer_filepath, 
        added_tokenizer_filepath, 
        output_tokenizer_filepath
    ):
    """
    将训练出来的词表扩充到一个现有的tokenizer上。
    """
    tokenizer = AutoTokenizer.from_pretrained(
        origin_hf_tokenizer_filepath
    )

    cn_model = model.ModelProto()
    cn_model.ParseFromString(open(added_tokenizer_filepath, "rb").read())

    added_pieces = []
    for piece in cn_model.pieces:
        word = piece.piece.strip('▁')
        added_pieces.append(word)
    
    added_pieces.extend(
        ['，', '？', '！', '。', '；', '“', '”', '《', '》', '、', '：', '（', '）', '——', '……']
    )

    new_tokens = set(added_pieces) - set(tokenizer.vocab.keys())
    print('new tokens num: ', len(new_tokens))
    tokenizer.add_tokens(list(new_tokens))
    tokenizer.save_pretrained(output_tokenizer_filepath)


if __name__ == '__main__':
    merge_auto_tokenizer(
        'openlm-research/open_llama_7b_v2',
        'test_tokenizer.model',
        'test_tokenizer'
    )

