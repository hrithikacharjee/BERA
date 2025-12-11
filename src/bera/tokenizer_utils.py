import sentencepiece as spm


def train_sentencepiece(input_file, model_prefix, vocab_size=32000):
    spm.SentencePieceTrainer.train(
        f"--input={input_file} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--model_type=bpe "
        f"--character_coverage=1.0"
    )


def load_sentencepiece(model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp
