import datasets
import tokenizers
from tokenizers import models, normalizers, pre_tokenizers, trainers, processors


def main():
    dataset = datasets.load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")
    print(dataset)

    tokenizer = tokenizers.Tokenizer(
        models.WordPiece(unk_token="[UNK]")
    )
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    print(tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))
    print("_" * 100)

    pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Punctuation()
    ])
    print(pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer."))
    print("_" * 100)

    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
    tokenizer.train_from_iterator(get_training_corpus(dataset), trainer=trainer)

    encoding = tokenizer.encode("Let's test this tokenizer.")
    print(encoding.tokens)
    print("_" * 100)

    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
    )

    encoding = tokenizer.encode("Let's test this tokenizer.")
    print(encoding.tokens)
    print("_" * 100)


def get_training_corpus(dataset):
    for i in range(0, len(dataset), 1000):
        yield dataset[i: i + 1000]["text"]


if __name__ == '__main__':
    main()
