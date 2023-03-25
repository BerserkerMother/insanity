import transformers
from collections import defaultdict

FINAL_VOCAB_SIZE = 70


def main():
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]

    base_tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    pre_tokenizer = base_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str
    print(pre_tokenizer(corpus[3]))
    print("_" * 100)

    word_freq = defaultdict(int)  # word frequency
    # tokenize sentence and extrac words frequency
    for sentence in corpus:
        sentence = pre_tokenizer(sentence)
        for word, _ in sentence:
            word_freq[word] += 1
    print(word_freq)
    print("_" * 100)

    # word splits and alphabet
    splits, alphabet = get_splits_and_alphabet(word_freq)
    # vocab
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()
    print(vocab)
    print("_" * 100)
    print(splits)
    print("_" * 100)

    while len(vocab) <= FINAL_VOCAB_SIZE:
        pair_freq, single_freq = get_pair_single_freq(word_freq, splits)
        if len(pair_freq.keys()) == 0:
            break  # we are done, there are nothing more to add!
        best_pair = get_best_pair(pair_freq, single_freq)
        # add to vocab freq
        new_token = best_pair[0] + best_pair[1][2:]
        vocab.append(new_token)

        update_splits(splits, best_pair)
    print(vocab)
    print("_" * 100)

    print(tokenize("This is the Hugging Face course!", vocab, pre_tokenizer))


def update_splits(splits, best_pair):
    a, b = best_pair
    # update splits and find new pair freq
    for word, split in splits.items():
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b[2:]] + split[i + 2:]
            else:
                i += 1
        splits[word] = split
    return splits


def get_best_pair(pair_freq, vocab_freq):
    # find the highest score freq
    best_score = 0
    best_pair = None
    for pair, freq in pair_freq.items():
        score = freq / (vocab_freq[pair[0]] * vocab_freq[pair[1]])
        if best_score < score:
            best_pair = pair
            best_score = score
    return best_pair


def get_pair_single_freq(word_freq, splits):
    pair_freq = defaultdict(int)
    single_freq = defaultdict(int)
    for word, freq in word_freq.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freq[pair] += freq
            single_freq[split[i]] += freq
        single_freq[split[-1]] += freq
    return pair_freq, single_freq


def get_splits_and_alphabet(word_freq):
    alphabet = []
    splits = {}
    for word, freq in word_freq.items():
        split = []
        for idx, char in enumerate(word):
            char = "##" + char if idx != 0 else char
            if char not in alphabet:
                alphabet.append(char)
            split.append(char)
        splits[word] = split
    alphabet.sort()
    return splits, alphabet


def encode_word(word, vocab):
    tokens = []
    while len(word) > 0:
        i = len(word)
        while i > 0 and word[:i] not in vocab:
            i -= 1
        if i == 0:
            return ["[UNK]"]
        tokens.append(word[:i])
        word = word[i:]
        if len(word) > 0:
            word = f"##{word}"
    return tokens


def tokenize(text, vocab, pre_tokenizer):
    pre_tokenize_result = pre_tokenizer(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    encoded_words = [encode_word(word, vocab) for word in pre_tokenized_text]
    return sum(encoded_words, [])


if __name__ == '__main__':
    main()
