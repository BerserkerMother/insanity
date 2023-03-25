from collections import defaultdict

import transformers


def main():
    # test corpus
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]

    word_frequency = defaultdict(int)

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

    for sentence in corpus:
        pre_tokenized = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sentence)
        sen_words = [word for word, offset in pre_tokenized]
        for word in sen_words:
            word_frequency[word] += 1

    print(word_frequency)

    alphabet = []

    for word in word_frequency.keys():
        for letter in word:
            if letter not in alphabet:
                alphabet.append(letter)

    print(alphabet)

    # complete vocab
    vocab = ["<|endoftext|>"] + alphabet

    # convert words to their corresponding chars
    splits = {word: [c for c in word] for word in word_frequency.keys()}

    merges = {}

    while len(vocab) < 50:
        pair_freq = build_and_compute_pair_frequency(word_frequency, splits)
        # most frequent pair
        pair, freq = None, 0
        for p, f in pair_freq.items():
            if f > freq:
                pair = p
                freq = f
        # merges
        merges[pair] = pair[0] + pair[1]
        vocab.append(merges[pair])

        splits = merge_pair(pair[0], pair[1], splits, word_frequency)

    res = tokenize("This is not a token", merges, tokenizer)
    print(res)


def build_and_compute_pair_frequency(word_freq, splits):
    # pair frequency
    pair_frequency = defaultdict(int)
    for word, freq in word_freq.items():
        word = splits[word]
        if len(word) == 1:
            continue
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_frequency[pair] += freq
    return pair_frequency


def merge_pair(a, b, splits, word_freq):
    # pair frequency
    for word in word_freq:
        split = splits[word]
        if len(word) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2:]
            else:
                i += 1
        splits[word] = split
    return splits


def tokenize(sen, merge_rules, tokenizer):
    pre_token = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sen)
    pre_token = [word for word, offset in pre_token]
    splits = [[c for c in word] for word in pre_token]

    for pair, merge in merge_rules.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if pair[0] == split[i] and pair[1] == split[i + 1]:
                    split = split[:i] + [merge] + split[i + 2:]
                else:
                    i += 1
            splits[idx] = split
    return sum(splits, [])


if __name__ == '__main__':
    main()
