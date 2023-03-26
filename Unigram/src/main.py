import math
import copy
from collections import defaultdict

import transformers

FINAL_VOCAB_SIZE = 70


def main():
    words = ["hug", "pug", "pun", "bun", "hugs"]
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]

    tokenizer = transformers.AutoTokenizer.from_pretrained("xlnet-base-cased")
    pre_tokenizer = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str

    word_freq = defaultdict(int)
    for sentence in corpus:
        sentence = pre_tokenizer(sentence)
        for word, offset in sentence:
            word_freq[word] += 1
    print(word_freq)
    print("_" * 100)

    # char and sub word freq
    sub_word_freq, char_freq = get_char_sub_word_freq(word_freq)
    sorted_sub_words = sorted(sub_word_freq.items(), key=lambda x: x[1], reverse=True)
    print(sorted_sub_words[:10])
    print("_" * 100)

    vocab_size = 300  # init vocab size
    token_freq = list(char_freq.items()) + sorted_sub_words[: vocab_size - len(char_freq)]
    token_freq = {key: value for key, value in token_freq}
    print(token_freq)
    print("_" * 100)

    # create model
    total_sum = sum([freq for token, freq in token_freq.items()])
    model = {token: -math.log(freq / total_sum) for token, freq in token_freq.items()}
    print(encode_word("Hopefully", model))
    print(encode_word("This", model))
    print("_" * 100)

    # main loop
    percent_to_remove = 0.1
    while len(model) > 100:
        scores = compute_scores(model, word_freq)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        # Remove percent_to_remove tokens with the lowest scores.
        for i in range(int(len(model) * percent_to_remove)):
            _ = token_freq.pop(sorted_scores[i][0])

        total_sum = sum([freq for token, freq in token_freq.items()])
        model = {token: -math.log(freq / total_sum) for token, freq in token_freq.items()}

    print(tokenize("This is the Hugging Face course.", model, pre_tokenizer))


def tokenize(text, model, pre_tokenizer):
    words_with_offsets = pre_tokenizer(text)
    pre_tokenized_text = [word for word, offset in words_with_offsets]
    encoded_words = [encode_word(word, model)[0] for word in pre_tokenized_text]
    return sum(encoded_words, [])


def compute_scores(model, word_freq):
    scores = {}
    model_loss = compute_loss(model, word_freq)
    for token, score in model.items():
        # We always keep tokens of length 1
        if len(token) == 1:
            continue
        model_without_token = copy.deepcopy(model)
        _ = model_without_token.pop(token)
        scores[token] = compute_loss(model_without_token, word_freq) - model_loss
    return scores


def compute_loss(model, word_freqs):
    loss = 0
    for word, freq in word_freqs.items():
        _, word_loss = encode_word(word, model)
        loss += freq * word_loss
    return loss


def encode_word(word, model):
    best_segmentations = [{"start": 0, "score": 1}] + [
        {"start": None, "score": None} for _ in range(len(word))
    ]
    for start_idx in range(len(word)):
        # This should be properly filled by the previous steps of the loop
        best_score_at_start = best_segmentations[start_idx]["score"]
        for end_idx in range(start_idx + 1, len(word) + 1):
            token = word[start_idx:end_idx]
            if token in model and best_score_at_start is not None:
                score = model[token] + best_score_at_start
                # If we have found a better segmentation ending at end_idx, we update
                if (
                        best_segmentations[end_idx]["score"] is None
                        or best_segmentations[end_idx]["score"] > score
                ):
                    best_segmentations[end_idx] = {"start": start_idx, "score": score}
    segmentation = best_segmentations[-1]
    if segmentation["score"] is None:
        # We did not find a tokenization of the word -> unknown
        return ["<unk>"], None

    score = segmentation["score"]
    start = segmentation["start"]
    end = len(word)
    tokens = []
    while start != 0:
        tokens.insert(0, word[start:end])
        next_start = best_segmentations[start]["start"]
        end = start
        start = next_start
    tokens.insert(0, word[start:end])
    return tokens, score


def get_char_sub_word_freq(word_freq):
    char_freq = defaultdict(int)
    sub_word_freq = defaultdict(int)
    for word, freq in word_freq.items():
        for i in range(len(word)):
            char_freq[word[i]] += freq
            for j in range(i + 2, len(word) + 1):
                sub_word_freq[word[i:j]] += freq
    return sub_word_freq, char_freq


if __name__ == '__main__':
    main()
