# import math
# from config import sequence_length

# with open("input.txt", "r") as f:
#     text = f.read()
#     words = text.split()

# vocab = sorted(list(set(text)))

# min_length = math.inf
# max_length = 0

# for word in words:
#     min_length = min(min_length, len(word))
#     max_length = max(max_length, len(word))

# stoi = {ch: i for i, ch in enumerate(vocab)}
# itos = {i: ch for i, ch in enumerate(vocab)}

# stoi["<"] = len(vocab)
# stoi[">"] = len(vocab) + 1
# itos[len(vocab)] = "<"
# itos[len(vocab) + 1] = ">"


# def encode(s):
#     return [stoi[c] for c in s]


# def decode(l):
#     return "".join(itos[i] for i in l)


# def get_data():
#     X, Y = [], []
#     for w in words:
#         w = "<" + w + ">"
#         for i in range(len(w) - sequence_length):
#             X.append(encode(w[i : i + sequence_length]))
#             Y.append(encode(w[i + sequence_length]))
#     return X, Y


# def get_vocab_size():
#     return len(vocab) + 2


import math
from config import sequence_length

with open("input1.txt", "r") as f:
    text = f.read()

vocab = sorted(list(set(text)))

stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}


def encode(s):
    return [stoi[c] for c in s]


def decode(l):
    return "".join(itos[i] for i in l)


def get_data():
    X, Y = [], []
    for i in range(len(text) - sequence_length):
        X.append(encode(text[i : i + sequence_length]))
        Y.append(encode(text[i + sequence_length]))
    return X, Y

def get_vocab_size():
    return len(vocab)
