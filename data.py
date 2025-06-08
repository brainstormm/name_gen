import math
from config import sequence_length

with open("input.txt", "r") as f:
    text = f.read()
    words = text.split()

vocab = sorted(list(set(text)))

min_length = math.inf
max_length = 0

for word in words:
    min_length = min(min_length, len(word))
    max_length = max(max_length, len(word))

stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}
stoi['<'] = len(vocab)
stoi['>'] = len(vocab) + 1
itos[len(vocab)] = '<'
itos[len(vocab) + 1] = '>'


X = []
Y = []

for w in words:
    w = "<" + w + ">"
    for i in range(len(w) - sequence_length + 1):
        X.append(w[i : i + sequence_length])
        Y.append(w[i + 1 : i + sequence_length + 1])

print(text)
print(len(text))
print(len(words))
print(vocab)
print(min_length, max_length)
print(stoi)
print(itos)
print(X)
print(Y)