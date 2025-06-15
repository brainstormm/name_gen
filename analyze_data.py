import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data import get_data, get_vocab_size, decode
from collections import Counter
import torch

# Get the training data
X, Y = get_data()
X_train = torch.stack([torch.tensor(x) for x in X])
Y_train = torch.stack([torch.tensor(y) for y in Y])

# 1. Distribution of sequence lengths
sequence_lengths = [len(x) for x in X]
plt.figure(figsize=(10, 6))
plt.hist(sequence_lengths, bins=30, edgecolor='black')
plt.title('Distribution of Sequence Lengths')
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.savefig('sequence_length_distribution.png')
plt.close()

# 2. Character frequency distribution
char_freq = Counter()
for y in Y:
    char_freq[decode(y)] += 1

plt.figure(figsize=(12, 6))
chars = list(char_freq.keys())
freqs = list(char_freq.values())
plt.bar(chars, freqs)
plt.title('Character Frequency Distribution')
plt.xlabel('Characters')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('character_frequency.png')
plt.close()

# 3. Training data shape analysis
print(f"Training data shapes:")
print(f"X shape: {X_train.shape}")
print(f"Y shape: {Y_train.shape}")
print(f"Vocabulary size: {get_vocab_size()}")

# 4. Sequence pattern analysis
first_chars = [decode([x[0]]) for x in X]
last_chars = [decode([x[-1]]) for x in X]
next_chars = [decode([y[0]]) for y in Y]

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(first_chars, bins=len(set(first_chars)))
plt.title('First Character Distribution')
plt.xticks(rotation=45)

plt.subplot(1, 3, 2)
plt.hist(last_chars, bins=len(set(last_chars)))
plt.title('Last Character Distribution')
plt.xticks(rotation=45)

plt.subplot(1, 3, 3)
plt.hist(next_chars, bins=len(set(next_chars)))
plt.title('Next Character Distribution')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('sequence_patterns.png')
plt.close()

# 5. Save summary statistics to a file
with open('data_analysis_summary.txt', 'w') as f:
    f.write("Data Analysis Summary\n")
    f.write("===================\n\n")
    f.write(f"Total number of sequences: {len(X)}\n")
    f.write(f"Vocabulary size: {get_vocab_size()}\n")
    f.write(f"Sequence length: {X_train.shape[1]}\n")
    f.write(f"Most common characters: {char_freq.most_common(10)}\n")
    f.write(f"Least common characters: {char_freq.most_common()[:-11:-1]}\n") 