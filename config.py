import torch

torch.manual_seed(1337)

sequence_length = 3
batch_size = 4096
epochs = 20
hidden_size = 128
learning_rate = 0.001
input_size = 1  # The number of dimensions of the input vector. This is 1 because we are just encoding the characters as integers.

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
