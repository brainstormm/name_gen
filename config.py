import torch

torch.manual_seed(1337)

sequence_length = 8
batch_size = 128
epochs = 1000
hidden_size = 32
num_layers = 2
learning_rate = 0.001

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
