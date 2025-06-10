import torch
from data import get_data, get_vocab_size
from config import sequence_length, epochs, learning_rate, batch_size, input_size, hidden_size
from model import RNN
import random
import logging
from datetime import datetime

# Set up logging
log_filename = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

X, Y = get_data()

X_train = torch.stack([torch.tensor(x) for x in X])
Y_train = torch.stack([torch.tensor(y) for y in Y])

logging.info(f"Training data shapes - X: {X_train.shape}, Y: {Y_train.shape}")

model = RNN(
    input_size=input_size, hidden_size=hidden_size, output_size=get_vocab_size()
)

data_size = X_train.shape[0]
logging.info(f"Starting training for {epochs} epochs with batch size {batch_size}")

for epoch in range(epochs):
    idx = list(range(data_size))
    random.shuffle(idx)
    init_hidden = model.init_hidden(batch_size)
    for i in range(0, data_size, batch_size):
        batch_idx = idx[i : i + batch_size]
        X_batch = X_train[batch_idx]
        Y_batch = Y_train[batch_idx]
        output, hidden = model(X_batch, init_hidden)

logging.info("Training completed!")
