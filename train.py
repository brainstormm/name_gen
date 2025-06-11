import torch, torch.nn as nn
from data import get_data, get_vocab_size
from config import (
    sequence_length,
    epochs,
    learning_rate,
    batch_size,
    input_size,
    hidden_size,
    learning_rate,
    device,
)
from model import RNN
import random
import logging
from datetime import datetime
import traceback
import sys

# Set up logging
log_filename = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Add a stream handler to also show logs in console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logging.getLogger().addHandler(console_handler)

try:
    X, Y = get_data()

    X_train = torch.stack([torch.tensor(x) for x in X])
    Y_train = torch.stack([torch.tensor(y) for y in Y])
    data_size = X_train.shape[0]
    X_train = X_train.reshape(data_size, sequence_length, 1)
    X_train = X_train / float(get_vocab_size())
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)

    logging.info(f"Training data shapes - X: {X_train.shape}, Y: {Y_train.shape}")

    model = RNN(
        input_size=input_size, hidden_size=hidden_size, output_size=get_vocab_size()
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    logging.info(f"Starting training for {epochs} epochs with batch size {batch_size}")

    for epoch in range(epochs):
        try:
            idx = list(range(data_size))
            random.shuffle(idx)

            for i in range(0, data_size, batch_size):
                batch_idx = idx[i : i + batch_size]
                X_batch = X_train[batch_idx]
                Y_batch = Y_train[batch_idx]
                Y_batch = Y_batch.reshape(-1)
                init_hidden = model.init_hidden(X_batch.shape[0])
                output, hidden = model(X_batch, init_hidden)
                loss = criterion(output, Y_batch)
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if i % 10000 == 0:
                    logging.info(
                        f"Epoch {epoch + 1}, Batch {i // batch_size + 1}, Loss: {loss.item()}"
                    )

        except Exception as e:
            logging.error(f"Error during epoch {epoch + 1}:")
            logging.error(traceback.format_exc())
            raise

    logging.info("Training completed!")

except Exception as e:
    logging.error("Fatal error occurred during training:")
    logging.error(traceback.format_exc())
    raise

torch.save(model.state_dict(), "model.pth")
