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
    num_layers,
)
from model import RNN
import random
import logging
from datetime import datetime
import traceback
import sys
import csv
from model_analysis import inspect_gradients

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

    # Convert to tensors
    X = torch.stack([torch.tensor(x) for x in X])
    Y = torch.stack([torch.tensor(y) for y in Y])

    # Split data into training and validation sets (90% train, 10% validation)
    data_size = X.shape[0]
    train_size = int(0.9 * data_size)

    # Shuffle indices
    indices = list(range(data_size))
    random.shuffle(indices)

    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create training and validation sets
    X_train = X[train_indices]
    Y_train = Y[train_indices]
    X_val = X[val_indices]
    Y_val = Y[val_indices]

    # Reshape data
    X_train = X_train.reshape(-1, sequence_length, 1)
    X_val = X_val.reshape(-1, sequence_length, 1)

    # Normalize
    X_train = X_train / float(get_vocab_size())
    X_val = X_val / float(get_vocab_size())

    # Move to device
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_val = X_val.to(device)
    Y_val = Y_val.to(device)

    logging.info(f"Training data shapes - X: {X_train.shape}, Y: {Y_train.shape}")
    logging.info(f"Validation data shapes - X: {X_val.shape}, Y: {Y_val.shape}")

    model = RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=get_vocab_size(),
        num_layers=num_layers,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    logging.info(f"Starting training for {epochs} epochs with batch size {batch_size}")

    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 5  # Number of epochs to wait for improvement
    patience_counter = 0
    best_model_state = None

    # Lists to store losses
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        try:
            # Training phase
            model.train()
            train_loss = 0.0
            num_train_batches = 0

            # Shuffle training data
            train_idx = list(range(len(X_train)))
            random.shuffle(train_idx)

            for i in range(0, len(X_train), batch_size):
                batch_idx = train_idx[i : i + batch_size]
                X_batch = X_train[batch_idx]
                Y_batch = Y_train[batch_idx]
                Y_batch = Y_batch.reshape(-1)

                init_hidden = model.init_hidden(X_batch.shape[0], num_layers)
                output, hidden = model(X_batch, init_hidden)
                loss = criterion(output, Y_batch)

                optimizer.zero_grad()
                loss.backward()

                if (epoch == 5 or epoch == epochs - 1) and i == 0:
                    grad_norm = inspect_gradients(model)
                    logging.info(f"Gradient norm: {grad_norm}")

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                num_train_batches += 1

                if i % 10000 == 0:
                    logging.info(
                        f"Epoch {epoch + 1}, Batch {i // batch_size + 1}, Loss: {loss.item()}"
                    )

            avg_train_loss = train_loss / num_train_batches
            train_losses.append(avg_train_loss)

            # Validation phase
            model.eval()
            val_loss = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for i in range(0, len(X_val), batch_size):
                    X_batch = X_val[i : i + batch_size]
                    Y_batch = Y_val[i : i + batch_size]
                    Y_batch = Y_batch.reshape(-1)

                    init_hidden = model.init_hidden(X_batch.shape[0], num_layers)
                    output, hidden = model(X_batch, init_hidden)
                    loss = criterion(output, Y_batch)

                    val_loss += loss.item()
                    num_val_batches += 1

            avg_val_loss = val_loss / num_val_batches
            val_losses.append(avg_val_loss)

            logging.info(
                f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                logging.info(f"New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                logging.info(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    logging.info("Early stopping triggered!")
                    break

        except Exception as e:
            logging.error(f"Error during epoch {epoch + 1}:")
            logging.error(traceback.format_exc())
            raise

    logging.info("Training completed!")

    # Save the best model
    if best_model_state is not None:
        torch.save(best_model_state, "model.pth")
        logging.info("Saved best model based on validation loss")
    else:
        torch.save(model.state_dict(), "model.pth")
        logging.info("Saved final model")

    # Save losses to CSV
    with open("loss_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            writer.writerow([i + 1, train_loss, val_loss])

except Exception as e:
    logging.error("Fatal error occurred during training:")
    logging.error(traceback.format_exc())
    raise
