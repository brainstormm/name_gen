import pandas as pd
import matplotlib.pyplot as plt

# Load the loss history from CSV
loss_data = pd.read_csv("loss_history.csv")

# Plot the loss curves
plt.figure(figsize=(10, 6))
plt.plot(
    loss_data["epoch"],
    loss_data["train_loss"],
    marker="o",
    linestyle="-",
    color="blue",
    label="Training Loss",
)
plt.plot(
    loss_data["epoch"],
    loss_data["val_loss"],
    marker="o",
    linestyle="-",
    color="red",
    label="Validation Loss",
)
plt.title("Training and Validation Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.savefig("loss_curve.png")
plt.close()

# Print some statistics
print("\nLoss Statistics:")
print(f"Final Training Loss: {loss_data['train_loss'].iloc[-1]:.4f}")
print(f"Final Validation Loss: {loss_data['val_loss'].iloc[-1]:.4f}")
print(
    f"Best Validation Loss: {loss_data['val_loss'].min():.4f} (Epoch {loss_data['val_loss'].idxmin() + 1})"
)
print(
    f"Training Loss Range: {loss_data['train_loss'].min():.4f} - {loss_data['train_loss'].max():.4f}"
)
print(
    f"Validation Loss Range: {loss_data['val_loss'].min():.4f} - {loss_data['val_loss'].max():.4f}"
)
