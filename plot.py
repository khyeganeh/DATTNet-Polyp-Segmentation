import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Create results folder
# -----------------------
os.makedirs("results", exist_ok=True)

# -----------------------
# Read history
# -----------------------
history = pd.read_csv("history.csv")

epochs = history["Epoch"]
train_loss = history["Train Loss"]
val_loss = history["Validation Loss"]

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(8,5))

plt.plot(
    epochs,
    train_loss,
    marker="o",
    linewidth=2,
    label="Train Loss"
)

plt.plot(
    epochs,
    val_loss,
    marker="s",
    linewidth=2,
    label="Validation Loss"
)

plt.title("Training History")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/loss_curve.png",
    dpi=300,
)

plt.show()

print("Loss curve saved in results/loss_curve.png")