import torch


class EarlyStopping:

    def __init__(
        self,
        patience=5,
        delta=0.0,
        path="checkpoints/best_model.pth",
    ):

        self.patience = patience
        self.delta = delta
        self.path = path

        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss, model):

        if val_loss < self.best_loss - self.delta:

            self.best_loss = val_loss
            self.counter = 0

            torch.save(
                model.state_dict(),
                self.path,
            )

            print("✅ Validation improved. Model saved.")

        else:

            self.counter += 1

            print(
                f"⚠ No improvement ({self.counter}/{self.patience})"
            )

            if self.counter >= self.patience:

                self.early_stop = True

                print("\n🛑 Early Stopping Triggered!")