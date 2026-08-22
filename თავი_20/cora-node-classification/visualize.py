import json
from pathlib import Path

import matplotlib.pyplot as plt


history_path = Path("results/history.json")

if not history_path.exists():
    raise FileNotFoundError(
        "results/history.json ვერ მოიძებნა. "
        "ჯერ გაუშვი train.py."
    )

with history_path.open("r", encoding="utf-8") as file:
    history = json.load(file)


epochs = range(1, len(history["train_loss"]) + 1)

figure, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12, 5)
)


# Loss-ის გრაფიკი
axes[0].plot(
    epochs,
    history["train_loss"],
    label="Train loss"
)

axes[0].plot(
    epochs,
    history["validation_loss"],
    label="Validation loss"
)

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss-ის ცვლილება")
axes[0].legend()
axes[0].grid(alpha=0.3)


# Accuracy-ის გრაფიკი
axes[1].plot(
    epochs,
    history["train_accuracy"],
    label="Train accuracy"
)

axes[1].plot(
    epochs,
    history["validation_accuracy"],
    label="Validation accuracy"
)

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("სიზუსტის ცვლილება")
axes[1].legend()
axes[1].grid(alpha=0.3)


figure.tight_layout()

plt.show()