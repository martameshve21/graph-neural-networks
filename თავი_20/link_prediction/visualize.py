import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix
)


ROOT_DIRECTORY = Path(__file__).resolve().parent

RESULTS_DIRECTORY = (
    ROOT_DIRECTORY / "results"
)


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} ვერ მოიძებნა. "
            "ჯერ გაუშვი: python train.py"
        )

    with path.open(
        mode="r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def plot_training_history(
    history
):
    epochs = [
        row["epoch"]
        for row in history
    ]

    train_losses = [
        row["train"]["loss"]
        for row in history
    ]

    validation_losses = [
        row["validation"]["loss"]
        for row in history
    ]

    train_auc = [
        row["train"]["roc_auc"]
        for row in history
    ]

    validation_auc = [
        row["validation"]["roc_auc"]
        for row in history
    ]

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13, 4.5)
    )

    axes[0].plot(
        epochs,
        train_losses,
        label="Train"
    )

    axes[0].plot(
        epochs,
        validation_losses,
        label="Validation"
    )

    axes[0].set_title(
        "Training and validation loss"
    )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        epochs,
        train_auc,
        label="Train"
    )

    axes[1].plot(
        epochs,
        validation_auc,
        label="Validation"
    )

    axes[1].set_title(
        "Training and validation ROC-AUC"
    )

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.tight_layout()

    figure.savefig(
        RESULTS_DIRECTORY /
        "training_curves.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(figure)


def plot_test_results(
    test_predictions
):
    labels = np.array([
        row["label"]
        for row in test_predictions
    ])

    probabilities = np.array([
        row["probability"]
        for row in test_predictions
    ])

    predicted_classes = (
        probabilities >= 0.5
    ).astype(int)

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, 5)
    )

    # Confusion matrix
    matrix = confusion_matrix(
        labels,
        predicted_classes
    )

    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "No edge",
            "Edge"
        ]
    ).plot(
        ax=axes[0],
        colorbar=False
    )

    axes[0].set_title(
        "Test confusion matrix"
    )

    # ROC curve
    RocCurveDisplay.from_predictions(
        labels,
        probabilities,
        ax=axes[1],
        name="GCN Link Predictor"
    )

    axes[1].plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray"
    )

    axes[1].set_title(
        "Test ROC curve"
    )

    # ალბათობების განაწილება
    sns.histplot(
        probabilities[labels == 0],
        bins=30,
        alpha=0.55,
        label="No edge",
        ax=axes[2]
    )

    sns.histplot(
        probabilities[labels == 1],
        bins=30,
        alpha=0.55,
        label="Edge",
        ax=axes[2]
    )

    axes[2].axvline(
        x=0.5,
        color="black",
        linestyle="--",
        label="Threshold"
    )

    axes[2].set_title(
        "Predicted edge probabilities"
    )

    axes[2].set_xlabel(
        "Probability of edge"
    )

    axes[2].legend()

    figure.tight_layout()

    figure.savefig(
        RESULTS_DIRECTORY /
        "test_results.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(figure)


def show_images():
    image_names = [
        "training_curves.png",
        "test_results.png"
    ]

    for image_name in image_names:
        image_path = (
            RESULTS_DIRECTORY /
            image_name
        )

        image = plt.imread(
            image_path
        )

        plt.figure(
            figsize=(14, 6)
        )

        plt.imshow(image)
        plt.title(image_name)
        plt.axis("off")

    plt.show()


def main():
    history = read_json(
        RESULTS_DIRECTORY /
        "history.json"
    )

    test_predictions = read_json(
        RESULTS_DIRECTORY /
        "test_predictions.json"
    )

    plot_training_history(
        history
    )

    plot_test_results(
        test_predictions
    )

    print(
        "ვიზუალიზაციები შენახულია:"
    )

    print(
        RESULTS_DIRECTORY /
        "training_curves.png"
    )

    print(
        RESULTS_DIRECTORY /
        "test_results.png"
    )

    show_images()


if __name__ == "__main__":
    main()