from pathlib import Path

import compatibility

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from data_loader import (
    load_cora_link_prediction_data
)

from model import (
    create_model_from_checkpoint
)


ROOT_DIRECTORY = Path(__file__).resolve().parent

CHECKPOINT_PATH = (
    ROOT_DIRECTORY /
    "weights" /
    "best_model.pt"
)


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "weights/best_model.pt ვერ მოიძებნა. "
            "ჯერ გაუშვი: python train.py"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    configuration = checkpoint["config"]

    (
        _,
        _,
        _,
        test_data
    ) = load_cora_link_prediction_data(
        seed=configuration["seed"]
    )

    test_data = test_data.to(device)

    model = create_model_from_checkpoint(
        checkpoint=checkpoint,
        device=device
    )

    model.eval()

    loss_function = (
        nn.BCEWithLogitsLoss()
    )

    with torch.no_grad():
        logits = model(test_data)

        labels = (
            test_data
            .edge_label
            .float()
        )

        loss = loss_function(
            logits,
            labels
        )

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= 0.5
        ).long()

    labels_numpy = (
        labels.cpu().numpy()
    )

    probabilities_numpy = (
        probabilities.cpu().numpy()
    )

    predictions_numpy = (
        predictions.cpu().numpy()
    )

    auc = roc_auc_score(
        labels_numpy,
        probabilities_numpy
    )

    accuracy = accuracy_score(
        labels_numpy,
        predictions_numpy
    )

    matrix = confusion_matrix(
        labels_numpy,
        predictions_numpy
    )

    report = classification_report(
        labels_numpy,
        predictions_numpy,
        target_names=[
            "No edge",
            "Edge"
        ],
        digits=4
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Test loss: "
        f"{loss.item():.4f}"
    )

    print(
        f"Test accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Test ROC-AUC: "
        f"{auc:.4f}"
    )

    print()
    print("Confusion matrix:")
    print(matrix)

    print()
    print("Classification report:")
    print(report)


if __name__ == "__main__":
    main()