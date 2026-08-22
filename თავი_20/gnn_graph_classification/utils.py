import json
import random
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score
)

from torch_geometric.loader import DataLoader


def set_seed(seed: int) -> None:
    """
    ექსპერიმენტის განმეორებადობის უზრუნველყოფა.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    GPU-ის არსებობის შემთხვევაში CUDA-ს არჩევა.
    """

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def create_data_loaders(
    dataset,
    batch_size: int
):
    """
    OGB-ის ოფიციალური train/validation/test დაყოფა.
    """

    split_indices = dataset.get_idx_split()

    train_dataset = dataset[
        split_indices["train"]
    ]

    validation_dataset = dataset[
        split_indices["valid"]
    ]

    test_dataset = dataset[
        split_indices["test"]
    ]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return (
        train_loader,
        validation_loader,
        test_loader
    )


def calculate_binary_metrics(
    labels: list[float],
    probabilities: list[float]
) -> dict:
    """
    Accuracy-სა და ROC-AUC-ის გამოთვლა.
    """

    labels_array = np.asarray(labels)
    probabilities_array = np.asarray(
        probabilities
    )

    predictions = (
        probabilities_array >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        labels_array,
        predictions
    )

    roc_auc = roc_auc_score(
        labels_array,
        probabilities_array
    )

    return {
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc)
    }


def save_json(
    data: dict | list,
    path: Path
) -> None:
    """
    შედეგების JSON ფაილში შენახვა.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        mode="w",
        encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file,
            indent=2
        )