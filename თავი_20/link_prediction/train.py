import argparse
from pathlib import Path

import compatibility

import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score
from torch.optim import Adam

from data_loader import (
    load_cora_link_prediction_data,
    print_dataset_information
)

from model import LinkPredictor


ROOT_DIRECTORY = Path(__file__).resolve().parent

WEIGHTS_DIRECTORY = (
    ROOT_DIRECTORY / "weights"
)

RESULTS_DIRECTORY = (
    ROOT_DIRECTORY / "results"
)


def set_seed(seed: int):
    """
    ექსპერიმენტის განმეორებადობის უზრუნველყოფა.
    """

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


def calculate_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor
):
    """
    ROC-AUC-ის გამოთვლა.
    """

    probabilities = torch.sigmoid(
        logits
    )

    auc = roc_auc_score(
        labels.detach().cpu().numpy(),
        probabilities.detach().cpu().numpy()
    )

    return float(auc)


def train_step(
    model,
    optimizer,
    loss_function,
    train_data,
    device
):
    """
    ერთი სასწავლო epoch.
    """

    model.train()

    train_data = train_data.to(device)

    optimizer.zero_grad()

    logits = model(train_data)

    labels = (
        train_data
        .edge_label
        .float()
    )

    loss = loss_function(
        logits,
        labels
    )

    loss.backward()

    optimizer.step()

    auc = calculate_metrics(
        logits=logits,
        labels=labels
    )

    return {
        "loss": float(loss.item()),
        "roc_auc": auc
    }


def evaluation_step(
    model,
    loss_function,
    evaluation_data,
    device
):
    """
    Validation ან test შეფასება.
    """

    model.eval()

    evaluation_data = (
        evaluation_data.to(device)
    )

    with torch.no_grad():
        logits = model(
            evaluation_data
        )

        labels = (
            evaluation_data
            .edge_label
            .float()
        )

        loss = loss_function(
            logits,
            labels
        )

        auc = calculate_metrics(
            logits=logits,
            labels=labels
        )

        probabilities = torch.sigmoid(
            logits
        )

    return {
        "loss": float(loss.item()),
        "roc_auc": auc
    }, labels, probabilities


def save_json(data, path: Path):
    """
    JSON ფაილის შენახვა.
    """

    import json

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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Cora link prediction training"
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128
    )

    parser.add_argument(
        "--num-gcn-layers",
        type=int,
        default=3
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    set_seed(arguments.seed)

    device = get_device()

    WEIGHTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"არჩეული მოწყობილობა: {device}"
    )

    (
        dataset,
        train_data,
        validation_data,
        test_data
    ) = load_cora_link_prediction_data(
        seed=arguments.seed
    )

    print_dataset_information(
        dataset=dataset,
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data
    )

    configuration = {
        "input_dim": dataset.num_features,
        "hidden_dim": arguments.hidden_dim,
        "num_gcn_layers": (
            arguments.num_gcn_layers
        ),
        "dropout": arguments.dropout,
        "learning_rate": (
            arguments.learning_rate
        ),
        "epochs": arguments.epochs,
        "seed": arguments.seed
    }

    model = LinkPredictor(
        input_dim=configuration["input_dim"],
        hidden_dim=configuration["hidden_dim"],
        num_gcn_layers=(
            configuration["num_gcn_layers"]
        ),
        dropout=configuration["dropout"]
    )

    model = model.to(device)

    optimizer = Adam(
        model.parameters(),
        lr=arguments.learning_rate
    )

    loss_function = (
        nn.BCEWithLogitsLoss()
    )

    history = []

    best_validation_auc = float("-inf")

    checkpoint_path = (
        WEIGHTS_DIRECTORY /
        "best_model.pt"
    )

    for epoch in range(
        1,
        arguments.epochs + 1
    ):
        train_metrics = train_step(
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            train_data=train_data,
            device=device
        )

        (
            validation_metrics,
            _,
            _
        ) = evaluation_step(
            model=model,
            loss_function=loss_function,
            evaluation_data=validation_data,
            device=device
        )

        epoch_information = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics
        }

        history.append(
            epoch_information
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train loss: "
            f"{train_metrics['loss']:.4f} | "
            f"Train AUC: "
            f"{train_metrics['roc_auc']:.4f} | "
            f"Validation loss: "
            f"{validation_metrics['loss']:.4f} | "
            f"Validation AUC: "
            f"{validation_metrics['roc_auc']:.4f}"
        )

        current_validation_auc = (
            validation_metrics["roc_auc"]
        )

        if (
            current_validation_auc >
            best_validation_auc
        ):
            best_validation_auc = (
                current_validation_auc
            )

            checkpoint = {
                "epoch": epoch,

                "model_state_dict": (
                    model.state_dict()
                ),

                "optimizer_state_dict": (
                    optimizer.state_dict()
                ),

                "validation_metrics": (
                    validation_metrics
                ),

                "config": configuration
            }

            torch.save(
                checkpoint,
                checkpoint_path
            )

            print(
                "  საუკეთესო მოდელი შენახულია."
            )

        save_json(
            data=history,
            path=(
                RESULTS_DIRECTORY /
                "history.json"
            )
        )

    # საუკეთესო მოდელის ჩატვირთვა
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    (
        test_metrics,
        test_labels,
        test_probabilities
    ) = evaluation_step(
        model=model,
        loss_function=loss_function,
        evaluation_data=test_data,
        device=device
    )

    final_metrics = {
        "best_epoch": checkpoint["epoch"],

        "best_validation": (
            checkpoint[
                "validation_metrics"
            ]
        ),

        "test": test_metrics,

        "config": configuration
    }

    test_predictions = []

    edge_label_index = (
        test_data
        .edge_label_index
        .detach()
        .cpu()
    )

    labels_list = (
        test_labels
        .detach()
        .cpu()
        .tolist()
    )

    probabilities_list = (
        test_probabilities
        .detach()
        .cpu()
        .tolist()
    )

    for index in range(
        len(labels_list)
    ):
        source_node = int(
            edge_label_index[0, index]
        )

        destination_node = int(
            edge_label_index[1, index]
        )

        test_predictions.append({
            "source_node": source_node,
            "destination_node": (
                destination_node
            ),
            "label": int(
                labels_list[index]
            ),
            "probability": float(
                probabilities_list[index]
            )
        })

    save_json(
        data=final_metrics,
        path=(
            RESULTS_DIRECTORY /
            "metrics.json"
        )
    )

    save_json(
        data=test_predictions,
        path=(
            RESULTS_DIRECTORY /
            "test_predictions.json"
        )
    )

    print()
    print("სწავლება დასრულებულია.")

    print(
        f"საუკეთესო epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"საუკეთესო validation AUC: "
        f"{checkpoint['validation_metrics']['roc_auc']:.4f}"
    )

    print(
        f"Test ROC-AUC: "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"წონები შენახულია: "
        f"{checkpoint_path}"
    )


if __name__ == "__main__":
    main()