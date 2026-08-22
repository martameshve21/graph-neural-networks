import argparse
from pathlib import Path

import torch
import torch.nn as nn


# PyTorch 2.6+ და OGB თავსებადობის შესწორება
_original_torch_load = torch.load


def torch_load_compatibility(*args, **kwargs):
    kwargs.setdefault(
        "weights_only",
        False
    )

    return _original_torch_load(
        *args,
        **kwargs
    )


torch.load = torch_load_compatibility


from ogb.graphproppred import (
    PygGraphPropPredDataset
)

from torch.optim import Adam

from model import GraphClassifier

from utils import (
    calculate_binary_metrics,
    create_data_loaders,
    get_device,
    save_json,
    set_seed
)

from ogb.graphproppred import (
    PygGraphPropPredDataset
)

from torch.optim import Adam

from model import GraphClassifier

from utils import (
    calculate_binary_metrics,
    create_data_loaders,
    get_device,
    save_json,
    set_seed
)


ROOT_DIRECTORY = Path(__file__).resolve().parent

DATA_DIRECTORY = ROOT_DIRECTORY / "data"
WEIGHTS_DIRECTORY = ROOT_DIRECTORY / "weights"
RESULTS_DIRECTORY = ROOT_DIRECTORY / "results"


def run_epoch(
    model,
    data_loader,
    loss_function,
    device,
    optimizer=None
):
    """
    ერთი train ან evaluation epoch-ის შესრულება.

    თუ optimizer გადმოცემულია, ფუნქცია სწავლების
    რეჟიმში მუშაობს. წინააღმდეგ შემთხვევაში,
    evaluation რეჟიმში.
    """

    is_training = optimizer is not None

    model.train(is_training)

    total_loss = 0.0
    total_graphs = 0

    all_labels = []
    all_probabilities = []

    for batch in data_loader:
        batch = batch.to(device)

        labels = batch.y.view(-1).float()

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(batch)

            loss = loss_function(
                logits,
                labels
            )

            if is_training:
                loss.backward()
                optimizer.step()

        probabilities = torch.sigmoid(
            logits
        )

        number_of_graphs = batch.num_graphs

        total_loss += (
            loss.item() * number_of_graphs
        )

        total_graphs += number_of_graphs

        all_labels.extend(
            labels.detach().cpu().tolist()
        )

        all_probabilities.extend(
            probabilities
            .detach()
            .cpu()
            .tolist()
        )

    metrics = calculate_binary_metrics(
        labels=all_labels,
        probabilities=all_probabilities
    )

    metrics["loss"] = (
        total_loss / total_graphs
    )

    return (
        metrics,
        all_labels,
        all_probabilities
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Train a GCN model on ogbg-molhiv"
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256
    )

    parser.add_argument(
        "--num-gcn-layers",
        type=int,
        default=5
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    # argparse-ის პარამეტრების dictionary
    configuration = vars(arguments)

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

    print(f"არჩეული მოწყობილობა: {device}")
    print("ogbg-molhiv მონაცემების ჩატვირთვა...")

    dataset = PygGraphPropPredDataset(
        name="ogbg-molhiv",
        root=DATA_DIRECTORY
    )

    (
        train_loader,
        validation_loader,
        test_loader
    ) = create_data_loaders(
        dataset=dataset,
        batch_size=arguments.batch_size
    )

    model = GraphClassifier(
        hidden_dim=arguments.hidden_dim,
        num_gcn_layers=(
            arguments.num_gcn_layers
        ),
        dropout=arguments.dropout
    )

    model = model.to(device)

    optimizer = Adam(
        model.parameters(),
        lr=arguments.learning_rate
    )

    loss_function = nn.BCEWithLogitsLoss()

    training_history = []

    best_validation_auc = float("-inf")

    checkpoint_path = (
        WEIGHTS_DIRECTORY /
        "best_model.pt"
    )

    for epoch in range(
        1,
        arguments.epochs + 1
    ):
        train_metrics, _, _ = run_epoch(
            model=model,
            data_loader=train_loader,
            loss_function=loss_function,
            device=device,
            optimizer=optimizer
        )

        validation_metrics, _, _ = run_epoch(
            model=model,
            data_loader=validation_loader,
            loss_function=loss_function,
            device=device
        )

        epoch_results = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics
        }

        training_history.append(
            epoch_results
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

        # საუკეთესო მოდელის შენახვა
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
                "ახალი საუკეთესო მოდელი "
                "შენახულია: "
                f"{checkpoint_path}"
            )

        # ისტორია ყოველი epoch-ის შემდეგ ინახება
        save_json(
            data=training_history,
            path=(
                RESULTS_DIRECTORY /
                "history.json"
            )
        )

    # საუკეთესო checkpoint-ის ჩატვირთვა
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # საუკეთესო მოდელის შეფასება test set-ზე
    (
        test_metrics,
        test_labels,
        test_probabilities
    ) = run_epoch(
        model=model,
        data_loader=test_loader,
        loss_function=loss_function,
        device=device
    )

    final_metrics = {
        "best_epoch": checkpoint["epoch"],

        "best_validation": (
            checkpoint["validation_metrics"]
        ),

        "test": test_metrics,

        "config": configuration
    }

    test_predictions = [
        {
            "label": int(label),
            "probability": probability
        }
        for label, probability in zip(
            test_labels,
            test_probabilities
        )
    ]

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
        f"Test accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )
    print(
        f"Test ROC-AUC: "
        f"{test_metrics['roc_auc']:.4f}"
    )
    print(
        f"წონები შენახულია: "
        f"{checkpoint_path}"
    )
    print(
        "ვიზუალიზაციისთვის გაუშვი: "
        "python visualize.py --show"
    )


if __name__ == "__main__":
    main()