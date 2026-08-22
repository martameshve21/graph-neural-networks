import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import torch

from torch_geometric.data.data import (
    Data,
    DataEdgeAttr,
    DataTensorAttr
)

from torch_geometric.data.storage import (
    GlobalStorage
)


# PyTorch 2.6+ და OGB თავსებადობა
torch.serialization.add_safe_globals([
    Data,
    DataEdgeAttr,
    DataTensorAttr,
    GlobalStorage
])


from ogb.graphproppred import (
    PygGraphPropPredDataset
)

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix
)

from torch_geometric.utils import (
    to_networkx
)

from model import create_model_from_checkpoint
from utils import get_device

ROOT_DIRECTORY = Path(__file__).resolve().parent

DATA_DIRECTORY = ROOT_DIRECTORY / "data"
WEIGHTS_DIRECTORY = ROOT_DIRECTORY / "weights"
RESULTS_DIRECTORY = ROOT_DIRECTORY / "results"


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"ფაილი ვერ მოიძებნა: {path}\n"
            "ჯერ გაუშვი: python train.py"
        )

    with path.open(
        mode="r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def plot_training_history(
    history: list
) -> None:
    epochs = [
        row["epoch"]
        for row in history
    ]

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(16, 4.5)
    )

    metric_information = [
        ("loss", "Loss"),
        ("accuracy", "Accuracy"),
        ("roc_auc", "ROC-AUC")
    ]

    for axis, (
        metric_name,
        metric_title
    ) in zip(axes, metric_information):

        train_values = [
            row["train"][metric_name]
            for row in history
        ]

        validation_values = [
            row["validation"][metric_name]
            for row in history
        ]

        axis.plot(
            epochs,
            train_values,
            label="Train"
        )

        axis.plot(
            epochs,
            validation_values,
            label="Validation"
        )

        axis.set_title(metric_title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric_title)
        axis.grid(alpha=0.25)
        axis.legend()

    figure.tight_layout()

    output_path = (
        RESULTS_DIRECTORY /
        "training_curves.png"
    )

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(figure)


def plot_test_predictions(
    predictions: list
) -> None:
    labels = np.array([
        row["label"]
        for row in predictions
    ])

    probabilities = np.array([
        row["probability"]
        for row in predictions
    ])

    predicted_classes = (
        probabilities >= 0.5
    ).astype(int)

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 4.5)
    )

    matrix = confusion_matrix(
        labels,
        predicted_classes
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Inactive",
            "Active"
        ]
    )

    display.plot(
        ax=axes[0],
        colorbar=False
    )

    axes[0].set_title(
        "Test Confusion Matrix"
    )

    sns.histplot(
        probabilities[labels == 0],
        bins=25,
        alpha=0.55,
        label="Inactive",
        ax=axes[1]
    )

    sns.histplot(
        probabilities[labels == 1],
        bins=25,
        alpha=0.55,
        label="Active",
        ax=axes[1]
    )

    axes[1].axvline(
        x=0.5,
        color="black",
        linestyle="--",
        label="Classification threshold"
    )

    axes[1].set_title(
        "Predicted HIV Activity Probabilities"
    )

    axes[1].set_xlabel(
        "Probability of active class"
    )

    axes[1].legend()

    figure.tight_layout()

    output_path = (
        RESULTS_DIRECTORY /
        "test_predictions.png"
    )

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(figure)


def plot_sample_molecules(
    dataset,
    number_of_graphs: int = 4
) -> None:
    figure, axes = plt.subplots(
        nrows=1,
        ncols=number_of_graphs,
        figsize=(4 * number_of_graphs, 4)
    )

    axes = np.atleast_1d(axes)

    for index, axis in enumerate(axes):
        molecule_data = dataset[index]

        graph = to_networkx(
            molecule_data,
            to_undirected=True
        )

        positions = nx.spring_layout(
            graph,
            seed=42
        )

        nx.draw_networkx(
            graph,
            pos=positions,
            node_size=90,
            with_labels=False,
            width=0.8,
            ax=axis
        )

        label = int(
            molecule_data.y.item()
        )

        axis.set_title(
            f"Molecule {index}\n"
            f"Label: {label}"
        )

        axis.axis("off")

    figure.tight_layout()

    output_path = (
        RESULTS_DIRECTORY /
        "sample_molecules.png"
    )

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(figure)


def show_saved_images() -> None:
    image_names = [
        "training_curves.png",
        "test_predictions.png",
        "sample_molecules.png"
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
            figsize=(12, 6)
        )

        plt.imshow(image)
        plt.title(image_name)
        plt.axis("off")

    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "შენახული მოდელის შედეგების "
            "ვიზუალიზაცია"
        )
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="შექმნილი გრაფიკების ეკრანზე ჩვენება"
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    history = read_json(
        RESULTS_DIRECTORY /
        "history.json"
    )

    test_predictions = read_json(
        RESULTS_DIRECTORY /
        "test_predictions.json"
    )

    checkpoint_path = (
        WEIGHTS_DIRECTORY /
        "best_model.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "weights/best_model.pt ვერ მოიძებნა.\n"
            "ჯერ გაუშვი: python train.py"
        )

    device = get_device()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    # მოდელის წარმატებით აღდგენის შემოწმება
    model = create_model_from_checkpoint(
        checkpoint=checkpoint,
        device=device
    )

    model.eval()

    dataset = PygGraphPropPredDataset(
        name="ogbg-molhiv",
        root=DATA_DIRECTORY
    )

    plot_training_history(
        history
    )

    plot_test_predictions(
        test_predictions
    )

    plot_sample_molecules(
        dataset
    )

    print(
        "მოდელი წარმატებით ჩაიტვირთა."
    )

    print(
        f"საუკეთესო epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"მოდელის კლასი: "
        f"{model.__class__.__name__}"
    )

    print(
        f"ვიზუალიზაციები შენახულია: "
        f"{RESULTS_DIRECTORY}"
    )

    if arguments.show:
        show_saved_images()


if __name__ == "__main__":
    main()