from pathlib import Path

import torch
import torch_geometric.transforms as T

from torch_geometric.datasets import Planetoid


ROOT_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = ROOT_DIRECTORY / "data"


def load_cora_link_prediction_data(
    seed: int = 42
):
    """
    Cora-ს მონაცემების ჩატვირთვა და წიბოების მიხედვით დაყოფა.

    დაბრუნებული ობიექტებია:
    dataset,
    train_data,
    validation_data,
    test_data.
    """

    # RandomLinkSplit-ის განმეორებადობა
    torch.manual_seed(seed)

    transform = T.Compose([
        T.NormalizeFeatures(),

        T.RandomLinkSplit(
            num_val=0.05,
            num_test=0.10,
            is_undirected=True,

            # სასწავლო მონაცემებში უარყოფითი
            # მაგალითების დამატება
            add_negative_train_samples=True,

            # დადებითი და უარყოფითი მაგალითების
            # თანაფარდობა იქნება 1:1
            neg_sampling_ratio=1.0
        )
    ])

    dataset = Planetoid(
        root=DATA_DIRECTORY,
        name="Cora",
        transform=transform
    )

    # მნიშვნელოვანია, რომ dataset[0] მხოლოდ ერთხელ გამოვიძახოთ,
    # რადგან transform შემთხვევით დაყოფას ასრულებს.
    train_data, validation_data, test_data = (
        dataset[0]
    )

    return (
        dataset,
        train_data,
        validation_data,
        test_data
    )


def print_dataset_information(
    dataset,
    train_data,
    validation_data,
    test_data
):
    """
    მონაცემთა ნაკრების ძირითადი ინფორმაციის დაბეჭდვა.
    """

    print()
    print("Cora მონაცემთა ნაკრები")
    print("----------------------")

    print(
        f"წვეროების მახასიათებლების რაოდენობა: "
        f"{dataset.num_features}"
    )

    print(
        f"წვეროების კლასების რაოდენობა: "
        f"{dataset.num_classes}"
    )

    print(
        f"წვეროების საერთო რაოდენობა: "
        f"{train_data.num_nodes}"
    )

    print()
    print("წიბოების დაყოფა")
    print("----------------------")

    print(
        f"Train message-passing edges: "
        f"{train_data.edge_index.size(1)}"
    )

    print(
        f"Train prediction examples: "
        f"{train_data.edge_label.numel()}"
    )

    print(
        f"Validation prediction examples: "
        f"{validation_data.edge_label.numel()}"
    )

    print(
        f"Test prediction examples: "
        f"{test_data.edge_label.numel()}"
    )