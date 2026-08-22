import torch
import torch.nn as nn
import torch.nn.functional as F

from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool


class GraphClassifier(nn.Module):
    """
    GCN მოდელი მოლეკულების ბინარული კლასიფიკაციისთვის.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_gcn_layers: int = 5,
        dropout: float = 0.5,
    ):
        super().__init__()

        if num_gcn_layers < 1:
            raise ValueError(
                "num_gcn_layers must be at least 1"
            )

        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers
        self.dropout = dropout

        # ატომების კატეგორიული მახასიათებლების
        # ვექტორულ წარმოდგენებად გარდაქმნა
        self.atom_encoder = AtomEncoder(
            emb_dim=hidden_dim
        )

        # GCN ფენები
        self.convolutions = nn.ModuleList([
            GCNConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim
            )
            for _ in range(num_gcn_layers)
        ])

        # ბოლო GCN ფენის გარდა ყველა ფენისთვის
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim)
            for _ in range(num_gcn_layers - 1)
        ])

        # თითოეული მოლეკულისთვის ერთი logit
        self.classifier = nn.Linear(
            in_features=hidden_dim,
            out_features=1
        )

    def forward(
        self,
        data: Batch
    ) -> torch.Tensor:

        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        # ატომების საწყისი წარმოდგენები
        x = self.atom_encoder(x)

        # ყველა GCN ფენა უკანასკნელის გარდა
        for convolution, batch_norm in zip(
            self.convolutions[:-1],
            self.batch_norms
        ):
            x = convolution(
                x,
                edge_index
            )

            x = batch_norm(x)
            x = F.relu(x)

            x = F.dropout(
                x,
                p=self.dropout,
                training=self.training
            )

        # ბოლო GCN ფენა
        x = self.convolutions[-1](
            x,
            edge_index
        )

        # ყველა ატომის წარმოდგენა ერთიანდება
        # მთლიანი მოლეკულის წარმოდგენად
        graph_embeddings = global_mean_pool(
            x,
            batch
        )

        # ბინარული კლასიფიკაცია
        logits = self.classifier(
            graph_embeddings
        )

        return logits.squeeze(-1)


def create_model_from_checkpoint(
    checkpoint: dict,
    device: torch.device
) -> GraphClassifier:
    """
    შენახული checkpoint-იდან მოდელის აღდგენა.
    """

    config = checkpoint["config"]

    model = GraphClassifier(
        hidden_dim=config["hidden_dim"],
        num_gcn_layers=config["num_gcn_layers"],
        dropout=config["dropout"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return model.to(device)