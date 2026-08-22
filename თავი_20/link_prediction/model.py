import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class LinkPredictor(nn.Module):
    """
    GCN მოდელი Cora-ს გრაფში წიბოების პროგნოზირებისთვის.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_gcn_layers: int = 3,
        dropout: float = 0.5
    ):
        super().__init__()

        if num_gcn_layers < 1:
            raise ValueError(
                "num_gcn_layers must be at least 1"
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers
        self.dropout = dropout

        self.convolutions = nn.ModuleList()

        # პირველი GCN ფენა:
        # 1433 -> 128
        self.convolutions.append(
            GCNConv(
                in_channels=input_dim,
                out_channels=hidden_dim
            )
        )

        # დარჩენილი GCN ფენები:
        # 128 -> 128
        for _ in range(
            num_gcn_layers - 1
        ):
            self.convolutions.append(
                GCNConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim
                )
            )

        # თითოეული GCN ფენისთვის BatchNorm
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim)
            for _ in range(num_gcn_layers)
        ])

    def encode_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        ყველა წვეროს ვექტორული წარმოდგენის გამოთვლა.
        """

        # ყველა GCN ფენა უკანასკნელის გარდა
        for convolution, batch_norm in zip(
            self.convolutions[:-1],
            self.batch_norms[:-1]
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

        x = self.batch_norms[-1](x)

        return x

    def decode_edges(
        self,
        node_embeddings: torch.Tensor,
        edge_label_index: torch.Tensor
    ) -> torch.Tensor:
        """
        წვეროთა წყვილებისთვის dot product-ის გამოთვლა.
        """

        source_node_indices = (
            edge_label_index[0]
        )

        destination_node_indices = (
            edge_label_index[1]
        )

        source_embeddings = (
            node_embeddings[
                source_node_indices
            ]
        )

        destination_embeddings = (
            node_embeddings[
                destination_node_indices
            ]
        )

        # თითოეული წყვილის dot product
        edge_logits = torch.sum(
            source_embeddings *
            destination_embeddings,
            dim=-1
        )

        return edge_logits

    def forward(
        self,
        data: Data
    ) -> torch.Tensor:
        """
        მთლიანი forward pass.
        """

        node_embeddings = self.encode_nodes(
            x=data.x,
            edge_index=data.edge_index
        )

        edge_logits = self.decode_edges(
            node_embeddings=node_embeddings,
            edge_label_index=(
                data.edge_label_index
            )
        )

        return edge_logits


def create_model_from_checkpoint(
    checkpoint: dict,
    device: torch.device
) -> LinkPredictor:
    """
    checkpoint-იდან მოდელის აღდგენა.
    """

    configuration = checkpoint["config"]

    model = LinkPredictor(
        input_dim=configuration["input_dim"],
        hidden_dim=configuration["hidden_dim"],
        num_gcn_layers=(
            configuration["num_gcn_layers"]
        ),
        dropout=configuration["dropout"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return model.to(device)