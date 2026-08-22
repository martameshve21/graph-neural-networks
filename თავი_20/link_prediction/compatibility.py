import torch

from torch_geometric.data.data import (
    Data,
    DataEdgeAttr,
    DataTensorAttr
)

from torch_geometric.data.storage import (
    GlobalStorage
)


torch.serialization.add_safe_globals([
    Data,
    DataEdgeAttr,
    DataTensorAttr,
    GlobalStorage
])