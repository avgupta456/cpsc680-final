import torch_geometric.transforms as T

from src.utils import device

transform = T.Compose([T.ToDevice(device), T.ToUndirected()])
link_transform = T.Compose(
    [
        T.ToDevice(device),
        T.ToUndirected(),
        T.RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            is_undirected=True,
            add_negative_train_samples=False,
        ),
    ]
)
