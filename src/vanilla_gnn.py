import argparse

import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv

from src.datasets.german import german
from src.datasets.pokec import pokec_z, pokec_n
from src.utils import device


class VanillaNode(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        block=GCNConv,
    ):
        super().__init__()

        self.block = block

        self.convs = torch.nn.ModuleList()
        self.convs.append(block(in_channels, hidden_channels[0]))
        for i in range(1, len(hidden_channels)):
            if block == GATConv:
                self.convs.append(block(hidden_channels[i - 1], hidden_channels[i], 1))
            else:
                self.convs.append(block(hidden_channels[i - 1], hidden_channels[i]))
        self.convs.append(block(hidden_channels[-1], out_channels))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = x.relu()

        x = self.convs[-1](x, edge_index)
        x = x.sigmoid()

        return x.squeeze()

    def embedding(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = x.relu()

        return x

    def __repr__(self):
        # Ex. GCNConv(16,32,32,1)
        return f"{self.__class__.__name__}({self.convs[0].in_channels},{','.join([str(conv.out_channels) for conv in self.convs])})"


def run_node_gnn(model, data, mask, optimizer=None):
    if optimizer:
        model.train()
        optimizer.zero_grad()

    out = model(data)[mask]
    pred = data.y[mask]

    loss = F.binary_cross_entropy(out, pred)

    if optimizer:
        loss.backward()
        optimizer.step()

    correct = out.round().eq(pred).sum().item()
    count = mask.sum().item()
    acc = correct / count

    return loss, acc


def train_model(model, dataset, optimizer, epochs):
    dataset_name = dataset.__class__.__name__
    model_name = repr(model)
    optimizer_name = optimizer.__class__.__name__

    print(f"Training {dataset_name} model...")

    data = dataset[0]
    best_model = None
    for epoch in range(epochs):
        train_loss, train_acc = run_node_gnn(model, data, data.train_mask, optimizer)
        val_loss, val_acc = run_node_gnn(model, data, data.val_mask)

        print(
            f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if best_model is None or val_loss < best_model[0]:
            best_model = (val_loss, val_acc, model.state_dict())

    print()

    model.load_state_dict(best_model[2])
    loss, acc = run_node_gnn(model, data, data.test_mask)
    print(f"Test Loss: {loss:.4f}, Test Acc: {acc:.4f}")
    print()

    # save model
    torch.save(
        model, f"models/{dataset_name}_{model_name}_{optimizer_name}_{epochs}.pt"
    )


def read_args():
    argparser = argparse.ArgumentParser()

    # Dataset
    argparser.add_argument(
        "--dataset",
        type=str,
        default="german",
        choices=["german", "pokec_n", "pokec_z"],
    )

    # Model
    argparser.add_argument(
        "--model",
        type=str,
        default="GCNConv",
        choices=["GCNConv", "GATConv", "SAGEConv", "GINConv"],
    )
    argparser.add_argument("--hidden", type=int, nargs="+", default=[16])

    # Training
    argparser.add_argument("--epochs", type=int, default=50)
    argparser.add_argument("--lr", type=float, default=5e-3)
    argparser.add_argument("--weight_decay", type=float, default=1e-3)

    return argparser.parse_args()


if __name__ == "__main__":
    args = read_args()

    dataset = None
    if args.dataset == "german":
        dataset = german
    elif args.dataset == "pokec_n":
        dataset = pokec_n
    elif args.dataset == "pokec_z":
        dataset = pokec_z

    block = None
    if args.model == "GCNConv":
        block = GCNConv
    elif args.model == "GATConv":
        block = GATConv
    elif args.model == "SAGEConv":
        block = SAGEConv
    elif args.model == "GINConv":
        block = GINConv

    model = VanillaNode(
        in_channels=dataset.num_features,
        hidden_channels=args.hidden,
        out_channels=1,
        block=block,
    ).to(device)

    lr, weight_decay, epochs = args.lr, args.weight_decay, args.epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_model(model, dataset, optimizer, epochs)
