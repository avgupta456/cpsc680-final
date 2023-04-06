import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.utils import negative_sampling

from src.node_gnn import VanillaNode


class VanillaEdge(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, block=GCNConv):
        super().__init__()

        self.block_name = {
            GCNConv: "GCNConv",
            GATConv: "GATConv",
            SAGEConv: "SAGEConv",
            GINConv: "GINConv",
        }[block]

        # NOTE: Out channels doesn't actually matter since we only use the embedding
        self.encoder = VanillaNode(in_channels, 1, hidden_channels, block)

    def encode(self, x, edge_index):
        return self.encoder.embedding(x, edge_index)

    def decode(self, z, edge_label_index):
        x1 = z[edge_label_index[0]]
        x2 = z[edge_label_index[1]]
        return (x1 * x2).sum(dim=1).sigmoid()

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

    def __repr__(self):
        # Ex. Edge_GCNConv(16,32,32)
        node_repr = ",".join(repr(self.encoder).split("_")[1].split(",")[:-1]) + ")"
        return f"Edge_{node_repr}"


def run_edge_gnn(model, data, optimizer=None):
    # NOTE: Instead of mask, we pass three separate data splits with different edge_label_index
    if optimizer:
        model.train()
        optimizer.zero_grad()

        # We perform a new round of negative sampling for every training epoch:
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=(data.x.shape[0], data.x.shape[0]),
            num_neg_samples=data.edge_label_index.shape[1],
            method="sparse",
        )

        edge_label_index = torch.cat([data.edge_label_index, neg_edge_index], dim=-1)
        edge_label = torch.cat(
            [data.edge_label, data.edge_label.new_zeros(neg_edge_index.size(1))], dim=0
        )

        out = model(data.x, data.edge_index, edge_label_index)
        pred = edge_label

        loss = F.binary_cross_entropy_with_logits(out, pred)

        loss.backward()
        optimizer.step()
    else:
        out = model(data.x, data.edge_index, data.edge_label_index)
        pred = data.edge_label

        loss = F.binary_cross_entropy_with_logits(out, pred)

    correct = out.round().eq(pred).sum().item()
    count = pred.size(0)
    acc = correct / count

    return loss, acc


def train_edge_model(model, dataset, optimizer, epochs):
    dataset_name = dataset.__class__.__name__
    model_name = repr(model)
    optimizer_name = optimizer.__class__.__name__

    print(f"Training {dataset_name} model...")

    train_data, val_data, test_data = dataset[0]
    best_model = None
    for epoch in range(epochs):
        train_loss, train_acc = run_edge_gnn(model, train_data, optimizer)
        val_loss, val_acc = run_edge_gnn(model, val_data)

        print(
            f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if best_model is None or val_loss < best_model[0]:
            best_model = (val_loss, val_acc, model.state_dict())

    print()

    model.load_state_dict(best_model[2])
    loss, acc = run_edge_gnn(model, test_data)
    print(f"Test Loss: {loss:.4f}, Test Acc: {acc:.4f}")
    print()

    # save model
    torch.save(
        model, f"models/{dataset_name}_{model_name}_{optimizer_name}_{epochs}.pt"
    )
