import tqdm

import torch
import torch.nn.functional as F
import torchmetrics

from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.utils import negative_sampling

from src.vanilla.node_gnn import VanillaNode


class VanillaEdge(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, block=GCNConv, dropout=0.0):
        super().__init__()

        self.block_name = {
            GCNConv: "GCNConv",
            GATConv: "GATConv",
            SAGEConv: "SAGEConv",
            GINConv: "GINConv",
        }[block]

        # NOTE: Out channels doesn't actually matter since we only use the embedding
        self.encoder = VanillaNode(in_channels, 1, hidden_channels, block, dropout)

    def encode(self, x, edge_index):
        return self.encoder.embedding(x, edge_index)

    def decode(self, z, edge_label_index):
        # NOTE: In logit space
        x1 = z[edge_label_index[0]]
        x2 = z[edge_label_index[1]]
        return (x1 * x2).sum(dim=1)

    def forward(self, x, edge_index, edge_label_index):
        # NOTE: In logit space
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

        loss = F.binary_cross_entropy_with_logits(out.sigmoid(), pred)

        # Add loss term to encourage mean of logits to be 0, or mean of probs to be 0.5
        # loss += 0.5 * (out.sigmoid().mean() - 0.5) ** 2  # tailed distrib with mode at 1
        loss += 0.02 * out.mean() ** 2  # normal distribution centered around 0.75

        loss.backward()
        optimizer.step()
    else:
        model.eval()
        out = model(data.x, data.edge_index, data.edge_label_index)
        pred = data.edge_label

        loss = F.binary_cross_entropy_with_logits(out.sigmoid(), pred)

    correct = out.sigmoid().round().eq(pred).sum().item()
    count = pred.size(0)
    acc = correct / count
    auc = torchmetrics.functional.auroc(out.sigmoid(), pred.to(int), task="binary")
    f1 = torchmetrics.functional.f1_score(out.sigmoid(), pred.to(int), task="binary")

    return loss, acc, auc, f1


def train_edge_model(model, dataset_name, dataset, optimizer, epochs, debug):
    print(f"Training {dataset_name} model...")

    train_data, val_data, test_data = dataset[0]
    best_model = None

    iterator = range(epochs) if debug else tqdm.tqdm(range(epochs))
    for epoch in iterator:
        train_loss, train_acc, _, _ = run_edge_gnn(model, train_data, optimizer)
        val_loss, val_acc, val_auc, val_f1 = run_edge_gnn(model, val_data)

        if debug:
            print(
                f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}",
            )

        if best_model is None or val_loss < best_model[0]:
            best_model = (val_loss, val_acc, model.state_dict())

    print()

    model.load_state_dict(best_model[2])
    loss, acc, auc, f1 = run_edge_gnn(model, test_data)
    print(
        f"Test Loss: {loss:.4f}, Test Acc: {acc:.4f}, Test AUC: {auc:.4f}, Test F1: {f1:.4f}"
    )
    print()

    # save model
    torch.save(model, f"models/{dataset_name}.pt")
