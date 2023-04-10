import tqdm

import torch
import torch.nn.functional as F
import torchmetrics

from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv


class VanillaNode(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        block=GCNConv,
        dropout=0.0,
    ):
        super().__init__()

        self.block_name = {
            GCNConv: "GCNConv",
            GATConv: "GATConv",
            SAGEConv: "SAGEConv",
            GINConv: "GINConv",
        }[block]

        self.convs = torch.nn.ModuleList()
        self.convs.append(block(in_channels, hidden_channels[0]))
        for i in range(1, len(hidden_channels)):
            if block == GATConv:
                self.convs.append(block(hidden_channels[i - 1], hidden_channels[i], 1))
            else:
                self.convs.append(block(hidden_channels[i - 1], hidden_channels[i]))
        self.convs.append(block(hidden_channels[-1], out_channels))

        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        x = x.sigmoid()

        return x.squeeze()

    def embedding(self, x, edge_index):
        # NOTE: No ReLU after last layer
        for conv in self.convs[:-2]:
            x = conv(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.convs[-2](x, edge_index)


def run_node_gnn(model, data, mask, optimizer=None):
    if optimizer:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    out = model(data.x, data.edge_index)[mask]
    pred = data.y[mask]

    loss = F.binary_cross_entropy(out, pred)

    if optimizer:
        loss.backward()
        optimizer.step()

    correct = out.round().eq(pred).sum().item()
    count = mask.sum().item()
    acc = correct / count
    auc = torchmetrics.functional.auroc(out, pred.to(int), task="binary")
    f1 = torchmetrics.functional.f1_score(out, pred.to(int), task="binary")

    return loss, acc, auc, f1


def train_node_model(model, dataset_name, dataset, optimizer, epochs, debug):
    print(f"Training {dataset_name} model...")

    data = dataset[0]
    best_model = None

    iterator = range(epochs) if debug else tqdm.tqdm(range(epochs))
    for epoch in iterator:
        train_loss, train_acc, _, _ = run_node_gnn(
            model, data, data.train_mask, optimizer
        )
        val_loss, val_acc, val_auc, val_f1 = run_node_gnn(model, data, data.val_mask)

        if debug:
            print(
                f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Auc: {val_auc:.4f}, Val F1: {val_f1:.4f}"
            )

        if best_model is None or val_loss < best_model[0]:
            best_model = (val_loss, val_acc, model.state_dict())

    print()

    model.load_state_dict(best_model[2])
    loss, acc, auc, f1 = run_node_gnn(model, data, data.test_mask)
    print(
        f"Test Loss: {loss:.4f}, Test Acc: {acc:.4f}, Test Auc: {auc:.4f}, Test F1: {f1:.4f}"
    )
    print()

    # save model
    torch.save(model, f"models/{dataset_name}.pt")
