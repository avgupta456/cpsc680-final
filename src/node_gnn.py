import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv


class VanillaNode(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        block=GCNConv,
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

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = x.relu()

        x = self.convs[-1](x, edge_index)
        x = x.sigmoid()

        return x.squeeze()

    def embedding(self, x, edge_index):
        # NOTE: No ReLU after last layer
        for conv in self.convs[:-2]:
            x = conv(x, edge_index)
            x = x.relu()

        return self.convs[-2](x, edge_index)

    def __repr__(self):
        # Ex. Node_GCNConv(16,32,32,1)
        return f"Node_{self.block_name}({','.join([str(conv.out_channels) for conv in self.convs])})"


def run_node_gnn(model, data, mask, optimizer=None):
    if optimizer:
        model.train()
        optimizer.zero_grad()

    out = model(data.x, data.edge_index)[mask]
    pred = data.y[mask]

    loss = F.binary_cross_entropy(out, pred)

    if optimizer:
        loss.backward()
        optimizer.step()

    correct = out.round().eq(pred).sum().item()
    count = mask.sum().item()
    acc = correct / count

    return loss, acc


def train_node_model(model, dataset, optimizer, epochs):
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
