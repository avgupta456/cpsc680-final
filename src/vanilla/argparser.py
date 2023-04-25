import argparse

import torch
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv

from src.argparser import (
    add_dataset_args,
    add_misc_args,
    parse_dataset_args,
    parse_misc_args,
)
from src.utils import device
from src.vanilla.edge_gnn import VanillaEdge, train_edge_model
from src.vanilla.node_gnn import VanillaNode, train_node_model


def add_model_args(argparser):
    # Model
    argparser.add_argument(
        "--model",
        type=str,
        default="GCNConv",
        choices=["GCNConv", "GATConv", "SAGEConv", "GINConv"],
    )
    argparser.add_argument("--type", type=str, default="node", choices=["node", "edge"])
    argparser.add_argument(
        "--target_name", type=str, default="label", choices=["label", "sens_attr"]
    )

    argparser.add_argument("--hidden", type=int, nargs="+", default=[16])
    argparser.add_argument("--dropout", type=float, default=0.0)

    # Training
    argparser.add_argument("--epochs", type=int, default=300)
    argparser.add_argument("--lr", type=float, default=3e-3)
    argparser.add_argument("--weight_decay", type=float, default=3e-3)


def parse_model_args(args, dataset):
    block = None
    if args.model == "GCNConv":
        block = GCNConv
    elif args.model == "GATConv":
        block = GATConv
    elif args.model == "SAGEConv":
        block = SAGEConv
    elif args.model == "GINConv":
        block = GINConv
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = None
    train_model = None

    if args.type == "edge":
        model = VanillaEdge(
            in_channels=dataset.num_features,
            hidden_channels=args.hidden,
            block=block,
            dropout=args.dropout,
        ).to(device)
        train_model = train_edge_model
    else:
        model = VanillaNode(
            in_channels=dataset.num_features,
            hidden_channels=args.hidden,
            out_channels=1,
            block=block,
            dropout=args.dropout,
        ).to(device)
        train_model = train_node_model

    target_name = args.target_name

    lr, weight_decay, epochs = args.lr, args.weight_decay, args.epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    return model, train_model, target_name, optimizer, epochs


def get_args():
    argparser = argparse.ArgumentParser()
    add_misc_args(argparser)
    add_dataset_args(argparser)
    add_model_args(argparser)

    return argparser.parse_args()


def parse_args(args):
    debug = parse_misc_args(args)
    dataset, dataset_name = parse_dataset_args(args)
    model, train_model, target_name, optimizer, epochs = parse_model_args(args, dataset)

    return (
        debug,
        dataset,
        dataset_name,
        model,
        train_model,
        target_name,
        optimizer,
        epochs,
    )
