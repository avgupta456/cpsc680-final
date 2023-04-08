import argparse

import torch
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv

from src.utils import set_random_seed


from src.datasets import (
    bail,
    bail_aware,
    bail_modified,
    bail_link_pred,
    credit,
    credit_aware,
    credit_modified,
    credit_link_pred,
    german,
    german_aware,
    german_modified,
    german_link_pred,
    pokec_z,
    pokec_z_aware,
    pokec_z_modified,
    pokec_z_link_pred,
    pokec_n,
    pokec_n_aware,
    pokec_n_modified,
    pokec_n_link_pred,
)
from src.node_gnn import VanillaNode, train_node_model
from src.edge_gnn import VanillaEdge, train_edge_model
from src.utils import device


def get_args():
    argparser = argparse.ArgumentParser()

    # Dataset
    argparser.add_argument(
        "--dataset",
        type=str,
        default="german",
        choices=["bail", "credit", "german", "pokec_n", "pokec_z"],
    )
    argparser.add_argument(
        "--aware",
        action="store_true",
    )
    argparser.add_argument(
        "--modified",
        action="store_true",
    )

    # Model
    argparser.add_argument("--type", type=str, default="node", choices=["node", "edge"])
    argparser.add_argument(
        "--model",
        type=str,
        default="GCNConv",
        choices=["GCNConv", "GATConv", "SAGEConv", "GINConv"],
    )
    argparser.add_argument("--hidden", type=int, nargs="+", default=[16])
    argparser.add_argument("--dropout", type=float, default=0.0)

    # Training
    argparser.add_argument("--epochs", type=int, default=50)
    argparser.add_argument("--lr", type=float, default=5e-3)
    argparser.add_argument("--weight_decay", type=float, default=1e-3)

    # Fairness Metrics
    argparser.add_argument("--model_path", type=str, default=None)

    # Misc
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--debug", action="store_true")

    return argparser.parse_args()


def parse_vanilla_args(args):
    set_random_seed(args.seed)

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

    dataset = None
    model = None
    train_model = None
    if args.type == "node":
        if args.dataset == "bail":
            dataset = (
                bail_modified if args.modified else (bail_aware if args.aware else bail)
            )
        elif args.dataset == "credit":
            dataset = credit_aware if args.aware else credit
        elif args.dataset == "german":
            dataset = (
                german_modified
                if args.modified
                else (german_aware if args.aware else german)
            )
        elif args.dataset == "pokec_n":
            dataset = (
                pokec_n_modified
                if args.modified
                else (pokec_n_aware if args.aware else pokec_n)
            )
        elif args.dataset == "pokec_z":
            dataset = (
                pokec_z_modified
                if args.modified
                else (pokec_z_aware if args.aware else pokec_z)
            )
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        model = VanillaNode(
            in_channels=dataset.num_features,
            hidden_channels=args.hidden,
            out_channels=1,
            block=block,
            dropout=args.dropout,
        ).to(device)
        train_model = train_node_model

    elif args.type == "edge":
        if args.dataset == "bail":
            dataset = bail_link_pred
        elif args.dataset == "credit":
            dataset = credit_link_pred
        elif args.dataset == "german":
            dataset = german_link_pred
        elif args.dataset == "pokec_n":
            dataset = pokec_n_link_pred
        elif args.dataset == "pokec_z":
            dataset = pokec_z_link_pred
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        model = VanillaEdge(
            in_channels=dataset.num_features,
            hidden_channels=args.hidden,
            block=block,
            dropout=args.dropout,
        ).to(device)
        train_model = train_edge_model

    else:
        raise ValueError(f"Unknown type: {args.type}")

    lr, weight_decay, epochs = args.lr, args.weight_decay, args.epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    debug = args.debug

    return model, train_model, dataset, optimizer, epochs, debug


def parse_metric_args(args):
    set_random_seed(args.seed)

    model_path = args.model_path
    model = torch.load(f"models/{model_path}").to(device)

    dataset_name = model_path.split("Dataset")[0]
    dataset = None
    if dataset_name == "Bail":
        dataset = bail
    elif dataset_name == "BailAware":
        dataset = bail_aware
    elif dataset_name == "BailModified":
        dataset = bail_modified
    elif dataset_name == "Credit":
        dataset = credit
    elif dataset_name == "CreditAware":
        dataset = credit_aware
    elif dataset_name == "CreditModified":
        dataset = credit_modified
    elif dataset_name == "German":
        dataset = german
    elif dataset_name == "GermanAware":
        dataset = german_aware
    elif dataset_name == "GermanModified":
        dataset = german_modified
    elif dataset_name == "PokecN":
        dataset = pokec_n
    elif dataset_name == "PokecNAware":
        dataset = pokec_n_aware
    elif dataset_name == "PokecNModified":
        dataset = pokec_n_modified
    elif dataset_name == "PokecZ":
        dataset = pokec_z
    elif dataset_name == "PokecZAware":
        dataset = pokec_z_aware
    elif dataset_name == "PokecZModified":
        dataset = pokec_z_modified
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset, model
