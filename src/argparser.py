import torch

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


def add_dataset_args(argparser):
    argparser.add_argument(
        "--dataset",
        type=str,
        default="german",
        choices=["bail", "credit", "german", "pokec_n", "pokec_z"],
    )
    argparser.add_argument("--aware", action="store_true")
    argparser.add_argument("--modified", action="store_true")
    argparser.add_argument("--type", type=str, default="node", choices=["node", "edge"])


def add_misc_args(argparser):
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--no_cuda", action="store_true")
    argparser.add_argument("--debug", action="store_true")


def parse_dataset_args(args):
    dataset_name = (
        args.dataset
        + ("_aware" if args.aware else "")
        + ("_modified" if args.modified else "")
        + ("_edge" if args.type == "edge" else "")
    )

    dataset = None
    if args.dataset == "bail":
        dataset = (
            bail_modified
            if args.modified
            else (
                bail_aware
                if args.aware
                else (bail_link_pred if args.type == "edge" else bail)
            )
        )
    elif args.dataset == "credit":
        dataset = (
            credit_modified
            if args.modified
            else (
                credit_aware
                if args.aware
                else (credit_link_pred if args.type == "edge" else credit)
            )
        )
    elif args.dataset == "german":
        dataset = (
            german_modified
            if args.modified
            else (
                german_aware
                if args.aware
                else (german_link_pred if args.type == "edge" else german)
            )
        )
    elif args.dataset == "pokec_n":
        dataset = (
            pokec_n_modified
            if args.modified
            else (
                pokec_n_aware
                if args.aware
                else (pokec_n_link_pred if args.type == "edge" else pokec_n)
            )
        )
    elif args.dataset == "pokec_z":
        dataset = (
            pokec_z_modified
            if args.modified
            else (
                pokec_z_aware
                if args.aware
                else (pokec_z_link_pred if args.type == "edge" else pokec_z)
            )
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    return dataset, dataset_name


def parse_misc_args(args):
    set_random_seed(args.seed)
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    debug = args.debug

    return device, debug
