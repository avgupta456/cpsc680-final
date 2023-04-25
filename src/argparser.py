from src.utils import set_random_seed
from src.datasets import *  # noqa: F401, F403


def add_dataset_args(argparser):
    argparser.add_argument(
        "--dataset",
        type=str,
        default="german",
    )


def add_misc_args(argparser):
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--no_cuda", action="store_true")
    argparser.add_argument("--debug", action="store_true")


def parse_dataset_args(args):
    dataset_name = args.dataset
    dataset = eval(dataset_name)
    return dataset, dataset_name


def parse_misc_args(args):
    set_random_seed(args.seed)
    debug = args.debug

    return debug
