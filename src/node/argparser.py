import argparse

from src.argparser import (
    add_dataset_args,
    add_misc_args,
    parse_dataset_args,
    parse_misc_args,
)


def add_node_args(argparser):
    argparser.add_argument(
        "--estimate_sens_attrs",
        action="store_true",
    )
    argparser.add_argument("--encoder_lr", type=float, default=1e-3)
    argparser.add_argument("--encoder_l1_rate", type=float, default=1e-3)
    argparser.add_argument("--classifier_lr", type=float, default=3e-3)
    argparser.add_argument("--classifier_weight_decay", type=float, default=3e-3)
    argparser.add_argument("--epochs", type=int, default=1000)


def parse_node_args(args):
    estimate_sens_attrs = args.estimate_sens_attrs
    encoder_lr = args.encoder_lr
    encoder_l1_rate = args.encoder_l1_rate
    classifier_lr = args.classifier_lr
    classifier_weight_decay = args.classifier_weight_decay
    epochs = args.epochs

    return (
        estimate_sens_attrs,
        encoder_lr,
        encoder_l1_rate,
        classifier_lr,
        classifier_weight_decay,
        epochs,
    )


def get_args():
    argparser = argparse.ArgumentParser()

    add_misc_args(argparser)
    add_dataset_args(argparser)
    add_node_args(argparser)

    return argparser.parse_args()


def parse_args(args):
    debug = parse_misc_args(args)
    dataset, dataset_name = parse_dataset_args(args)
    (
        estimate_sens_attrs,
        encoder_lr,
        encoder_l1_rate,
        classifier_lr,
        classifier_weight_decay,
        epochs,
    ) = parse_node_args(args)

    return (
        debug,
        dataset,
        dataset_name,
        estimate_sens_attrs,
        encoder_lr,
        encoder_l1_rate,
        classifier_lr,
        classifier_weight_decay,
        epochs,
    )
