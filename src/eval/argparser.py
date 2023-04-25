import argparse

from src.argparser import (
    add_dataset_args,
    add_misc_args,
    parse_dataset_args,
    parse_misc_args,
)


def get_args():
    argparser = argparse.ArgumentParser()

    add_misc_args(argparser)
    add_dataset_args(argparser)

    return argparser.parse_args()


def parse_args(args):
    debug = parse_misc_args(args)
    dataset, dataset_name = parse_dataset_args(args)

    return dataset, dataset_name, debug
