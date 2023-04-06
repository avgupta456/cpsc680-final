import random

import numpy as np
import torch
import torch_geometric


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch_geometric.seed_everything(seed)


def get_device():
    device = torch.device("cpu")

    """
    # Currently Pytorch Geometric does not support MPS

    # Check that MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.backends.mps.is_built():
        # print("MPS not available because the current MacOS version is not 12.3+")
        pass
    else:
        # print("MPS not available because of the current PyTorch install")
        pass
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")

    return device


device = get_device()
