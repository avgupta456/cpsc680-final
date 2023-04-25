import numpy as np
from scipy.stats import wasserstein_distance
import scipy.sparse as sp

"""
Adapted from EDITS paper
"""


def get_attribute_bias(x, idx):
    norm_x = (x / x.norm(dim=0)).detach().cpu().numpy()

    emd_distances = []
    for i in range(x.shape[1]):
        emd_distances.append(wasserstein_distance(norm_x[idx, i], norm_x[~idx, i]))

    emd_distances = [0 if np.isnan(x) else x for x in emd_distances]
    avg_distance = np.mean(emd_distances)
    return 1e3 * avg_distance


def normalize_scipy(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def get_structural_bias(x, edge_index, idx, alpha=0.5, beta=0.9, num_hops=2):
    norm_x = (x / x.norm(dim=0)).detach().cpu().numpy()

    # use sparse matrix for adj
    adj = sp.coo_matrix(
        (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
        shape=(x.shape[0], x.shape[0]),
    )

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_scipy(adj)

    identity = sp.eye(adj.shape[0])
    adj_norm = alpha * adj + (1 - alpha) * identity

    cumulation = np.zeros_like(norm_x)
    for i in range(num_hops):
        cumulation += pow(beta, i) * adj_norm.dot(norm_x)

    emd_distances = []
    for i in range(x.shape[1]):
        emd_distances.append(
            wasserstein_distance(cumulation[idx, i], cumulation[~idx, i])
        )

    emd_distances = [0 if np.isnan(x) else x for x in emd_distances]
    avg_distance = np.mean(emd_distances)
    return 1e3 * avg_distance
