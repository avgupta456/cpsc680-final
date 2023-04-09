import numpy as np
from scipy.stats import wasserstein_distance


def get_attribute_bias(x, idx):
    norm_x = (x / x.norm(dim=0)).detach().cpu().numpy()

    print(norm_x[0])

    emd_distances = []
    for i in range(x.shape[1]):
        emd_distances.append(wasserstein_distance(norm_x[idx, i], norm_x[~idx, i]))

    emd_distances = [0 if np.isnan(x) else x for x in emd_distances]

    print(emd_distances)

    avg_distance = np.mean(emd_distances)

    return 1e3 * avg_distance
