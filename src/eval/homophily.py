from torch_geometric.utils import homophily


def get_edge_homophily(edge_index, sens_attrs):
    return homophily(edge_index, sens_attrs, method="edge")


def get_node_homophily(edge_index, sens_attrs):
    return homophily(edge_index, sens_attrs, method="node")
