from src.datasets.bail import bail, bail_aware, bail_modified, bail_link_pred
from src.datasets.credit import credit, credit_aware, credit_modified, credit_link_pred
from src.datasets.german import german, german_aware, german_modified, german_link_pred
from src.datasets.pokec import (
    pokec_z,
    pokec_z_aware,
    pokec_z_modified,
    pokec_z_link_pred,
    pokec_n,
    pokec_n_aware,
    pokec_n_modified,
    pokec_n_link_pred,
)

__all__ = [
    "bail",
    "bail_aware",
    "bail_modified",
    "bail_link_pred",
    "credit",
    "credit_aware",
    "credit_modified",
    "credit_link_pred",
    "german",
    "german_aware",
    "german_modified",
    "german_link_pred",
    "pokec_z",
    "pokec_z_aware",
    "pokec_z_modified",
    "pokec_z_link_pred",
    "pokec_n",
    "pokec_n_aware",
    "pokec_n_modified",
    "pokec_n_link_pred",
]
