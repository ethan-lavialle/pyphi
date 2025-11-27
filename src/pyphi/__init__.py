"""
PyPhi - Multivariate Analysis Toolbox

A minimal Python library for JRPLS and TPLS analysis.

Version: 0.1.0
"""

from .utils import (
    # Statistical functions
    mean,
    std,
    meancenterscale,
    f95,
    f99,
    spe_ci,
    # NaN utilities
    n2z,
    # Array utilities
    unique,
    # DataFrame reconciliation
    isin_ordered_col0,
    reconcile_rows,
    reconcile_rows_to_columns,
    # JRPLS/TPLS data parsing
    parse_materials,
)

# PCA functions (minimal)
from .pca import hott2

# PLS functions (minimal)
from .pls import (
    pls,
    pls_,
)

# JRPLS and TPLS functions
from .advanced_pls import (
    jrpls,
    jrpls_pred,
    tpls,
    tpls_pred,
)

__version__ = "0.1.0"
__author__ = "Ethan Lavialle <ethan.lavialle@polymodelshub.com>"

__all__ = [
    # Statistical functions
    "mean",
    "std",
    "meancenterscale",
    "f95",
    "f99",
    "spe_ci",
    # NaN utilities
    "n2z",
    # Array utilities
    "unique",
    # DataFrame reconciliation
    "isin_ordered_col0",
    "reconcile_rows",
    "reconcile_rows_to_columns",
    # JRPLS/TPLS data parsing
    "parse_materials",
    # PCA (minimal)
    "hott2",
    # PLS (minimal)
    "pls",
    "pls_",
    # JRPLS and TPLS
    "jrpls",
    "jrpls_pred",
    "tpls",
    "tpls_pred",
]
