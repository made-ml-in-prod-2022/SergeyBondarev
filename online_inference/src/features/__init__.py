"""
This package is aimed to be used as a library for feature engineering.
At the moment it does:
- One hot encoding on categorical features
- Scaling of numerical features
- Outlier removal
- Splitting to train and test sets
"""

from .build_features import build_features
from .build_features import build_features_done
