"""Prepare table columns using the same steps for every model.

fit() learns from training rows. transform() reuses those learned values on
validation or evaluation rows. A Pipeline keeps these two steps in order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data import normalize, duplicate_columns


class Cleaner(TransformerMixin, BaseEstimator):
    """Remember useful training columns, then select those same columns later."""

    def __init__(self, types):
        self.types = types

    def fit(self, X, y=None):
        normal = normalize(X, self.types)
        duplicate = duplicate_columns(normal)
        self.columns_ = []
        for name in normal:
            if normal[name].nunique(dropna=False) > 1 and name not in duplicate:
                self.columns_.append(name)
        return self

    def transform(self, X):
        normal = normalize(X, self.types)
        if not self.columns_:
            return pd.DataFrame({"__bias__": np.zeros(len(X))}, index=X.index)
        return normal[self.columns_]


def general_preprocessor(types, sparse=True):
    """Fill missing values, standardize numbers, and one-hot encode categories."""
    numeric = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
    )
    categorical = make_pipeline(
        SimpleImputer(strategy="most_frequent", keep_empty_features=True),
        # Pool rare categories to keep wide tables manageable. Unknown values
        # use the pooled group when one exists, otherwise an all-zero encoding.
        OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=2,
                      max_categories=64, sparse_output=sparse),
    )
    # Use scikit-learn's usual density threshold: mostly numeric tables stay
    # dense, while wide one-hot tables can use sparse storage to save memory.
    columns = ColumnTransformer([
        ("numeric", numeric, make_column_selector(dtype_include=np.number)),
        ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
    ], sparse_threshold=0.3 if sparse else 0.0)
    return make_pipeline(Cleaner(types), columns)
