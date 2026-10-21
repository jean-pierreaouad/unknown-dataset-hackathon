"""Preprocessors fit exclusively on each training partition."""
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
    def __init__(self, types):
        self.types = types

    def fit(self, X, y=None):
        normal = normalize(X, self.types)
        duplicate = duplicate_columns(normal)
        self.columns_ = [name for name in normal if normal[name].nunique(dropna=False) > 1 and name not in duplicate]
        return self

    def transform(self, X):
        normal = normalize(X, self.types)
        if not self.columns_:
            return pd.DataFrame({"__bias__": np.zeros(len(X))}, index=X.index)
        return normal[self.columns_]


class TreeEncoder(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        self.categories_ = {}
        self.categorical_mask_ = []
        for name in self.columns_:
            categorical = not pd.api.types.is_numeric_dtype(X[name])
            self.categorical_mask_.append(categorical)
            if categorical:
                counts = X[name].dropna().value_counts()
                self.categories_[name] = {value: index for index, value in enumerate(counts.index[:253])}
                self.categories_[name + "\0known"] = set(counts.index)
        return self

    def transform(self, X):
        output = pd.DataFrame(index=X.index)
        for name, categorical in zip(self.columns_, self.categorical_mask_, strict=True):
            if categorical:
                mapping = self.categories_[name]
                known = self.categories_[name + "\0known"]
                values = X[name].map(lambda value: mapping.get(value, 253) if pd.notna(value) and value in known else np.nan)
                output[name] = pd.Categorical(values, categories=list(range(254)))
            else:
                output[name] = X[name].to_numpy(dtype=float)
        return output


def general_preprocessor(types, sparse=True):
    numeric = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), StandardScaler())
    categorical = make_pipeline(SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=2, max_categories=64, sparse_output=sparse))
    columns = ColumnTransformer([
        ("numeric", numeric, make_column_selector(dtype_include=np.number)),
        ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
    ], sparse_threshold=1.0 if sparse else 0.0)
    return make_pipeline(Cleaner(types), columns)
