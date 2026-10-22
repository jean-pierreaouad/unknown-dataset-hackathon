"""Input validation, deterministic type handling, and raw-data profiling."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd


def read_csv(path, text_columns=()):
    """Read a nonempty table, preserving declared categories and IDs as text."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        headers = next(csv.reader(stream), None)
    if not headers or any(not name.strip() for name in headers):
        raise ValueError("CSV must have a nonempty header for every column")
    if len(headers) != len(set(headers)):
        raise ValueError("Duplicate CSV column names are not supported")
    # Respect categorical metadata before pandas infers numbers and discards
    # distinctions such as '001' versus '1'. Preserve explicit identifiers too.
    dtypes = {name: str for name in text_columns if name in headers}
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, dtype=dtypes)
    if len(frame) == 0:
        raise ValueError("CSV must contain at least one data row")
    return frame


def infer_types(frame, declared):
    """Use the supplied feature types; infer only the unspecified columns."""
    unknown = set(declared) - set(frame.columns)
    if unknown:
        raise ValueError(f"feature_types contains unknown features: {sorted(unknown)}")
    types = {}
    for name in frame:
        values = frame[name]
        if name in declared:
            types[name] = declared[name]
        elif pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            types[name] = "numeric"
        else:
            present = values.dropna().astype(str).str.strip()
            present = present[present.ne("")]
            converted = pd.to_numeric(present, errors="coerce")
            types[name] = "numeric" if len(present) and converted.notna().mean() >= 0.98 else "categorical"
    return types


def numeric(values):
    """Turn invalid numbers and infinities into missing values for imputation."""
    array = pd.to_numeric(values, errors="coerce").astype(float)
    return array.replace([np.inf, -np.inf], np.nan)


def stable_median(values):
    """Avoid overflow when averaging the two central finite values."""
    ordered = np.sort(np.asarray(values, dtype=float))
    if not len(ordered):
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    lower, upper = ordered[middle - 1:middle + 1]
    # Same-sign subtraction is bounded; opposite-sign addition is bounded.
    if np.signbit(lower) == np.signbit(upper):
        return float(lower + (upper - lower) / 2)
    return float((lower + upper) / 2)


def normalize(frame, types):
    """Give numeric and category columns consistent pandas representations."""
    output = pd.DataFrame(index=frame.index)
    for name, kind in types.items():
        values = frame[name]
        if kind == "numeric":
            # Bound extreme magnitudes to keep scaling and squared losses finite.
            output[name] = numeric(values).clip(-1e30, 1e30)
        else:
            output[name] = values.map(lambda value: np.nan if pd.isna(value) else str(value)).astype(object)
    return output


def duplicate_columns(frame):
    """Map each duplicate column to its first equal column in this table."""
    seen = {}
    duplicates = {}
    for name in frame:
        values = frame[name]
        digest = hashlib.sha256(pd.util.hash_pandas_object(values, index=False).values.tobytes()).hexdigest()
        # Hashes only shortlist comparisons; equality is verified to avoid hash collisions.
        bucket = seen.setdefault(digest, [])
        match = next((other for other in bucket if values.equals(frame[other])), None)
        if match is not None:
            duplicates[name] = match
        else:
            bucket.append(name)
    return duplicates


def profile_frame(frame, types):
    """Describe the original feature table before fitting any preparation."""
    columns = {}
    for name in frame:
        values = frame[name]
        missing = int(values.isna().sum())
        unique = int(values.nunique(dropna=True))
        item = {
            "type": types[name],
            "missing_count": missing,
            "missing_fraction": missing / len(values),
            "unique_nonmissing": unique,
            "constant": unique <= 1,
            "all_missing": missing == len(values),
        }
        if types[name] == "numeric":
            converted = pd.to_numeric(values, errors="coerce").astype(float)
            finite = converted[np.isfinite(converted)]
            item["infinite_count"] = int(np.isinf(converted).sum())
            item["invalid_numeric_count"] = int((values.notna() & converted.isna()).sum())
            item["numeric_clipped_count"] = int((finite.abs() > 1e30).sum())
            item["statistics"] = None
            if len(finite):
                # Calculate the mean on a bounded scale, then restore units.
                # This avoids summing huge values into infinity.
                scale = max(1.0, float(finite.abs().max()))
                item["statistics"] = {
                    "min": float(finite.min()),
                    "max": float(finite.max()),
                    "median": stable_median(finite),
                    "mean": float((finite / scale).mean() * scale),
                }
        else:
            counts = values.dropna().astype(str).value_counts().head(10)
            item["top_values"] = [{"value": str(value), "count": int(count)} for value, count in counts.items()]
        columns[name] = item
    return {"rows": len(frame), "features": len(frame.columns), "columns": columns,
            "constant_columns": [name for name, item in columns.items() if item["constant"]],
            "duplicate_columns": duplicate_columns(frame),
            "duplicate_row_count": int(frame.duplicated().sum()),
            "missing_cells": int(frame.isna().sum().sum())}


def load_data(config):
    """Return features, evaluation rows, targets, types, profile, and row IDs."""
    text_columns = [name for name, kind in config["feature_types"].items() if kind == "categorical"]
    if config.get("id_column"):
        text_columns.append(config["id_column"])
    raw = read_csv(config["train"], text_columns)
    test = read_csv(config["test"], text_columns) if config.get("test") else None
    target = config.get("target")
    y = None
    if target:
        if target not in raw:
            raise ValueError(f"Target column not found: {target}")
        y = raw.pop(target)
        if y.isna().any():
            raise ValueError("Training targets contain missing values")
        if test is not None and target in test:
            raise ValueError("Evaluation CSV must not contain the target column")
        if config["task"] == "regression":
            y = pd.to_numeric(y, errors="coerce").astype(float)
            if not np.isfinite(y).all():
                raise ValueError("Regression targets must be finite numbers")
            if config["metric"] == "rmsle" and (y < 0).any():
                raise ValueError("rmsle requires nonnegative targets")
    id_column = config.get("id_column")
    if id_column:
        if id_column not in raw or (test is not None and id_column not in test):
            raise ValueError("id_column must exist in every supplied CSV")
        if id_column == target:
            raise ValueError("id_column and target must differ")
        row_ids = (test if test is not None else raw)[id_column].tolist()
        if any(pd.isna(value) for value in row_ids) or len(set(row_ids)) != len(row_ids):
            raise ValueError("Output row identifiers must be nonmissing and unique")
        raw = raw.drop(columns=[id_column])
        if test is not None:
            test = test.drop(columns=[id_column])
    else:
        row_ids = list(range(len(test) if test is not None else len(raw)))
    if not len(raw.columns):
        raise ValueError("Dataset must contain at least one feature column")
    if test is not None:
        if set(test.columns) != set(raw.columns):
            raise ValueError(f"Train/test features differ: missing={sorted(set(raw)-set(test))}, extra={sorted(set(test)-set(raw))}")
        test = test.loc[:, raw.columns]
    if config["task"] == "clustering" and config["n_clusters"] > len(raw):
        raise ValueError("n_clusters must not exceed the number of training rows")
    types = infer_types(raw, config["feature_types"])
    profile = {"schema_version": "1.0", "task": config["task"], "feature_types": types,
               "training": profile_frame(raw, types)}
    if test is not None:
        profile["evaluation"] = profile_frame(test, types)
    if y is not None:
        if config["task"] == "classification":
            counts = y.value_counts()
            profile["target"] = {"name": target, "classes": [{"label": value.item() if isinstance(value, np.generic) else value, "count": int(count)} for value, count in counts.items()],
                                 "imbalance_ratio": float(counts.max() / counts.min())}
        else:
            profile["target"] = {"name": target, "min": float(y.min()), "max": float(y.max()), "median": stable_median(y)}
    return raw, test, y, types, profile, row_ids
