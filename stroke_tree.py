# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

"""Corrected reconstruction of a cost-sensitive stroke classification study.

Version 2 fixes preprocessing leakage and duplicate-value threshold handling.
Historical coursework scores have NOT been reproduced. Educational use only:
not a clinical model, diagnosis, screening tool, or future-risk estimate.
"""
# %%
# 1) Environment and imports
from __future__ import annotations

import argparse
import hashlib
import platform
import importlib.metadata
import json
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Reproducible display settings
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 140)


# %% [markdown]
# ## 2) Configuration
#
# These retain the historical cost assumptions, not the historical results. Risk bands are descriptive
# outputs only; they do not affect the binary decision or total-cost calculation.

# %%
# Business loss
COST_FN = 10.0
COST_FP = 1.0
BUSINESS_THR = COST_FP / (COST_FN + COST_FP)

# Descriptive risk bands. The old conversation also tested validation quantiles;
# because repeated leaf probabilities can make two quantiles identical, fixed
# cutoffs are the stable default here.
RISK_BANDS = (0.12, 0.20)

# Tree complexity / pre-pruning
MAX_DEPTH = 3
MIN_LEAF = 20
MIN_DELTA_COST = 0.0
MAX_CANDIDATES = None  # Exhaustive distinct-value midpoints by default.

# Corrected stratified raw-row split; different from the historical split.
VAL_RATIO = 0.20
TEST_RATIO = 0.20
SEED = 42

# Dataset columns
POSSIBLE_TARGET_NAMES = ("stroke", "Stroke", "label", "target")
DROP_COLS = ("id", "ID")
NUMERIC_COLS_DEFAULT = (
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease",
)
CATEGORICAL_COLS_DEFAULT = (
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
)


def print_configuration() -> None:
    print(
        f"Business threshold p* = {BUSINESS_THR:.4f} "
        f"(FN cost={COST_FN:g}, FP cost={COST_FP:g})"
    )
    print(
        f"Default MAX_DEPTH={MAX_DEPTH}, MIN_LEAF={MIN_LEAF}, "
        f"MIN_DELTA_COST={MIN_DELTA_COST:g}"
    )
    print(f"Risk bands: low < {RISK_BANDS[0]:.3f}, mid < {RISK_BANDS[1]:.3f}, high otherwise")


# %% [markdown]
# ## 3) Data loading and cleaning
#
# Fit statistics and category vocabularies on training rows only.
# Raw labels are validated separately and are never imputed.

def load_stroke_csv(path_or_url: str | Path) -> pd.DataFrame:
    """Read a user-supplied CSV. No dataset is downloaded automatically."""
    if not str(path_or_url).strip():
        raise ValueError("Please provide a CSV path.")
    df = pd.read_csv(path_or_url)
    df.columns = [str(c).strip() for c in df.columns]
    if not df.columns.is_unique:
        raise ValueError("Column names must be unique.")
    return df


def extract_target(df: pd.DataFrame) -> pd.Series:
    """Reject missing, fractional, and non-binary targets before conversion."""
    names = [c for c in POSSIBLE_TARGET_NAMES if c in df.columns]
    if len(names) != 1:
        raise ValueError("Expected exactly one target column.")
    y = pd.to_numeric(df[names[0]], errors="raise")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("Target must contain only observed binary 0/1 labels.")
    return y.astype(int).reset_index(drop=True).rename("stroke")


class StrokePreprocessor:
    """Train-only medians and categorical vocabulary with JSON-safe state.

    Unseen categories map to all-zero indicators for that feature. Missing
    categories use a named token if observed during fitting. Neither operation
    changes the fitted vocabulary. Numeric missing values use fitted medians.
    """

    def __init__(self) -> None:
        self.medians: dict[str, float] = {}
        self.categories: dict[str, list[str]] = {}
        self.feature_names: list[str] = []
        self.fitted = False

    @staticmethod
    def _check_frame(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame) or not df.columns.is_unique:
            raise ValueError("Input must be a DataFrame with unique column names.")
        required = NUMERIC_COLS_DEFAULT + CATEGORICAL_COLS_DEFAULT
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required raw features: {missing}")

    @staticmethod
    def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
        values = pd.to_numeric(df[column], errors="raise").astype(float)
        if np.isinf(values.to_numpy()).any():
            raise ValueError(f"Infinite feature values are not supported: {column}")
        return values

    @staticmethod
    def _categorical(df: pd.DataFrame, column: str) -> pd.Series:
        return df[column].astype("string").fillna("__MISSING__")

    def fit(self, df: pd.DataFrame) -> "StrokePreprocessor":
        self._check_frame(df)
        if df.empty:
            raise ValueError("Cannot fit preprocessing on empty data.")
        medians = {}
        categories = {}
        for column in NUMERIC_COLS_DEFAULT:
            values = self._numeric(df, column)
            if not values.notna().any():
                raise ValueError(f"No observed training values for {column}")
            medians[column] = float(values.median())
        for column in CATEGORICAL_COLS_DEFAULT:
            categories[column] = sorted(self._categorical(df, column).unique().tolist())
        self.medians = medians
        self.categories = categories
        self.feature_names = list(NUMERIC_COLS_DEFAULT) + [
            f"{column}={value}"
            for column in CATEGORICAL_COLS_DEFAULT
            for value in categories[column]
        ]
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Fit preprocessing on training data first.")
        self._check_frame(df)
        columns = {
            c: self._numeric(df, c).fillna(self.medians[c]).to_numpy()
            for c in NUMERIC_COLS_DEFAULT
        }
        for c in CATEGORICAL_COLS_DEFAULT:
            values = self._categorical(df, c)
            for category in self.categories[c]:
                columns[f"{c}={category}"] = (values == category).to_numpy(dtype=float)
        return pd.DataFrame(columns, columns=self.feature_names, dtype=float)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def to_dict(self) -> dict[str, Any]:
        if not self.fitted:
            raise ValueError("Cannot serialize an unfitted preprocessor.")
        return {"medians": self.medians, "categories": self.categories,
                "feature_names": self.feature_names}

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "StrokePreprocessor":
        obj = cls()
        obj.medians = {k: float(v) for k, v in state["medians"].items()}
        obj.categories = {k: list(v) for k, v in state["categories"].items()}
        obj.feature_names = list(state["feature_names"])
        expected = list(NUMERIC_COLS_DEFAULT) + [
            f"{c}={v}" for c in CATEGORICAL_COLS_DEFAULT for v in obj.categories[c]
        ]
        if (set(obj.medians) != set(NUMERIC_COLS_DEFAULT)
                or obj.feature_names != expected
                or not np.isfinite(list(obj.medians.values())).all()):
            raise ValueError("Invalid preprocessing state.")
        obj.fitted = True
        return obj


def resolve_csv_path(csv_path: str | Path | None = None) -> Path:
    """Find the dataset in a supplied path or common Colab/local locations."""
    candidates: list[Path] = []
    if csv_path is not None and str(csv_path).strip():
        candidates.append(Path(csv_path).expanduser())
    candidates.extend(
        [
            Path("healthcare-dataset-stroke-data.csv"),
            Path("/content/healthcare-dataset-stroke-data.csv"),
            Path("/content/drive/MyDrive/MachineLearning/healthcare-dataset-stroke-data.csv"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Stroke CSV not found. Pass --csv or set CSV_PATH in the notebook. Tried:\n" + tried
    )


# %% [markdown]
# ## 4) Fixed split, majority baseline, and cost evaluation

# %%
def split_row_positions(
    y: pd.Series,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratify raw row positions before any data-dependent preprocessing."""
    if not (0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1):
        raise ValueError("Invalid validation/test ratios.")
    labels = np.asarray(y)
    if pd.isna(labels).any() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Split labels must be observed binary 0/1.")
    if len(np.unique(labels)) != 2:
        raise ValueError("Both classes are required for this experiment.")
    positions = np.arange(len(y))
    development, test = train_test_split(
        positions, test_size=test_ratio, stratify=labels, random_state=seed
    )
    train, val = train_test_split(
        development, test_size=val_ratio / (1 - test_ratio),
        stratify=labels[development], random_state=seed
    )
    return train, val, test


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = SEED,
) -> tuple[tuple[pd.DataFrame, pd.Series], ...]:
    """Split raw data, without learning medians or category vocabularies."""
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")
    positions = split_row_positions(y, val_ratio, test_ratio, seed)
    return tuple((X.iloc[i].reset_index(drop=True), y.iloc[i].reset_index(drop=True))
                 for i in positions)


def majority_baseline_fit(y_train: pd.Series) -> int:
    """Return the most frequent class label."""
    return int(y_train.value_counts().idxmax())


def majority_baseline_predict_proba(n_rows: int, majority_label: int) -> np.ndarray:
    """Return a constant p(class=1) vector for the majority baseline."""
    return np.full(n_rows, 1.0 if majority_label == 1 else 0.0, dtype=float)


def compute_confusion_and_metrics(
    y_true: Iterable[int],
    p1: Iterable[float],
    thr: float = BUSINESS_THR,
) -> dict[str, float | int]:
    """Compute confusion counts and standard binary metrics."""
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(p1, dtype=float)
    if y_arr.shape[0] != p_arr.shape[0]:
        raise ValueError("y_true and p1 must have the same length.")
    y_pred = (p_arr >= float(thr)).astype(int)

    tp = int(((y_arr == 1) & (y_pred == 1)).sum())
    tn = int(((y_arr == 0) & (y_pred == 0)).sum())
    fp = int(((y_arr == 0) & (y_pred == 1)).sum())
    fn = int(((y_arr == 1) & (y_pred == 0)).sum())

    accuracy = (tp + tn) / max(len(y_arr), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "acc": float(accuracy),
        "prec": float(precision),
        "rec": float(recall),
        "fpr": float(fpr),
        "f1": float(f1),
    }


def evaluate_full(
    y_true: Iterable[int],
    p1: Iterable[float],
    thr: float = BUSINESS_THR,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
    verbose: bool = True,
) -> dict[str, float | int]:
    """Evaluate standard metrics plus Total Cost = cost_fn*FN + cost_fp*FP."""
    metrics = compute_confusion_and_metrics(y_true, p1, thr=thr)
    cost = cost_fn * int(metrics["fn"]) + cost_fp * int(metrics["fp"])
    result: dict[str, float | int] = {
        "cost": float(cost),
        **metrics,
        "thr": float(thr),
    }
    if verbose:
        print(
            f"[Total Cost] {cost:.1f} = FN({metrics['fn']})*{cost_fn:g} "
            f"+ FP({metrics['fp']})*{cost_fp:g}"
        )
        print(
            f"Acc={metrics['acc']:.3f}  Prec={metrics['prec']:.3f}  "
            f"Rec={metrics['rec']:.3f}  FPR={metrics['fpr']:.3f}  "
            f"F1={metrics['f1']:.3f}  @thr={thr:.3f}"
        )
        print("Confusion Matrix (TN FP / FN TP):")
        print(
            f"{int(metrics['tn']):4d} {int(metrics['fp']):4d}\n"
            f"{int(metrics['fn']):4d} {int(metrics['tp']):4d}"
        )
    return result


def evaluate_cost_simple(
    y_true: Iterable[int],
    p1: Iterable[float],
    thr: float = BUSINESS_THR,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
    verbose: bool = True,
) -> dict[str, float | int]:
    """Backward-compatible name used in the original notebook."""
    return evaluate_full(y_true, p1, thr, cost_fn, cost_fp, verbose)


def print_ein_eout_summary(
    name: str,
    train_eval: dict[str, float | int],
    val_eval: dict[str, float | int],
) -> None:
    gap = float(val_eval["cost"]) - float(train_eval["cost"])
    print(
        f"[{name}] Ein(cost)={float(train_eval['cost']):.1f} | "
        f"Eout(cost)={float(val_eval['cost']):.1f} | gap={gap:.1f}"
    )


# %% [markdown]
# ## 5) Core bricks: leaf probabilities, cost, and split search

# %%
def leaf_proba_from_y_with_laplace(y: Iterable[int]) -> tuple[float, float]:
    """Return (p0, p1) using Laplace/add-one smoothing."""
    y_arr = np.asarray(y, dtype=int)
    n = len(y_arr)
    if n == 0:
        return 0.5, 0.5
    pos = int((y_arr == 1).sum())
    neg = n - pos
    p1 = (pos + 1.0) / (pos + neg + 2.0)
    return float(1.0 - p1), float(p1)


def best_judgment_threshold(
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> float:
    """Return the Bayes threshold p* = C_FP / (C_FN + C_FP)."""
    if cost_fn < 0 or cost_fp < 0 or cost_fn + cost_fp <= 0:
        raise ValueError("Costs must be non-negative and not both zero.")
    return float(cost_fp / (cost_fn + cost_fp))


def leaf_confusion_given_threshold(
    y_true: Iterable[int],
    p1: float,
    thr: float | None = None,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> tuple[int, int, int, int]:
    """Return (TP, FP, TN, FN) when one leaf assigns the same p1 to all rows."""
    if thr is None:
        thr = best_judgment_threshold(cost_fn, cost_fp)
    y_arr = np.asarray(y_true, dtype=int)
    y_pred = int(float(p1) >= float(thr))
    tp = int(((y_arr == 1) & (y_pred == 1)).sum())
    tn = int(((y_arr == 0) & (y_pred == 0)).sum())
    fp = int(((y_arr == 0) & (y_pred == 1)).sum())
    fn = int(((y_arr == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn


def cost_of_leaf(
    y_true: Iterable[int],
    p1: float,
    thr: float | None = None,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> float:
    """Compute the business cost if a node is kept as one leaf."""
    _, fp, _, fn = leaf_confusion_given_threshold(y_true, p1, thr, cost_fn, cost_fp)
    return float(cost_fn * fn + cost_fp * fp)


def risk_band(p1: float, bands: Sequence[float] = RISK_BANDS) -> str:
    """Map a probability to low/mid/high descriptive risk."""
    if len(bands) != 2 or not (0 <= bands[0] < bands[1] <= 1):
        raise ValueError("bands must be two increasing values in [0, 1].")
    if p1 < bands[0]:
        return "low"
    if p1 < bands[1]:
        return "mid"
    return "high"


def fit_quantile_risk_bands(
    scores: Iterable[float],
    q_low: float = 0.50,
    q_high: float = 0.80,
    min_gap: float = 0.01,
) -> tuple[float, float]:
    """Optional quantile bands with an enforced minimum separation."""
    values = np.asarray(scores, dtype=float)
    if len(values) == 0:
        raise ValueError("scores must not be empty.")
    low = float(np.quantile(values, q_low))
    high = float(np.quantile(values, q_high))
    low = float(np.clip(low, 0.0, 1.0))
    high = float(np.clip(max(high, low + min_gap), 0.0, 1.0))
    if high <= low:
        low = max(0.0, min(low, 1.0 - min_gap))
        high = min(1.0, low + min_gap)
    return low, high


def parent_leaf_cost(
    y_parent: Iterable[int],
    thr: float = BUSINESS_THR,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> float:
    """Compute the cost of leaving a parent node unsplit."""
    _, p1 = leaf_proba_from_y_with_laplace(y_parent)
    return cost_of_leaf(y_parent, p1, thr, cost_fn, cost_fp)


def candidate_thresholds_numeric(
    x: pd.Series,
    y: pd.Series,
    max_candidates: int | None = MAX_CANDIDATES,
) -> list[float]:
    """Enumerate adjacent distinct-value midpoints, independent of row/label order.

    An explicit cap is optional approximate search. The default is exhaustive.
    Labels are retained in the signature for source compatibility, not used.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have equal lengths.")
    values = np.asarray(x, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Threshold inputs must be finite after preprocessing.")
    if max_candidates is not None and (not isinstance(max_candidates, int) or max_candidates < 1):
        raise ValueError("max_candidates must be a positive integer or None.")
    unique = np.unique(values)
    candidates = unique[:-1] / 2.0 + unique[1:] / 2.0
    candidates = np.where(candidates >= unique[1:], unique[:-1], candidates)
    if max_candidates is not None and len(candidates) > max_candidates:
        positions = np.linspace(0, len(candidates) - 1, max_candidates).astype(int)
        candidates = candidates[positions]
    return candidates.tolist()


def best_threshold_for_feature_by_cost(
    x_col: pd.Series,
    y: pd.Series,
    min_leaf: int = MIN_LEAF,
    thr: float = BUSINESS_THR,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
    max_candidates: int | None = MAX_CANDIDATES,
) -> dict[str, Any] | None:
    """Find the threshold with the largest parent-to-children total-cost drop."""
    if len(x_col) != len(y):
        raise ValueError("x_col and y must have the same length.")
    if min_leaf < 1:
        raise ValueError("min_leaf must be at least 1.")

    base_cost = parent_leaf_cost(y, thr, cost_fn, cost_fp)
    unique_values = np.unique(np.asarray(x_col, dtype=float))
    is_binary = len(unique_values) <= 2 and np.all(np.isin(unique_values, [0.0, 1.0]))
    candidates = [0.5] if is_binary else candidate_thresholds_numeric(x_col, y, max_candidates)

    best: dict[str, Any] | None = None
    x_values = np.asarray(x_col, dtype=float)
    for threshold in candidates:
        left_mask = x_values <= threshold
        right_mask = ~left_mask
        if int(left_mask.sum()) < min_leaf or int(right_mask.sum()) < min_leaf:
            continue

        _, p1_left = leaf_proba_from_y_with_laplace(y[left_mask])
        _, p1_right = leaf_proba_from_y_with_laplace(y[right_mask])
        left_cost = cost_of_leaf(y[left_mask], p1_left, thr, cost_fn, cost_fp)
        right_cost = cost_of_leaf(y[right_mask], p1_right, thr, cost_fn, cost_fp)
        delta_cost = base_cost - (left_cost + right_cost)

        if best is None or delta_cost > float(best["delta_cost"]):
            best = {
                "feature_threshold": float(threshold),
                "delta_cost": float(delta_cost),
                "left_mask": left_mask,
                "right_mask": right_mask,
                "left_cost": float(left_cost),
                "right_cost": float(right_cost),
                "parent_cost": float(base_cost),
            }
    return best


def best_split_over_features_by_cost(
    X: pd.DataFrame,
    y: pd.Series,
    min_leaf: int = MIN_LEAF,
    thr: float = BUSINESS_THR,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
    max_candidates: int | None = MAX_CANDIDATES,
) -> dict[str, Any] | None:
    """Search all feature-threshold pairs and return the largest cost drop."""
    best_global: dict[str, Any] | None = None
    for column in X.columns:
        result = best_threshold_for_feature_by_cost(
            X[column],
            y,
            min_leaf=min_leaf,
            thr=thr,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
            max_candidates=max_candidates,
        )
        if result is None:
            continue
        if best_global is None or float(result["delta_cost"]) > float(best_global["delta_cost"]):
            best_global = {"feature": str(column), **result}
    return best_global


# %% [markdown]
# ## 6) Recursive tree, prediction, and readable rules

# %%
def make_leaf(y: pd.Series) -> dict[str, Any]:
    p0, p1 = leaf_proba_from_y_with_laplace(y)
    return {
        "is_leaf": True,
        "proba": [float(p0), float(p1)],
        "n": int(len(y)),
        "pos": int((np.asarray(y) == 1).sum()),
        "neg": int((np.asarray(y) == 0).sum()),
    }


def build_tree_by_cost(
    X: pd.DataFrame,
    y: pd.Series,
    depth: int = 0,
    max_depth: int = MAX_DEPTH,
    min_leaf: int = MIN_LEAF,
    min_delta_cost: float = MIN_DELTA_COST,
    thr: float = BUSINESS_THR,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
    max_candidates: int | None = MAX_CANDIDATES,
) -> dict[str, Any]:
    """Build a greedy, pre-pruned decision tree using total-cost reduction."""
    n = len(y)
    pos = int((np.asarray(y) == 1).sum())
    neg = n - pos

    if depth >= max_depth:
        return make_leaf(y)
    if n < 2 * min_leaf:
        return make_leaf(y)
    if pos == 0 or neg == 0:
        return make_leaf(y)

    best = best_split_over_features_by_cost(
        X,
        y,
        min_leaf=min_leaf,
        thr=thr,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        max_candidates=max_candidates,
    )
    if best is None or float(best["delta_cost"]) <= min_delta_cost:
        return make_leaf(y)

    feature = str(best["feature"])
    threshold = float(best["feature_threshold"])
    left_mask = np.asarray(best["left_mask"], dtype=bool)
    right_mask = np.asarray(best["right_mask"], dtype=bool)

    left_node = build_tree_by_cost(
        X[left_mask],
        y[left_mask],
        depth=depth + 1,
        max_depth=max_depth,
        min_leaf=min_leaf,
        min_delta_cost=min_delta_cost,
        thr=thr,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        max_candidates=max_candidates,
    )
    right_node = build_tree_by_cost(
        X[right_mask],
        y[right_mask],
        depth=depth + 1,
        max_depth=max_depth,
        min_leaf=min_leaf,
        min_delta_cost=min_delta_cost,
        thr=thr,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        max_candidates=max_candidates,
    )

    return {
        "is_leaf": False,
        "feature": feature,
        "thr": threshold,
        "left": left_node,
        "right": right_node,
        "depth": int(depth),
        "delta_cost": float(best["delta_cost"]),
        "parent_cost": float(best["parent_cost"]),
        "left_cost": float(best["left_cost"]),
        "right_cost": float(best["right_cost"]),
        "n": int(n),
        "pos": int(pos),
        "neg": int(neg),
    }


def predict_one(tree: dict[str, Any], x_row: pd.Series | dict[str, Any]) -> float:
    """Traverse one encoded row to a leaf and return p(class=1)."""
    row = pd.Series(x_row) if isinstance(x_row, dict) else x_row
    node = tree
    while not bool(node.get("is_leaf", False)):
        feature = str(node["feature"])
        if feature not in row.index:
            raise KeyError(f"Input row is missing encoded feature: {feature}")
        value = float(row[feature])
        if not np.isfinite(value):
            raise ValueError(f"Encoded feature must be finite: {feature}")
        node = node["left"] if value <= float(node["thr"]) else node["right"]
    return float(node["proba"][1])


def predict_many(tree: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """Predict p(class=1) for every encoded row."""
    return np.asarray([predict_one(tree, X.iloc[i]) for i in range(len(X))], dtype=float)


def _walk_rules(
    node: dict[str, Any],
    path: list[str],
    output: list[dict[str, Any]],
    bands: Sequence[float],
) -> None:
    if bool(node.get("is_leaf", False)):
        p1 = float(node["proba"][1])
        output.append(
            {
                "rule": " AND ".join(path) if path else "(all)",
                "p1": p1,
                "risk": risk_band(p1, bands),
                "n": int(node.get("n", 0)),
                "pos": int(node.get("pos", 0)),
                "neg": int(node.get("neg", 0)),
            }
        )
        return
    feature = str(node["feature"])
    threshold = float(node["thr"])
    _walk_rules(node["left"], path + [f"{feature} <= {threshold:.4g}"], output, bands)
    _walk_rules(node["right"], path + [f"{feature} > {threshold:.4g}"], output, bands)


def extract_rules(
    tree: dict[str, Any],
    max_rules: int | None = 10,
    sort_by: str = "p1_desc",
    bands: Sequence[float] = RISK_BANDS,
) -> list[dict[str, Any]]:
    """Return human-readable root-to-leaf rules."""
    rules: list[dict[str, Any]] = []
    _walk_rules(tree, [], rules, bands)
    if sort_by == "n_desc":
        rules.sort(key=lambda row: int(row["n"]), reverse=True)
    else:
        rules.sort(key=lambda row: float(row["p1"]), reverse=True)
    return rules if max_rules is None else rules[:max_rules]


# %% [markdown]
# ## 7) Visualisation and risk-band summaries

# %%
def plot_confusion_matrix(
    metrics: dict[str, float | int],
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot and optionally save a 2x2 confusion matrix."""
    matrix = np.array(
        [
            [int(metrics["tn"]), int(metrics["fp"])],
            [int(metrics["fn"]), int(metrics["tp"])],
        ]
    )
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    cutoff = matrix.max() / 2.0 if matrix.size else 0.0
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > cutoff else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=12)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def describe_risk_bands(
    y_true: Iterable[int],
    p1: Iterable[float],
    bands: Sequence[float] = RISK_BANDS,
) -> pd.DataFrame:
    """Count observations and positives in each descriptive probability band."""
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(p1, dtype=float)
    labels = np.asarray([risk_band(float(score), bands) for score in p_arr])
    rows: list[dict[str, float | int | str]] = []
    for label in ("low", "mid", "high"):
        mask = labels == label
        n = int(mask.sum())
        positives = int((y_arr[mask] == 1).sum()) if n else 0
        rows.append(
            {
                "band": label,
                "n": n,
                "pos": positives,
                "pos_rate": float(positives / n) if n else 0.0,
            }
        )
    return pd.DataFrame(rows)


def print_risk_bands_summary(rows: pd.DataFrame, title: str = "Risk bands summary") -> None:
    print(title)
    print(rows.to_string(index=False, formatters={"pos_rate": "{:.3f}".format}))


# %% [markdown]
# ## 8) Validation grid, final retraining, and logistic reference

# %%
def grid_search_by_cost(
    trainX: pd.DataFrame,
    trainY: pd.Series,
    valX: pd.DataFrame,
    valY: pd.Series,
    depths: Sequence[int] = (2, 3, 4, 5),
    min_leaves: Sequence[int] = (10, 20, 30, 40, 60),
    min_delta_cost: float = MIN_DELTA_COST,
    thr: float = BUSINESS_THR,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
    max_candidates: int | None = MAX_CANDIDATES,
    verbose: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select depth/min_leaf using validation total cost only."""
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for depth, min_leaf in product(depths, min_leaves):
        tree = build_tree_by_cost(
            trainX,
            trainY,
            max_depth=int(depth),
            min_leaf=int(min_leaf),
            min_delta_cost=min_delta_cost,
            thr=thr,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
            max_candidates=max_candidates,
        )
        train_p1 = predict_many(tree, trainX)
        val_p1 = predict_many(tree, valX)
        train_eval = evaluate_full(trainY, train_p1, thr, cost_fn, cost_fp, verbose=False)
        val_eval = evaluate_full(valY, val_p1, thr, cost_fn, cost_fp, verbose=False)
        row = {
            "depth": int(depth),
            "min_leaf": int(min_leaf),
            "train_cost": float(train_eval["cost"]),
            "val_cost": float(val_eval["cost"]),
            "train_rec": float(train_eval["rec"]),
            "val_rec": float(val_eval["rec"]),
            "val_fpr": float(val_eval["fpr"]),
            "train_f1": float(train_eval["f1"]),
            "val_f1": float(val_eval["f1"]),
        }
        rows.append(row)
        # Preserve the original notebook's first-in-grid tie break.
        if best is None or float(row["val_cost"]) < float(best["val_cost"]):
            best = {**row, "model": tree}

    if best is None:
        raise RuntimeError("Grid search produced no model.")

    results = pd.DataFrame(rows).sort_values(
        ["val_cost", "depth", "min_leaf"], kind="stable"
    ).reset_index(drop=True)
    if verbose:
        print("Top-5 on VAL by total cost:")
        print(
            results.head(5).to_string(
                index=False,
                formatters={
                    "train_cost": "{:.1f}".format,
                    "val_cost": "{:.1f}".format,
                    "train_rec": "{:.3f}".format,
                    "val_rec": "{:.3f}".format,
                    "val_fpr": "{:.3f}".format,
                    "train_f1": "{:.3f}".format,
                    "val_f1": "{:.3f}".format,
                },
            )
        )
    return best, results


def plot_ablation(
    results: pd.DataFrame,
    output_dir: str | Path,
    show: bool = True,
) -> tuple[Path, Path]:
    """Plot validation total cost against depth and minimum leaf size."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    depth_path = out / "ablation_depth.png"
    fig1, ax1 = plt.subplots(figsize=(6.2, 4.2))
    for min_leaf, group in results.groupby("min_leaf", sort=True):
        group = group.sort_values("depth")
        ax1.plot(group["depth"], group["val_cost"], marker="o", label=f"min_leaf={min_leaf}")
    ax1.set_xlabel("max_depth")
    ax1.set_ylabel("Validation total cost")
    ax1.set_title("Ablation: depth by minimum leaf size")
    ax1.legend(fontsize=8)
    fig1.tight_layout()
    fig1.savefig(depth_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig1)

    leaf_path = out / "ablation_min_leaf.png"
    fig2, ax2 = plt.subplots(figsize=(6.2, 4.2))
    for depth, group in results.groupby("depth", sort=True):
        group = group.sort_values("min_leaf")
        ax2.plot(group["min_leaf"], group["val_cost"], marker="o", label=f"depth={depth}")
    ax2.set_xlabel("min_leaf")
    ax2.set_ylabel("Validation total cost")
    ax2.set_title("Ablation: minimum leaf size by depth")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(leaf_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig2)

    return depth_path, leaf_path


def threshold_sensitivity(
    y_true: Iterable[int],
    p1: Iterable[float],
    thresholds: Iterable[float] | None = None,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> pd.DataFrame:
    """Evaluate validation cost near the theoretical business threshold."""
    if thresholds is None:
        thresholds = np.linspace(max(BUSINESS_THR - 0.03, 0.01), BUSINESS_THR + 0.03, 13)
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        result = evaluate_full(
            y_true,
            p1,
            thr=float(threshold),
            cost_fn=cost_fn,
            cost_fp=cost_fp,
            verbose=False,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "cost": float(result["cost"]),
                "recall": float(result["rec"]),
                "precision": float(result["prec"]),
                "fpr": float(result["fpr"]),
            }
        )
    return pd.DataFrame(rows)


# %% [markdown]
# ## 9) Deployment-style function interface and JSON persistence
#
# This is a local Python function interface, not an external web/API service.
# Inputs must already use the same encoded feature columns as training.

# %%
def predict_proba_and_risk(
    tree: dict[str, Any],
    X_input: dict[str, Any] | pd.Series | pd.DataFrame,
    thr: float = BUSINESS_THR,
    bands: Sequence[float] = RISK_BANDS,
) -> dict[str, float | int | str] | pd.DataFrame:
    """Predict one encoded row or a batch and return p1, risk, and binary label."""
    if isinstance(X_input, (dict, pd.Series)):
        p1 = predict_one(tree, X_input)
        return {
            "p1": float(p1),
            "risk": risk_band(float(p1), bands),
            "label": int(p1 >= thr),
        }
    if isinstance(X_input, pd.DataFrame):
        p1 = predict_many(tree, X_input)
        output = X_input.copy()
        output["p1"] = p1
        output["risk"] = [risk_band(float(score), bands) for score in p1]
        output["label"] = (p1 >= thr).astype(int)
        return output
    raise TypeError("X_input must be a dict, pandas Series, or pandas DataFrame.")


def save_tree(tree: dict[str, Any], path: str | Path = "mycosttree.json") -> Path:
    """Save the nested tree dictionary as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(tree, stream, ensure_ascii=False, indent=2)
    return output_path


def load_tree(path: str | Path = "mycosttree.json") -> dict[str, Any]:
    """Load a JSON tree."""
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def save_model_bundle(
    tree: dict[str, Any],
    preprocessor: StrokePreprocessor,
    path: str | Path,
    threshold: float = BUSINESS_THR,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> Path:
    """Save a complete raw-input educational model, without training records."""
    bundle = {
        "format_version": 2,
        "model_version": "corrected-reconstruction-v2",
        "intended_use": "educational only; not clinically validated",
        "preprocessor": preprocessor.to_dict(),
        "tree": tree,
        "threshold": float(threshold),
        "cost_fn": float(cost_fn),
        "cost_fp": float(cost_fp),
        "risk_bands": list(RISK_BANDS),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, allow_nan=False), encoding="utf-8")
    return path


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    if bundle.get("format_version") != 2:
        raise ValueError("Unsupported model bundle format.")
    StrokePreprocessor.from_dict(bundle["preprocessor"])
    if not 0 <= float(bundle["threshold"]) <= 1:
        raise ValueError("Invalid model threshold.")
    return bundle


def predict_raw(bundle: dict[str, Any], raw: pd.DataFrame) -> pd.DataFrame:
    """Predict from raw features using the saved training-only preprocessing."""
    preprocessor = StrokePreprocessor.from_dict(bundle["preprocessor"])
    X = preprocessor.transform(raw)
    result = predict_proba_and_risk(
        bundle["tree"], X, thr=bundle["threshold"], bands=bundle["risk_bands"]
    )
    return result[["p1", "risk", "label"]]


# %% [markdown]
# ## 10) Robustness: repeated selection and bootstrap uncertainty
#
# Two different uncertainty questions are kept separate:
#
# - **Repeated split/selection variability:** keep the original held-out test set
#   fixed, vary only the internal train/validation split, select hyperparameters,
#   retrain on the same non-test pool, and evaluate each selected model on the
#   fixed test set.
# - **Test-sample variability:** hold the fitted final model fixed and bootstrap its
#   test predictions.

# %%
def mean_ci_95(values: Iterable[float]) -> tuple[float, tuple[float, float], float]:
    """Return mean, normal-approximation 95% CI for the mean, and sample SD."""
    x = np.asarray(list(values), dtype=float)
    if len(x) == 0:
        raise ValueError("values must not be empty.")
    mean = float(np.mean(x))
    if len(x) == 1:
        return mean, (mean, mean), 0.0
    sd = float(np.std(x, ddof=1))
    half_width = 1.96 * sd / np.sqrt(len(x))
    return mean, (mean - half_width, mean + half_width), sd


def metric_total_cost_from_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> float:
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return float(cost_fn * fn + cost_fp * fp)


def metric_recall_from_labels(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return float(tp / max(tp + fn, 1))


def bootstrap_ci(
    y_true: Iterable[int],
    y_score_or_pred: Iterable[float | int],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    B: int = 1000,
    thr: float = BUSINESS_THR,
    use_score: bool = True,
    seed: int = 2025,
) -> tuple[float, tuple[float, float], np.ndarray]:
    """Non-parametric percentile bootstrap for a metric on fixed predictions."""
    if B < 1:
        raise ValueError("B must be at least 1.")
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y_true, dtype=int)
    base = np.asarray(y_score_or_pred)
    if len(y_arr) != len(base):
        raise ValueError("y_true and y_score_or_pred must have the same length.")

    values = np.empty(B, dtype=float)
    n = len(y_arr)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        sampled_y = y_arr[idx]
        sampled_pred = (
            (base[idx].astype(float) >= thr).astype(int)
            if use_score
            else base[idx].astype(int)
        )
        values[i] = float(metric_fn(sampled_y, sampled_pred))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(np.mean(values)), (float(low), float(high)), values


def repeated_retrains_fixed_test(
    trainX: pd.DataFrame,
    trainY: pd.Series,
    valX: pd.DataFrame,
    valY: pd.Series,
    testX: pd.DataFrame,
    testY: pd.Series,
    n_repeats: int = 30,
    depths: Sequence[int] = (2, 3, 4, 5),
    min_leaves: Sequence[int] = (10, 20, 30, 40, 60),
    thr: float = BUSINESS_THR,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
    verbose: bool = True,
) -> pd.DataFrame:
    """Diagnostic selection sensitivity on a fixed test set, receiving RAW rows.

    Each inner training fold learns its own preprocessing. This diagnostic is
    not independent validation and must never be used to choose a model.
    """
    X_non_test = pd.concat([trainX, valX], axis=0).reset_index(drop=True)
    y_non_test = pd.concat([trainY, valY], axis=0).reset_index(drop=True)

    records: list[dict[str, float | int]] = []
    for split_seed in range(n_repeats):
        X_train_i, X_val_i, y_train_i, y_val_i = train_test_split(
            X_non_test,
            y_non_test,
            test_size=0.20,
            stratify=y_non_test,
            random_state=split_seed,
        )
        X_train_i = X_train_i.reset_index(drop=True)
        X_val_i = X_val_i.reset_index(drop=True)
        y_train_i = y_train_i.reset_index(drop=True)
        y_val_i = y_val_i.reset_index(drop=True)

        fold_preprocessor = StrokePreprocessor().fit(X_train_i)
        best, _ = grid_search_by_cost(
            fold_preprocessor.transform(X_train_i),
            y_train_i,
            fold_preprocessor.transform(X_val_i),
            y_val_i,
            depths=depths,
            min_leaves=min_leaves,
            min_delta_cost=MIN_DELTA_COST,
            thr=thr,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
            verbose=False,
        )

        # Refit preprocessing and tree on non-test rows only.
        final_preprocessor = StrokePreprocessor().fit(X_non_test)
        model = build_tree_by_cost(
            final_preprocessor.transform(X_non_test),
            y_non_test,
            max_depth=int(best["depth"]),
            min_leaf=int(best["min_leaf"]),
            min_delta_cost=MIN_DELTA_COST,
            thr=thr,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
        )
        p_test = predict_many(model, final_preprocessor.transform(testX))
        result = evaluate_full(testY, p_test, thr, cost_fn, cost_fp, verbose=False)
        records.append(
            {
                "seed": split_seed,
                "cost": float(result["cost"]),
                "recall": float(result["rec"]),
                "precision": float(result["prec"]),
                "fpr": float(result["fpr"]),
                "depth": int(best["depth"]),
                "min_leaf": int(best["min_leaf"]),
            }
        )

    table = pd.DataFrame(records).sort_values(["cost", "seed"], kind="stable").reset_index(drop=True)
    if verbose:
        for column, label, decimals in (
            ("cost", "Total cost", 1),
            ("recall", "Recall", 3),
            ("fpr", "FPR", 3),
            ("precision", "Precision", 3),
        ):
            mean, interval, sd = mean_ci_95(table[column])
            print(
                f"{label}: mean={mean:.{decimals}f}, "
                f"95% CI=[{interval[0]:.{decimals}f}, {interval[1]:.{decimals}f}], "
                f"SD={sd:.{decimals}f}, N={len(table)}"
            )
    return table


def plot_repeated_distributions(
    repeated_results: pd.DataFrame,
    output_dir: str | Path,
    show: bool = True,
) -> tuple[Path, Path]:
    """Save report-style distributions for cost and recall across repeats."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cost_path = out / "repeated_cost_distribution.png"
    fig1, ax1 = plt.subplots(figsize=(6.2, 4.2))
    ax1.hist(repeated_results["cost"], bins="auto", color="#4C78A8", edgecolor="white")
    ax1.axvline(repeated_results["cost"].mean(), color="#D62728", linestyle="--", label="Mean")
    ax1.set_xlabel("Test total cost")
    ax1.set_ylabel("Count")
    ax1.set_title("Fixed-test cost across repeated train/validation selections")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(cost_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig1)

    recall_path = out / "repeated_recall_distribution.png"
    fig2, ax2 = plt.subplots(figsize=(6.2, 4.2))
    ax2.hist(repeated_results["recall"], bins="auto", color="#59A14F", edgecolor="white")
    ax2.axvline(repeated_results["recall"].mean(), color="#D62728", linestyle="--", label="Mean")
    ax2.set_xlabel("Test recall")
    ax2.set_ylabel("Count")
    ax2.set_title("Fixed-test recall across repeated train/validation selections")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(recall_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig2)

    return cost_path, recall_path


# %% [markdown]
# ## 11) Complete end-to-end experiment
#
# The function below produces all main tables, plots, JSON model output, the 30-run
# stability table, and the bootstrap summary. Set `run_robustness=False` for a quick
# first run.

# %%
def run_complete_analysis(
    csv_path: str | Path,
    output_dir: str | Path = "stroke_tree_outputs",
    run_robustness: bool = False,
    n_repeats: int = 30,
    bootstrap_B: int = 1000,
    show_plots: bool = True,
) -> dict[str, Any]:
    """Run corrected experiments; historical scores are not expected outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print_configuration()

    raw = load_stroke_csv(csv_path)
    y = extract_target(raw)
    (raw_train, trainY), (raw_val, valY), (raw_test, testY) = split_train_val_test(raw, y)
    selection_preprocessor = StrokePreprocessor().fit(raw_train)
    trainX = selection_preprocessor.transform(raw_train)
    valX = selection_preprocessor.transform(raw_val)
    testX = selection_preprocessor.transform(raw_test)
    print(f"Using CSV: {csv_path}")
    print(f"Rows: {len(raw)}; training-encoded features: {len(trainX.columns)}")
    print(f"Split sizes: train={len(trainX)}, val={len(valX)}, test={len(testX)}")
    print(f"Positive counts: train={int(trainY.sum())}, val={int(valY.sum())}, test={int(testY.sum())}")

    # Majority baseline
    majority_label = majority_baseline_fit(trainY)
    baseline_results: dict[str, dict[str, float | int]] = {}
    print("\n== Majority baseline ==")
    for split_name, split_y in (("train", trainY), ("val", valY), ("test", testY)):
        print(f"-- {split_name.upper()} --")
        probabilities = majority_baseline_predict_proba(len(split_y), majority_label)
        baseline_results[split_name] = evaluate_full(split_y, probabilities)

    # Demonstrate the first split evidence reported in the journal.
    print("\n== First-split probes ==")
    for feature in ("age", "avg_glucose_level"):
        if feature in trainX.columns:
            probe = best_threshold_for_feature_by_cost(trainX[feature], trainY)
            compact = (
                None
                if probe is None
                else {
                    "feature_threshold": probe["feature_threshold"],
                    "delta_cost": probe["delta_cost"],
                }
            )
            print(f"{feature}: {compact}")
    global_probe = best_split_over_features_by_cost(trainX, trainY)
    if global_probe is not None:
        print(
            "Best over all features:",
            {
                "feature": global_probe["feature"],
                "feature_threshold": global_probe["feature_threshold"],
                "delta_cost": global_probe["delta_cost"],
            },
        )

    # Default tree, used for train/validation learning diagnostics.
    default_tree = build_tree_by_cost(
        trainX,
        trainY,
        max_depth=MAX_DEPTH,
        min_leaf=MIN_LEAF,
        min_delta_cost=MIN_DELTA_COST,
    )
    default_train_p1 = predict_many(default_tree, trainX)
    default_val_p1 = predict_many(default_tree, valX)
    print("\n== Default MyCostTree: train/validation diagnostic ==")
    default_train_eval = evaluate_full(trainY, default_train_p1)
    default_val_eval = evaluate_full(valY, default_val_p1)
    print_ein_eout_summary("MyCostTree", default_train_eval, default_val_eval)

    print("Top default-tree rules:")
    for i, rule in enumerate(extract_rules(default_tree, max_rules=8), start=1):
        print(
            f"{i}. [{rule['risk']}] p1={rule['p1']:.3f} | n={rule['n']} | "
            f"pos={rule['pos']} | {rule['rule']}"
        )

    # Hyperparameter selection on validation only.
    print("\n== Validation grid search ==")
    best, grid_results = grid_search_by_cost(trainX, trainY, valX, valY)
    grid_results.to_csv(output_dir / "grid_search_results.csv", index=False)
    plot_ablation(grid_results, output_dir, show=show_plots)
    print(
        f"Selected: depth={int(best['depth'])}, min_leaf={int(best['min_leaf'])}, "
        f"VAL cost={float(best['val_cost']):.1f}"
    )

    best_val_p1 = predict_many(best["model"], valX)
    sensitivity = threshold_sensitivity(valY, best_val_p1)
    sensitivity.to_csv(output_dir / "validation_threshold_sensitivity.csv", index=False)

    # Final retraining on train+validation, followed by final test evaluation.
    raw_train_val = pd.concat([raw_train, raw_val], axis=0).reset_index(drop=True)
    final_preprocessor = StrokePreprocessor().fit(raw_train_val)
    X_train_val = final_preprocessor.transform(raw_train_val)
    testX = final_preprocessor.transform(raw_test)
    y_train_val = pd.concat([trainY, valY], axis=0).reset_index(drop=True)
    final_tree = build_tree_by_cost(
        X_train_val,
        y_train_val,
        max_depth=int(best["depth"]),
        min_leaf=int(best["min_leaf"]),
        min_delta_cost=MIN_DELTA_COST,
    )
    final_train_val_p1 = predict_many(final_tree, X_train_val)
    final_test_p1 = predict_many(final_tree, testX)

    print("\n== Final tree: train+validation ==")
    final_train_val_eval = evaluate_full(y_train_val, final_train_val_p1)
    print("\n== Final tree: held-out test ==")
    final_test_eval = evaluate_full(testY, final_test_p1)

    final_rules = pd.DataFrame(extract_rules(final_tree, max_rules=None))
    final_rules.to_csv(output_dir / "final_tree_rules.csv", index=False)
    print("Top final-tree rules:")
    print(final_rules.head(8).to_string(index=False, formatters={"p1": "{:.3f}".format}))

    plot_confusion_matrix(
        final_test_eval,
        title="MyCostTree (held-out test)",
        save_path=output_dir / "mycosttree_test_confusion.png",
        show=show_plots,
    )
    risk_table = describe_risk_bands(testY, final_test_p1)
    risk_table.to_csv(output_dir / "mycosttree_test_risk_bands.csv", index=False)
    print_risk_bands_summary(risk_table, "Final-tree test risk bands")

    # Reference logistic regression under the same business threshold.
    logistic = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(solver="lbfgs", max_iter=2000, random_state=SEED),
    )
    logistic.fit(X_train_val, y_train_val)
    logistic_train_val_p1 = logistic.predict_proba(X_train_val)[:, 1]
    logistic_test_p1 = logistic.predict_proba(testX)[:, 1]
    print("\n== Logistic regression: train+validation ==")
    logistic_train_val_eval = evaluate_full(y_train_val, logistic_train_val_p1)
    print("\n== Logistic regression: held-out test ==")
    logistic_test_eval = evaluate_full(testY, logistic_test_p1)

    summary = pd.DataFrame(
        [
            {"model": "MajorityBaseline", "split": "test", **baseline_results["test"]},
            {"model": "MyCostTree", "split": "test", **final_test_eval},
            {"model": "LogisticRegression", "split": "test", **logistic_test_eval},
        ]
    )
    summary.to_csv(output_dir / "test_model_summary.csv", index=False)
    print("\n== Held-out test summary ==")
    print(
        summary[["model", "cost", "acc", "prec", "rec", "fpr", "f1"]].to_string(
            index=False,
            formatters={
                "cost": "{:.1f}".format,
                "acc": "{:.3f}".format,
                "prec": "{:.3f}".format,
                "rec": "{:.3f}".format,
                "fpr": "{:.3f}".format,
                "f1": "{:.3f}".format,
            },
        )
    )

    # Save/load consistency check.
    tree_path = save_tree(final_tree, output_dir / "mycosttree.json")
    loaded_tree = load_tree(tree_path)
    p_before = predict_many(final_tree, testX)
    p_after = predict_many(loaded_tree, testX)
    reload_max_delta = float(np.max(np.abs(p_before - p_after)))
    print(f"Save/load Max |delta| = {reload_max_delta:.12g}")
    if reload_max_delta != 0.0:
        raise AssertionError("Tree predictions changed after JSON save/load.")

    bundle_path = save_model_bundle(final_tree, final_preprocessor, output_dir / "model_bundle.json")
    bundle_predictions = predict_raw(load_model_bundle(bundle_path), raw_test)
    if not np.array_equal(final_test_p1, bundle_predictions["p1"].to_numpy()):
        raise AssertionError("Raw-input predictions changed after bundle save/load.")

    split_positions = split_row_positions(y)
    manifest = {
        "model_version": "corrected-reconstruction-v2",
        "historical_results_reproduced": False,
        "dataset_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        "rows": len(raw), "positive_labels": int(y.sum()),
        "split_seed": SEED,
        "split_method": "stratified raw rows, then train-only preprocessing",
        "split_row_positions": dict(zip(["train", "validation", "test"],
                                       [i.tolist() for i in split_positions])),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ["numpy", "pandas", "matplotlib", "scikit-learn"]},
        "cost_fn": COST_FN, "cost_fp": COST_FP, "threshold": BUSINESS_THR,
        "threshold_candidates": "all distinct-value midpoints",
        "final_config": {"depth": int(best["depth"]), "min_leaf": int(best["min_leaf"])},
        "final_preprocessor_fit_rows": len(raw_train_val),
        "selection_preprocessor_fit_rows": len(raw_train),
        "robustness_requested": run_robustness,
        "robustness_repeats": n_repeats if run_robustness else 0,
        "bootstrap_resamples": bootstrap_B if run_robustness else 0,
        "selection_grid": {"depths": [2, 3, 4, 5], "min_leaves": [10, 20, 30, 40, 60]},
        "test_used_for_selection": False, "clinical_validation": False,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )

    robustness: dict[str, Any] = {}
    if run_robustness:
        print("\n== Repeated train/validation selection with fixed test set ==")
        repeated = repeated_retrains_fixed_test(
            raw_train,
            trainY,
            raw_val,
            valY,
            raw_test,
            testY,
            n_repeats=n_repeats,
        )
        repeated.to_csv(output_dir / "repeated_fixed_test_results.csv", index=False)
        plot_repeated_distributions(repeated, output_dir, show=show_plots)

        print("\n== Bootstrap uncertainty of fixed final-tree test predictions ==")
        bootstrap_cost_mean, bootstrap_cost_ci, bootstrap_cost_values = bootstrap_ci(
            testY,
            final_test_p1,
            metric_total_cost_from_labels,
            B=bootstrap_B,
            thr=BUSINESS_THR,
            use_score=True,
            seed=2025,
        )
        bootstrap_recall_mean, bootstrap_recall_ci, bootstrap_recall_values = bootstrap_ci(
            testY,
            final_test_p1,
            metric_recall_from_labels,
            B=bootstrap_B,
            thr=BUSINESS_THR,
            use_score=True,
            seed=2025,
        )
        bootstrap_table = pd.DataFrame(
            {
                "replicate": np.arange(bootstrap_B),
                "cost": bootstrap_cost_values,
                "recall": bootstrap_recall_values,
            }
        )
        bootstrap_table.to_csv(output_dir / "bootstrap_test_results.csv", index=False)
        print(
            f"Total cost: bootstrap mean={bootstrap_cost_mean:.1f}, "
            f"95% CI=[{bootstrap_cost_ci[0]:.1f}, {bootstrap_cost_ci[1]:.1f}]"
        )
        print(
            f"Recall: bootstrap mean={bootstrap_recall_mean:.3f}, "
            f"95% CI=[{bootstrap_recall_ci[0]:.3f}, {bootstrap_recall_ci[1]:.3f}]"
        )

        robustness = {
            "repeated_results": repeated,
            "bootstrap_cost_mean": bootstrap_cost_mean,
            "bootstrap_cost_ci": bootstrap_cost_ci,
            "bootstrap_recall_mean": bootstrap_recall_mean,
            "bootstrap_recall_ci": bootstrap_recall_ci,
        }

    return {
        "y": y,
        "preprocessor": final_preprocessor,
        "manifest": manifest,
        "model_bundle": bundle_path,
        "train": (final_preprocessor.transform(raw_train), trainY),
        "val": (final_preprocessor.transform(raw_val), valY),
        "selection_preprocessor": selection_preprocessor,
        "selection_train": (trainX, trainY),
        "selection_val": (valX, valY),
        "test": (testX, testY),
        "default_tree": default_tree,
        "best_config": best,
        "grid_results": grid_results,
        "final_tree": final_tree,
        "final_test_probabilities": final_test_p1,
        "final_test_evaluation": final_test_eval,
        "logistic_pipeline": logistic,
        "logistic_test_evaluation": logistic_test_eval,
        "summary": summary,
        "reload_max_delta": reload_max_delta,
        "robustness": robustness,
        "output_dir": output_dir,
    }


# %% [markdown]
# ## 12) Notebook run cell
#
# In Colab, upload the CSV or mount Drive, set `CSV_PATH`, and run:
#
# ```python
# CSV_PATH = "/content/drive/MyDrive/MachineLearning/healthcare-dataset-stroke-data.csv"
# results = run_complete_analysis(CSV_PATH)
# ```
#
# Use `run_robustness=False` for a quicker first pass.

# %%
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=str, default=None, help="Path to healthcare-dataset-stroke-data.csv")
    parser.add_argument("--output-dir", type=str, default="stroke_tree_outputs")
    parser.add_argument("--run-robustness", action="store_true",
                        help="Optional repeated-selection diagnostic on the fixed test set")
    parser.add_argument("--skip-robustness", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--no-show", action="store_true", help="Save plots without opening plot windows")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_path = resolve_csv_path(args.csv)
    run_complete_analysis(
        dataset_path,
        output_dir=args.output_dir,
        run_robustness=args.run_robustness and not args.skip_robustness,
        n_repeats=args.repeats,
        bootstrap_B=args.bootstrap,
        show_plots=not args.no_show,
    )
