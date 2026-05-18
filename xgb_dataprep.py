#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
AUTOMATED DATA PREPARATION FOR XGBOOST ON SAGEMAKER
=============================================================================
All rules are hardcoded in the code (see the constants block below). No YAML.

TWO MAIN FUNCTIONS:

  prepare_training_data(train_csv, target_col, output_dir, ...)
      Analyzes the training data, fits all transformations (types, drops,
      encoders, target encoding, imbalance handling) and saves:
        * train.csv, validation.csv - in SageMaker XGBoost format
        * pipeline.pkl - the fitted mapper/encoder for reuse
        * encoders.json - human-readable description of encoders and maps
        * training_config.json - objective, scale_pos_weight, hyperparameters,
          plus hyperparameter ranges for SageMaker Automatic Model Tuning
        * feature_schema.json - final column order (for inference)
        * decision_log.csv - what was done and why

  prepare_test_data(test_csv, pipeline_pkl, output_dir, ...)
      Loads pipeline.pkl produced on the training data and applies the SAME
      transformations to the test data. Nothing is recomputed.
      Saves test.csv in SageMaker XGBoost format.

OUTPUT CSV FORMAT (required by SageMaker built-in XGBoost):
  target = FIRST column, NO header, numeric values only, NaN = empty.

CLI:
  python xgb_dataprep.py train data.csv --target churned --output-dir ./out
  python xgb_dataprep.py test  test.csv --pipeline ./out/pipeline.pkl --output-dir ./out

Dependencies: pandas, numpy, scikit-learn
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split, GroupShuffleSplit


# =============================================================================
# RULES AND THRESHOLDS (hardcoded). Change them here - this is the single
# source of truth.
# =============================================================================
HIGH_CARDINALITY_RATIO   = 0.70   # >70% unique in a categorical column -> identifier column
HIGH_CARDINALITY_ABS_CAP = 10000  # or an absolute cap on the number of unique values
QUASI_CONSTANT_SHARE     = 0.99   # one value covers >99% of rows -> drop
HIGH_MISSING_RATIO       = 0.70   # >70% missing -> drop (keeping an is_missing flag)
ZERO_VARIANCE_STD        = 1e-8   # std below this -> numeric constant
LEAKAGE_TARGET_CORR      = 0.98   # |correlation with target| above this -> leakage suspect, drop
MULTICOLLINEAR_CORR      = 0.97   # |correlation| between features above this -> drop one of the pair
ONEHOT_MAX_CARDINALITY   = 15     # <=15 unique -> one-hot, otherwise target/frequency encoding
RARE_CATEGORY_SHARE      = 0.01   # category with frequency <1% -> merge into '__other__'
TARGET_SKEW_THRESHOLD    = 1.0    # |skew| of target above this (regression, >0) -> log1p transform
IMBALANCE_RATIO          = 5.0    # n_major/n_minor above this -> compute scale_pos_weight
CLASSIFICATION_MAX_UNIQUE = 20    # <=20 unique numeric target values -> classification (for task='auto')
TARGET_ENC_SMOOTHING     = 10.0   # smoothing for target encoding
TARGET_ENC_FOLDS         = 5      # number of folds for out-of-fold target encoding

MISSING_TOKEN = "__missing__"
OTHER_TOKEN   = "__other__"

# Name patterns for automatic detection of problematic columns
ID_NAME_PATTERNS    = ["id", "uuid", "guid", "index", "row_number", "rownum", "hash", "key"]
LEAKY_NAME_PATTERNS = ["outcome", "result", "final", "is_fraud", "churned",
                       "closed", "post_", "after_", "_leak"]
BOOLEAN_TOKENS = {"true", "false", "yes", "no", "t", "f", "y", "n", "0", "1"}

# Default XGBoost hyperparameters (baseline before tuning)
HP_DEFAULTS = {
    "num_round": 1000, "early_stopping_rounds": 30, "eta": 0.1, "max_depth": 6,
    "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8,
    "gamma": 0, "lambda": 1, "alpha": 0, "verbosity": 1,
}
SAGEMAKER_XGBOOST_VERSION = "1.7-1"


# =============================================================================
# ADAPTIVE RANGES FOR SAGEMAKER AUTOMATIC MODEL TUNING (AMT)
# -----------------------------------------------------------------------------
# AMT is Bayesian optimization on the AWS side: you provide RANGES (not concrete
# values), and AMT launches many training jobs to search for the best
# combination within those ranges. AMT does not adjust the bounds itself - it
# only searches inside what you give it, so the bounds are set here based on
# the data.
#
# Adaptation logic by dataset size (based on XGBoost tuning practices):
#   * Small data overfits easily -> narrower ranges, fewer jobs, less parallelism.
#   * Deep trees memorize small data -> smaller max_depth for small datasets.
#   * subsample/colsample below 0.6 -> underfitting; for small data keep them
#     closer to 1.0, for large data they can go lower (regularization + speed).
#   * min_child_weight grows with size: a leaf with 1 row out of millions = noise.
#   * Large data: lower eta; regularization is less necessary.
# Size buckets by row count: tiny <1k, small <10k, medium <100k, large >=100k.

_AMT_SIZE_BUCKETS = [(1000, "tiny"), (10000, "small"), (100000, "medium")]

# parameter -> {bucket: (min, max)} for tuning
_AMT_RANGES = {
    "max_depth": {
        "tiny": (2, 4), "small": (3, 6), "medium": (4, 8), "large": (5, 10),
    },
    "min_child_weight": {
        "tiny": (1.0, 4.0), "small": (1.0, 8.0),
        "medium": (1.0, 15.0), "large": (1.0, 40.0),
    },
    "eta": {
        "tiny": (0.05, 0.3), "small": (0.03, 0.3),
        "medium": (0.02, 0.2), "large": (0.01, 0.1),
    },
    "subsample": {
        "tiny": (0.8, 1.0), "small": (0.7, 1.0),
        "medium": (0.6, 1.0), "large": (0.5, 0.9),
    },
    "gamma": {
        "tiny": (0.0, 5.0), "small": (0.0, 5.0),
        "medium": (0.0, 3.0), "large": (0.0, 1.0),
    },
    "lambda": {
        "tiny": (1.0, 20.0), "small": (1.0, 10.0),
        "medium": (0.5, 5.0), "large": (0.1, 3.0),
    },
    "alpha": {
        "tiny": (0.0, 5.0), "small": (0.0, 5.0),
        "medium": (0.0, 2.0), "large": (0.0, 1.0),
    },
}
# AMT parameter type and scaling_type:
#   integer/continuous; Logarithmic for quantities spanning a wide range of orders.
_AMT_PARAM_TYPE = {
    "max_depth":        ("integer",    "Auto"),
    "min_child_weight": ("continuous", "Logarithmic"),
    "eta":              ("continuous", "Logarithmic"),
    "subsample":        ("continuous", "Linear"),
    "gamma":            ("continuous", "Linear"),
    "lambda":           ("continuous", "Logarithmic"),
    "alpha":            ("continuous", "Linear"),
    "colsample_bytree": ("continuous", "Linear"),
}
# colsample_bytree depends on the NUMBER OF FEATURES, not rows (tree decorrelation)
_AMT_COLSAMPLE = {
    "narrow": (0.7, 1.0),   # <10 features - cannot drop many
    "medium": (0.6, 1.0),   # 10-50 features
    "wide":   (0.4, 0.9),   # >50 features - decorrelation is valuable
}
# (max_jobs, max_parallel_jobs) by dataset size.
# Keep parallelism moderate: Bayesian optimization learns sequentially.
_AMT_JOBS = {
    "tiny":   (20, 2),   # fewer jobs - protects against overtuning small data
    "small":  (30, 3),
    "medium": (50, 5),
    "large":  (50, 5),   # jobs are expensive - do not take more
}
# eval_metric -> (AMT metric name, optimization direction)
# SageMaker XGBoost publishes metrics in the 'validation:<metric>' format.
_AMT_METRIC = {
    "auc":      ("validation:auc", "Maximize"),
    "aucpr":    ("validation:aucpr", "Maximize"),
    "rmse":     ("validation:rmse", "Minimize"),
    "mae":      ("validation:mae", "Minimize"),
    "logloss":  ("validation:logloss", "Minimize"),
    "mlogloss": ("validation:mlogloss", "Minimize"),
    "error":    ("validation:error", "Minimize"),
    "merror":   ("validation:merror", "Minimize"),
}


def _amt_size_bucket(n_rows):
    for limit, name in _AMT_SIZE_BUCKETS:
        if n_rows < limit:
            return name
    return "large"


def build_amt_ranges(n_rows, n_features, eval_metric, imbalanced=False):
    """Build the Automatic Model Tuning section based on dataset characteristics.

    n_rows, n_features : size of the prepared dataset.
    eval_metric        : validation metric ('auc', 'rmse', 'mlogloss', ...).
    imbalanced         : True -> shrink max(min_child_weight) (otherwise a large
                         min_child_weight blocks splits on the rare class).

    Returns a dict ready to be placed into training_config.json under the
    'automatic_model_tuning' key and maps directly onto SageMaker
    HyperparameterTuner.
    """
    bucket = _amt_size_bucket(n_rows)

    # --- ranges for the parameters being tuned ---
    ranges = {}
    for param, buckets in _AMT_RANGES.items():
        lo, hi = buckets[bucket]
        ptype, scaling = _AMT_PARAM_TYPE[param]
        ranges[param] = {"type": ptype, "min_value": lo, "max_value": hi,
                         "scaling_type": scaling}

    # colsample_bytree - by number of features
    if n_features < 10:
        c_lo, c_hi = _AMT_COLSAMPLE["narrow"]
    elif n_features < 50:
        c_lo, c_hi = _AMT_COLSAMPLE["medium"]
    else:
        c_lo, c_hi = _AMT_COLSAMPLE["wide"]
    if bucket in ("tiny", "small"):        # for small data, do not drop many columns
        c_lo = max(c_lo, 0.7)
    ranges["colsample_bytree"] = {"type": "continuous", "min_value": c_lo,
                                  "max_value": c_hi, "scaling_type": "Linear"}

    # imbalance: a large min_child_weight blocks splits on the rare class
    if imbalanced:
        ranges["min_child_weight"]["max_value"] = min(
            ranges["min_child_weight"]["max_value"], 8.0)

    metric_name, direction = _AMT_METRIC.get(
        eval_metric, (f"validation:{eval_metric}", "Minimize"))
    max_jobs, max_parallel = _AMT_JOBS[bucket]

    notes = (f"dataset '{bucket}' ({n_rows} rows, {n_features} features); "
             f"max_depth {ranges['max_depth']['min_value']}-"
             f"{ranges['max_depth']['max_value']}; "
             f"{'min_child_weight narrowed due to imbalance; ' if imbalanced else ''}"
             f"{max_jobs} jobs, {max_parallel} in parallel. "
             f"num_round is not tuned - it is fixed in static_hyperparameters, "
             f"the tree count is bounded by early_stopping_rounds.")

    return {
        "strategy": "Bayesian",
        "objective_metric_name": metric_name,
        "objective_type": direction,
        "max_jobs": max_jobs,
        "max_parallel_jobs": max_parallel,
        "hyperparameter_ranges": ranges,
        "dataset_basis": {"size_bucket": bucket, "n_rows": n_rows,
                          "n_features": n_features, "imbalanced": imbalanced,
                          "notes": notes},
        "usage_hint": ("hyperparameter_ranges -> sagemaker.tuner "
                       "ContinuousParameter/IntegerParameter; "
                       "static_hyperparameters -> estimator.set_hyperparameters(); "
                       "objective_metric_name + objective_type -> HyperparameterTuner."),
    }


# =============================================================================
# HELPER FUNCTIONS FOR TYPE DETECTION AND CONVERSION
# =============================================================================
def _try_numeric(series):
    """Convert a string column to numeric: strip commas, currency symbols, spaces."""
    cleaned = (series.astype(str)
               .str.strip()
               .str.replace(",", "", regex=False)
               .str.replace(r"[$€£₴]", "", regex=True)
               .replace({"": np.nan, "nan": np.nan, "none": np.nan, "None": np.nan,
                         "NaN": np.nan, "null": np.nan}))
    return pd.to_numeric(cleaned, errors="coerce")


def _is_numeric_convertible(series, ratio=0.99):
    non_null = series.dropna()
    if non_null.empty:
        return False
    return _try_numeric(non_null).notna().mean() >= ratio


def _try_datetime(series):
    return pd.to_datetime(series, errors="coerce", format="mixed")


def _is_datetime(series, ratio=0.95):
    """True if a string column reliably parses as a date within years 1900-2100."""
    if pd.api.types.is_numeric_dtype(series):
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    sample = non_null.sample(min(len(non_null), 500), random_state=0)
    parsed = _try_datetime(sample)
    if parsed.notna().mean() < ratio:
        return False
    years = parsed.dropna().dt.year
    return bool(len(years)) and years.between(1900, 2100).mean() > 0.9


def _is_boolean(series):
    uniq = set(series.dropna().astype(str).str.strip().str.lower().unique())
    return 0 < len(uniq) <= 2 and uniq.issubset(BOOLEAN_TOKENS)


def _to_boolean(series):
    mapping = {"true": 1, "yes": 1, "t": 1, "y": 1, "1": 1,
               "false": 0, "no": 0, "f": 0, "n": 0, "0": 0}
    return series.astype(str).str.strip().str.lower().map(mapping)


def _json_safe(obj):
    """Make a structure suitable for json.dump (set->list, numpy->python)."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (set, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# =============================================================================
# MAIN CLASS: FITTED ON TRAIN, APPLIED TO TRAIN/VAL/TEST
# =============================================================================
class XGBoostDataPrep:
    """Analyzes data, fits all transformations and applies them.

    fit(df)            - learn the transformations on the training data.
    fit_transform(df)  - fit + return the ready train set (with out-of-fold target encoding).
    transform(df)      - apply the fitted transformations to val/test.
    """

    def __init__(self, target_col, task="auto", time_col=None,
                 group_col=None, random_state=42):
        self.target_col = target_col
        self.task = task                 # 'auto' | 'binary' | 'multiclass' | 'regression'
        self.time_col = time_col
        self.group_col = group_col
        self.random_state = random_state

        # --- fitted state (this is the "mapper/encoder") ---
        self.schema = {}                 # column -> type
        self.dropped = {}                # column -> drop reason
        self.added_flag_cols = []        # created is_missing_* columns
        self.datetime_cols = []          # remaining datetime columns (for FE)
        self.rare_maps = {}              # column -> set of kept categories
        self.encoding_strategy = {}      # column -> 'onehot' | 'target' | 'frequency'
        self.onehot_categories = {}      # column -> ordered list of categories
        self.target_enc_maps = {}        # column -> {category: value}
        self.target_enc_global = {}      # column -> global mean of the target
        self.freq_maps = {}              # column -> {category: frequency}
        self.feature_order = []          # final order of numeric features
        # --- target ---
        self.resolved_task = None
        self.label_mapping = None        # classification: original value -> 0..n-1
        self.num_class = None
        self.log_transform_target = False
        self.scale_pos_weight = None
        # --- bookkeeping ---
        self.decisions = []              # decision log
        self.n_rows_fit = 0
        self.fitted = False

    # ---- decision logging ---------------------------------------------------
    def _log(self, stage, column, action, detail, reason):
        self.decisions.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage, "column": column, "action": action,
            "detail": detail, "reason": reason,
        })

    # =========================================================================
    # FIT - learn the transformations on the training data
    # =========================================================================
    def fit(self, df):
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' is not in the dataset.")
        df = df.copy()
        self.n_rows_fit = len(df)

        self._detect_schema(df)
        df = self._apply_type_conversions(df)
        self._decide_drops(df)
        self._detect_leakage(df)
        df = self._drop_columns(df)
        self._detect_multicollinearity(df)
        df = self._drop_columns(df)
        self._fit_target(df)
        self._fit_categorical_encoders(df)

        # set the final column order by running a trial transformation
        sample_out = self._transform_features(df, use_oof=False)
        self.feature_order = list(sample_out.columns)
        self.fitted = True
        return self

    def fit_transform(self, df):
        """Fit + ready train set. Target-encoded columns are computed out-of-fold."""
        self.fit(df)
        out = self.transform(df)
        # recompute target-encoded columns out-of-fold (no target leakage)
        oof_cols = [c for c, s in self.encoding_strategy.items() if s == "target"]
        if oof_cols:
            work = self._apply_type_conversions(df.copy())
            y = self._encode_target_series(work[self.target_col])
            for col in oof_cols:
                oof = self._compute_oof_target_encoding(work[col], y, col)
                out[f"te_{col}"] = oof.values
        return out

    # =========================================================================
    # TRANSFORM - apply the fitted transformations (val / test)
    # =========================================================================
    def transform(self, df):
        if not self.fitted:
            raise RuntimeError("Call fit() or fit_transform() first.")
        df = df.copy()
        has_target = self.target_col in df.columns

        df = self._apply_type_conversions(df, fitting=False)
        features = self._transform_features(df, use_oof=False)
        features = features.reindex(columns=self.feature_order, fill_value=0)

        if has_target:
            y = self._encode_target_series(df[self.target_col])
            out = pd.concat([y.rename(self.target_col).reset_index(drop=True),
                             features.reset_index(drop=True)], axis=1)
        else:
            # inference mode: no target, return features only
            out = features.reset_index(drop=True)
        return out

    # ---- STAGE 1: type detection -------------------------------------------
    def _detect_schema(self, df):
        for col in df.columns:
            if col == self.target_col:
                self.schema[col] = "target"
                continue
            s = df[col]
            if pd.api.types.is_numeric_dtype(s) and not _is_boolean(s):
                self.schema[col] = "numeric"
            elif _is_boolean(s):
                self.schema[col] = "boolean"
            elif _is_datetime(s):
                self.schema[col] = "datetime"
            elif _is_numeric_convertible(s):
                self.schema[col] = "numeric_str"
            else:
                self.schema[col] = "categorical"
        self._log("01_schema", "*", "detect_types",
                  {k: v for k, v in self.schema.items()},
                  "type classification is needed to choose the processing")

    def _apply_type_conversions(self, df, fitting=True):
        df = df.copy()
        for col, typ in self.schema.items():
            if col not in df.columns:
                continue
            if typ == "numeric_str":
                df[col] = _try_numeric(df[col])
                if fitting:
                    self._log("01_schema", col, "convert_numeric",
                              "number stored as text",
                              "otherwise it would be encoded as a category")
            elif typ == "boolean":
                df[col] = _to_boolean(df[col])
                if fitting:
                    self._log("01_schema", col, "convert_boolean",
                              "2 boolean values", "a binary feature must be numeric")
            elif typ == "datetime":
                df[col] = _try_datetime(df[col])
                if fitting:
                    self._log("01_schema", col, "parse_datetime",
                              "date column", "needs decomposition into numeric features")
            elif typ == "numeric":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    # ---- STAGE 2: drop non-representative columns ---------------------------
    def _decide_drops(self, df):
        n = len(df)
        feature_cols = [c for c in df.columns if c != self.target_col]

        for col in feature_cols:
            s = df[col]
            # constant
            if s.nunique(dropna=False) <= 1:
                self.dropped[col] = "constant_column"
                self._log("02_drop", col, "drop", "1 unique value",
                          "zero variance - no signal for the trees")
                continue
            # quasi-constant
            top_share = s.value_counts(normalize=True, dropna=False).iloc[0]
            if top_share > QUASI_CONSTANT_SHARE:
                self.dropped[col] = f"quasi_constant ({top_share:.1%})"
                self._log("02_drop", col, "drop", f"{top_share:.1%} of one value",
                          "a near-constant carries almost no information")
                continue
            # high missing ratio
            miss = s.isna().mean()
            if miss > HIGH_MISSING_RATIO:
                self.dropped[col] = f"high_missing ({miss:.1%})"
                self.added_flag_cols.append(col)   # instead keep an is_missing flag
                self._log("02_drop", col, "drop+flag", f"{miss:.1%} missing",
                          "column is rarely useful, but the missingness is kept as is_missing_*")
                continue
            # zero variance for numeric columns
            if self.schema.get(col) in ("numeric", "numeric_str", "boolean"):
                std = pd.to_numeric(s, errors="coerce").std(skipna=True)
                if std is not None and not np.isnan(std) and std < ZERO_VARIANCE_STD:
                    self.dropped[col] = "zero_variance_numeric"
                    self._log("02_drop", col, "drop", "std ~ 0",
                              "a numeric constant is useless for splits")
                    continue
            # high cardinality categorical -> identifier
            if self.schema.get(col) == "categorical":
                nun = s.nunique(dropna=True)
                ratio = nun / n if n else 0
                if ratio > HIGH_CARDINALITY_RATIO or nun > HIGH_CARDINALITY_ABS_CAP:
                    self.dropped[col] = f"high_cardinality ({ratio:.1%}, {nun} unique)"
                    self._log("02_drop", col, "drop",
                              f"{ratio:.1%} unique values ({nun})",
                              "near-unique values = an identifier, not a feature")
                    continue
            # identifier by column name
            low = col.lower()
            if any(p == low or low.endswith("_" + p) or low.startswith(p + "_")
                   for p in ID_NAME_PATTERNS):
                self.dropped[col] = "identifier_by_name"
                self._log("02_drop", col, "drop", "name looks like an identifier",
                          "technical IDs have no predictive value")

        # duplicate columns
        seen = {}
        for col in feature_cols:
            if col in self.dropped:
                continue
            key = tuple(pd.util.hash_pandas_object(
                df[col].astype("object").fillna("__na__"), index=False).values[:5000])
            if key in seen:
                self.dropped[col] = f"duplicate_of '{seen[key]}'"
                self._log("02_drop", col, "drop", f"duplicate of '{seen[key]}'",
                          "duplicates distort feature importance")
            else:
                seen[key] = col

    # ---- STAGE 3: target leakage -------------------------------------------
    def _detect_leakage(self, df):
        feature_cols = [c for c in df.columns
                        if c != self.target_col and c not in self.dropped]
        # suspicious names
        for col in feature_cols:
            low = col.lower()
            if any(p in low for p in LEAKY_NAME_PATTERNS):
                self.dropped[col] = "leakage_suspicious_name"
                self._log("03_leakage", col, "drop",
                          "name hints at post-fact data",
                          "likely target leakage - the data is unavailable at prediction time")
        # near-perfect correlation with the target
        y_num = pd.to_numeric(df[self.target_col], errors="coerce")
        if y_num.notna().mean() > 0.5:
            for col in feature_cols:
                if col in self.dropped:
                    continue
                if self.schema.get(col) not in ("numeric", "numeric_str", "boolean"):
                    continue
                s = pd.to_numeric(df[col], errors="coerce")
                if s.notna().sum() < 10:
                    continue
                corr = s.corr(y_num)
                if corr is not None and not np.isnan(corr) and abs(corr) > LEAKAGE_TARGET_CORR:
                    self.dropped[col] = f"leakage_corr ({corr:.3f})"
                    self._log("03_leakage", col, "drop",
                              f"correlation with target = {corr:.3f}",
                              "near-perfect correlation with the target - almost always leakage")

    # ---- STAGE 3b: multicollinearity ---------------------------------------
    def _detect_multicollinearity(self, df):
        num_cols = [c for c in df.columns
                    if c != self.target_col and c not in self.dropped
                    and self.schema.get(c) in ("numeric", "numeric_str", "boolean")]
        if len(num_cols) < 2:
            return
        corr = df[num_cols].apply(pd.to_numeric, errors="coerce").corr().abs()
        for i, a in enumerate(num_cols):
            if a in self.dropped:
                continue
            for b in num_cols[i + 1:]:
                if b in self.dropped:
                    continue
                val = corr.loc[a, b]
                if pd.notna(val) and val > MULTICOLLINEAR_CORR:
                    self.dropped[b] = f"multicollinear with '{a}' ({val:.3f})"
                    self._log("03b_corr", b, "drop",
                              f"|correlation| with '{a}' = {val:.3f}",
                              "multicollinearity blurs feature importance")

    def _drop_columns(self, df):
        cols = [c for c in self.dropped if c in df.columns]
        return df.drop(columns=cols)

    # ---- STAGE 4: target analysis ------------------------------------------
    def _fit_target(self, df):
        y = df[self.target_col]
        # determine the task type
        if self.task != "auto":
            self.resolved_task = self.task
        else:
            if (not pd.api.types.is_numeric_dtype(y)) or _is_boolean(y):
                self.resolved_task = ("binary" if y.nunique(dropna=True) <= 2
                                      else "multiclass")
            else:
                nun = y.nunique(dropna=True)
                is_int_like = np.allclose(y.dropna() % 1, 0)
                if nun <= CLASSIFICATION_MAX_UNIQUE and is_int_like:
                    self.resolved_task = "binary" if nun <= 2 else "multiclass"
                else:
                    self.resolved_task = "regression"

        if self.resolved_task in ("binary", "multiclass"):
            classes = sorted(y.dropna().unique(), key=lambda v: str(v))
            self.label_mapping = {orig: i for i, orig in enumerate(classes)}
            self.num_class = len(classes)
            self._log("04_target", self.target_col, "label_encode",
                      {str(k): v for k, v in self.label_mapping.items()},
                      "XGBoost requires labels 0..n-1")
            if self.resolved_task == "binary":
                codes = y.map(self.label_mapping)
                n_pos = int((codes == 1).sum())
                n_neg = int((codes == 0).sum())
                if n_pos > 0 and n_neg > 0:
                    ratio = max(n_pos, n_neg) / min(n_pos, n_neg)
                    if ratio > IMBALANCE_RATIO:
                        self.scale_pos_weight = round(n_neg / n_pos, 4)
                        self._log("04_target", self.target_col, "set_scale_pos_weight",
                                  f"imbalance {ratio:.1f}:1 -> scale_pos_weight={self.scale_pos_weight}",
                                  "rebalances the gradients for the rare class")
        else:
            y_num = pd.to_numeric(y, errors="coerce")
            skew = y_num.skew()
            if skew is not None and abs(skew) > TARGET_SKEW_THRESHOLD and (y_num.dropna() > 0).all():
                self.log_transform_target = True
                self._log("04_target", self.target_col, "log1p_transform",
                          f"target skewness = {skew:.2f}",
                          "log1p stabilizes the error for a skewed target")

    def _encode_target_series(self, y):
        """Transform the target: label-encode for classification, log1p for skewed regression."""
        if self.resolved_task in ("binary", "multiclass"):
            codes = y.map(self.label_mapping)
            if codes.isna().any():
                unknown = sorted(set(y.dropna().unique()) - set(self.label_mapping))
                if unknown:
                    print(f"  [WARNING] unknown target classes in the new data: {unknown}")
            return codes.astype("Int64")
        y_num = pd.to_numeric(y, errors="coerce")
        return np.log1p(y_num) if self.log_transform_target else y_num

    # ---- STAGE 5: fit categorical encoders ----------------------------------
    def _fit_categorical_encoders(self, df):
        cat_cols = [c for c in df.columns
                    if c != self.target_col and c not in self.dropped
                    and self.schema.get(c) == "categorical"]
        n = len(df)
        for col in cat_cols:
            s = df[col].astype("object")
            # rare categories -> __other__
            freq = s.value_counts(normalize=True, dropna=True)
            kept = set(freq[freq >= RARE_CATEGORY_SHARE].index)
            self.rare_maps[col] = kept
            grouped = self._apply_rare(s, col)
            n_grouped_unique = grouped.nunique(dropna=False)

            if n_grouped_unique <= ONEHOT_MAX_CARDINALITY:
                cats = sorted(grouped.dropna().astype(str).unique())
                self.onehot_categories[col] = cats
                self.encoding_strategy[col] = "onehot"
                self._log("05_encode", col, "onehot",
                          f"{n_grouped_unique} categories after grouping",
                          "low cardinality -> one-hot, no leakage risk")
            elif self.resolved_task in ("binary", "regression"):
                y = self._encode_target_series(df[self.target_col]).astype(float)
                gmean = float(np.nanmean(y))
                self.target_enc_global[col] = gmean
                self.target_enc_maps[col] = self._smoothed_target_map(
                    grouped.astype(str), y, gmean)
                self.encoding_strategy[col] = "target"
                self._log("05_encode", col, "target_encoding",
                          f"{n_grouped_unique} categories, smoothing={TARGET_ENC_SMOOTHING}",
                          "high cardinality -> target encoding (smoothed, OOF)")
            else:  # multiclass -> frequency encoding (target encoding is incorrect)
                fmap = (grouped.astype(str).value_counts() / max(n, 1)).to_dict()
                self.freq_maps[col] = fmap
                self.encoding_strategy[col] = "frequency"
                self._log("05_encode", col, "frequency_encoding",
                          f"{n_grouped_unique} categories, multiclass task",
                          "target encoding is incorrect for multiclass -> frequency encoding")

    def _apply_rare(self, series, col):
        """Replace NaN with __missing__ and rare values with __other__."""
        s = series.astype("object").where(series.notna(), MISSING_TOKEN)
        kept = self.rare_maps.get(col, set())
        return s.map(lambda v: v if (v == MISSING_TOKEN or v in kept) else OTHER_TOKEN)

    @staticmethod
    def _smoothed_target_map(cat_series, y, global_mean):
        stats = pd.DataFrame({"cat": cat_series.values, "y": np.asarray(y, dtype=float)})
        agg = stats.groupby("cat")["y"].agg(["count", "mean"])
        smooth = ((agg["count"] * agg["mean"] + TARGET_ENC_SMOOTHING * global_mean)
                  / (agg["count"] + TARGET_ENC_SMOOTHING))
        return smooth.to_dict()

    def _compute_oof_target_encoding(self, raw_series, y, col):
        """Out-of-fold target encoding for the training rows (leakage protection)."""
        grouped = self._apply_rare(raw_series, col).astype(str)
        y = pd.Series(np.asarray(y, dtype=float)).reset_index(drop=True)
        grouped = grouped.reset_index(drop=True)
        oof = pd.Series(np.nan, index=grouped.index, dtype=float)
        gmean = self.target_enc_global.get(col, float(np.nanmean(y)))
        kf = KFold(n_splits=min(TARGET_ENC_FOLDS, max(2, len(grouped) // 2)),
                   shuffle=True, random_state=self.random_state)
        for tr_idx, va_idx in kf.split(grouped):
            mask = ~np.isnan(y.iloc[tr_idx].values)
            fold_map = self._smoothed_target_map(
                grouped.iloc[tr_idx][mask], y.iloc[tr_idx][mask], gmean)
            oof.iloc[va_idx] = grouped.iloc[va_idx].map(fold_map).fillna(gmean).values
        return oof.fillna(gmean)

    # ---- STAGE 6: build numeric features -----------------------------------
    def _transform_features(self, df, use_oof=False):
        """Convert all features into numeric columns (excluding the target)."""
        out = pd.DataFrame(index=df.index)

        # is_missing flags for columns dropped due to a high missing ratio
        for col in self.added_flag_cols:
            if col in df.columns:
                out[f"is_missing_{col}"] = df[col].isna().astype(int)

        for col in df.columns:
            if col == self.target_col or col in self.dropped:
                continue
            typ = self.schema.get(col)

            if typ in ("numeric", "numeric_str", "boolean"):
                # XGBoost handles NaN natively; we do NOT scale
                out[col] = pd.to_numeric(df[col], errors="coerce")

            elif typ == "datetime":
                out = pd.concat([out, self._datetime_features(df[col], col)], axis=1)

            elif typ == "categorical":
                strategy = self.encoding_strategy.get(col)
                grouped = self._apply_rare(df[col], col).astype(str)
                if strategy == "onehot":
                    cats = self.onehot_categories[col]
                    cat = pd.Categorical(grouped, categories=cats)
                    dummies = pd.get_dummies(cat, prefix=col, prefix_sep="__").astype(int)
                    dummies.index = df.index
                    out = pd.concat([out, dummies], axis=1)
                elif strategy == "target":
                    gmean = self.target_enc_global[col]
                    out[f"te_{col}"] = grouped.map(self.target_enc_maps[col]).fillna(gmean)
                elif strategy == "frequency":
                    out[f"freq_{col}"] = grouped.map(self.freq_maps[col]).fillna(0.0)
        return out

    @staticmethod
    def _datetime_features(series, col):
        """Decompose a date into numeric features + cyclic sin/cos."""
        dt = pd.to_datetime(series, errors="coerce")
        feat = pd.DataFrame(index=series.index)
        feat[f"{col}_year"] = dt.dt.year
        feat[f"{col}_month"] = dt.dt.month
        feat[f"{col}_day"] = dt.dt.day
        feat[f"{col}_dayofweek"] = dt.dt.dayofweek
        feat[f"{col}_dayofyear"] = dt.dt.dayofyear
        feat[f"{col}_hour"] = dt.dt.hour
        feat[f"{col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype("Int64")
        # cyclic encoding of periodic quantities
        feat[f"{col}_month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
        feat[f"{col}_month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
        feat[f"{col}_dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
        feat[f"{col}_dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
        feat[f"{col}_hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
        feat[f"{col}_hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
        return feat

    # ---- artifacts ----------------------------------------------------------
    def encoders_summary(self):
        """Human-readable description of all encoders and maps."""
        return _json_safe({
            "target_column": self.target_col,
            "resolved_task": self.resolved_task,
            "n_rows_fit": self.n_rows_fit,
            "schema": self.schema,
            "dropped_columns": self.dropped,
            "missing_flag_columns": [f"is_missing_{c}" for c in self.added_flag_cols],
            "datetime_columns": [c for c, t in self.schema.items()
                                 if t == "datetime" and c not in self.dropped],
            "rare_category_kept": {c: sorted(v) for c, v in self.rare_maps.items()},
            "encoding_strategy": self.encoding_strategy,
            "onehot_categories": self.onehot_categories,
            "target_encoding_maps": self.target_enc_maps,
            "target_encoding_global_mean": self.target_enc_global,
            "frequency_maps": self.freq_maps,
            "label_mapping": self.label_mapping,
            "log_transform_target": self.log_transform_target,
            "final_feature_order": self.feature_order,
        })

    def training_config(self):
        """Config for a SageMaker XGBoost training job.

        Contains two run modes:
          * hyperparameters - baseline values for a single training job;
          * automatic_model_tuning - search ranges for SageMaker AMT, adapted
            to the dataset size (build_amt_ranges).
        """
        cfg = {"sagemaker_xgboost_version": SAGEMAKER_XGBOOST_VERSION,
               "hyperparameters": dict(HP_DEFAULTS)}
        if self.resolved_task == "binary":
            cfg["hyperparameters"]["objective"] = "binary:logistic"
            cfg["hyperparameters"]["eval_metric"] = "auc"
            if self.scale_pos_weight is not None:
                cfg["hyperparameters"]["scale_pos_weight"] = self.scale_pos_weight
        elif self.resolved_task == "multiclass":
            cfg["hyperparameters"]["objective"] = "multi:softprob"
            cfg["hyperparameters"]["eval_metric"] = "mlogloss"
            cfg["hyperparameters"]["num_class"] = self.num_class
        else:
            cfg["hyperparameters"]["objective"] = "reg:squarederror"
            cfg["hyperparameters"]["eval_metric"] = "rmse"
        cfg["task"] = self.resolved_task
        cfg["label_mapping"] = self.label_mapping
        cfg["target_log_transform"] = self.log_transform_target
        cfg["target_inverse_transform"] = "expm1" if self.log_transform_target else "none"
        cfg["instance_recommendation"] = {
            "type": "ml.m5.xlarge",
            "note": "XGBoost is memory-bound; use a general-purpose instance, RAM must fit the dataset",
        }
        cfg["channels"] = {"train": "train.csv", "validation": "validation.csv"}
        cfg["csv_format"] = "target=first column, no header, numeric values only"

        # --- ranges for SageMaker Automatic Model Tuning ---
        eval_metric = cfg["hyperparameters"]["eval_metric"]
        amt = build_amt_ranges(
            n_rows=self.n_rows_fit,
            n_features=len(self.feature_order),
            eval_metric=eval_metric,
            imbalanced=self.scale_pos_weight is not None,
        )
        # static_hyperparameters - what is NOT tuned: passed to the estimator as is.
        # num_round is fixed high, the tree count is bounded by early stopping.
        static = {"objective": cfg["hyperparameters"]["objective"],
                  "eval_metric": eval_metric,
                  "num_round": HP_DEFAULTS["num_round"],
                  "early_stopping_rounds": HP_DEFAULTS["early_stopping_rounds"],
                  "verbosity": 1}
        if self.resolved_task == "multiclass":
            static["num_class"] = self.num_class
        if self.scale_pos_weight is not None:
            static["scale_pos_weight"] = self.scale_pos_weight
        amt["static_hyperparameters"] = static
        cfg["automatic_model_tuning"] = amt

        return _json_safe(cfg)


# =============================================================================
# TRAIN / VALIDATION SPLIT (encoders are fitted on the train part only)
# =============================================================================
def _split_train_validation(df, target_col, validation_size, task,
                             time_col, group_col, random_state):
    if validation_size <= 0:
        return df, None, "no_validation_split"
    if time_col and time_col in df.columns:
        ordered = df.sort_values(time_col)
        cut = int(len(ordered) * (1 - validation_size))
        return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy(), "time_based"
    if group_col and group_col in df.columns:
        gss = GroupShuffleSplit(n_splits=1, test_size=validation_size,
                                random_state=random_state)
        tr_idx, va_idx = next(gss.split(df, groups=df[group_col]))
        return df.iloc[tr_idx].copy(), df.iloc[va_idx].copy(), "group_based"
    stratify = df[target_col] if task in ("binary", "multiclass") else None
    try:
        tr, va = train_test_split(df, test_size=validation_size,
                                  random_state=random_state, stratify=stratify)
    except ValueError:
        tr, va = train_test_split(df, test_size=validation_size,
                                  random_state=random_state)
    return tr.copy(), va.copy(), "random_stratified"


def _write_sagemaker_csv(df, path):
    """Write a CSV in SageMaker XGBoost format: target first, no header."""
    df.to_csv(path, index=False, header=False)


# =============================================================================
# FUNCTION 1 - PREPARE TRAINING DATA
# =============================================================================
def prepare_training_data(train_csv, target_col, output_dir,
                          task="auto", time_col=None, group_col=None,
                          validation_size=0.15, random_state=42):
    """Analyze the training CSV, fit transformations, save all artifacts.

    Returns a dict with paths to the created files.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=== PREPARING TRAINING DATA: {train_csv} ===")
    df = pd.read_csv(train_csv)
    print(f"  loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # preliminary task type detection (for stratifying the split)
    pre = XGBoostDataPrep(target_col, task=task)
    pre._detect_schema(df)
    pre._fit_target(pre._apply_type_conversions(df))
    resolved_task = pre.resolved_task

    # split; encoders are fitted on the train part ONLY (leakage protection)
    train_df, val_df, split_method = _split_train_validation(
        df, target_col, validation_size, resolved_task, time_col, group_col, random_state)
    print(f"  split ({split_method}): train={len(train_df)}"
          + (f", validation={len(val_df)}" if val_df is not None else ""))

    pipeline = XGBoostDataPrep(target_col, task=task, time_col=time_col,
                               group_col=group_col, random_state=random_state)
    train_out = pipeline.fit_transform(train_df)

    print(f"  task: {pipeline.resolved_task}")
    print(f"  dropped columns: {len(pipeline.dropped)}")
    print(f"  final features: {len(pipeline.feature_order)}")
    if pipeline.scale_pos_weight is not None:
        print(f"  scale_pos_weight: {pipeline.scale_pos_weight}")

    paths = {}
    # train.csv
    p = os.path.join(output_dir, "train.csv")
    _write_sagemaker_csv(train_out, p); paths["train"] = p
    # validation.csv
    if val_df is not None:
        val_out = pipeline.transform(val_df)
        p = os.path.join(output_dir, "validation.csv")
        _write_sagemaker_csv(val_out, p); paths["validation"] = p
    # pipeline.pkl - the fitted mapper/encoder for prepare_test_data
    p = os.path.join(output_dir, "pipeline.pkl")
    with open(p, "wb") as fh:
        pickle.dump(pipeline, fh)
    paths["pipeline"] = p
    # encoders.json - human-readable description
    p = os.path.join(output_dir, "encoders.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(pipeline.encoders_summary(), fh, ensure_ascii=False, indent=2)
    paths["encoders"] = p
    # training_config.json (baseline hyperparameters + AMT ranges)
    config = pipeline.training_config()
    p = os.path.join(output_dir, "training_config.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)
    paths["training_config"] = p
    amt = config.get("automatic_model_tuning", {})
    if amt:
        print(f"  AMT: {amt['dataset_basis']['notes']}")
    # feature_schema.json - column order (for inference)
    p = os.path.join(output_dir, "feature_schema.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(_json_safe({"target": target_col,
                              "feature_order": pipeline.feature_order}),
                  fh, ensure_ascii=False, indent=2)
    paths["feature_schema"] = p
    # decision_log.csv - audit trail
    p = os.path.join(output_dir, "decision_log.csv")
    pd.DataFrame(pipeline.decisions).to_csv(p, index=False)
    paths["decision_log"] = p

    print(f"  artifacts saved to: {output_dir}")
    return paths


# =============================================================================
# FUNCTION 2 - PREPARE TEST DATA (using the transformations from train)
# =============================================================================
def prepare_test_data(test_csv, pipeline_pkl, output_dir,
                       output_name="test.csv"):
    """Apply the transformations fitted on the training data to a test CSV.

    Nothing is recomputed: it loads pipeline.pkl from prepare_training_data.
    Returns the path to the prepared test.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=== PREPARING TEST DATA: {test_csv} ===")
    with open(pipeline_pkl, "rb") as fh:
        pipeline = pickle.load(fh)
    if not isinstance(pipeline, XGBoostDataPrep):
        raise TypeError("pipeline_pkl does not contain an XGBoostDataPrep object.")

    df = pd.read_csv(test_csv)
    print(f"  loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    has_target = pipeline.target_col in df.columns

    test_out = pipeline.transform(df)
    out_path = os.path.join(output_dir, output_name)
    _write_sagemaker_csv(test_out, out_path)

    print(f"  target in data: {'yes' if has_target else 'no (inference mode)'}")
    print(f"  output columns: {test_out.shape[1]}")
    print(f"  saved: {out_path}")
    return out_path


# =============================================================================
# CLI
# =============================================================================
def _build_cli():
    parser = argparse.ArgumentParser(
        description="Data preparation for XGBoost on SageMaker.")
    sub = parser.add_subparsers(dest="mode", required=True)

    pt = sub.add_parser("train", help="prepare training data")
    pt.add_argument("csv")
    pt.add_argument("--target", required=True)
    pt.add_argument("--output-dir", default="./out")
    pt.add_argument("--task", default="auto",
                    choices=["auto", "binary", "multiclass", "regression"])
    pt.add_argument("--time-col", default=None)
    pt.add_argument("--group-col", default=None)
    pt.add_argument("--validation-size", type=float, default=0.15)
    pt.add_argument("--random-state", type=int, default=42)

    ps = sub.add_parser("test", help="prepare test data")
    ps.add_argument("csv")
    ps.add_argument("--pipeline", required=True)
    ps.add_argument("--output-dir", default="./out")
    ps.add_argument("--output-name", default="test.csv")
    return parser


def main(argv=None):
    args = _build_cli().parse_args(argv)
    if args.mode == "train":
        prepare_training_data(
            args.csv, args.target, args.output_dir, task=args.task,
            time_col=args.time_col, group_col=args.group_col,
            validation_size=args.validation_size, random_state=args.random_state)
    else:
        prepare_test_data(args.csv, args.pipeline, args.output_dir,
                          output_name=args.output_name)


if __name__ == "__main__":
    main()
