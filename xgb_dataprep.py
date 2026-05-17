#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
АВТОМАТИЗОВАНА ПІДГОТОВКА ДАНИХ ДЛЯ XGBOOST У SAGEMAKER
=============================================================================
Усі правила захардкоджені в коді (див. блок констант нижче). YAML не читається.

ДВІ ГОЛОВНІ ФУНКЦІЇ:

  prepare_training_data(train_csv, target_col, output_dir, ...)
      Аналізує тренувальні дані, фітить усі трансформації (типи, дропи,
      енкодери, target encoding, обробку дисбалансу) і зберігає:
        * train.csv, validation.csv  — у форматі SageMaker XGBoost
        * pipeline.pkl               — фітнутий маппер/енкодер для повторного використання
        * encoders.json              — людиночитабельний опис енкодерів та мап
        * training_config.json       — objective, scale_pos_weight, гіперпараметри
        * feature_schema.json        — порядок фінальних колонок (для inference)
        * decision_log.csv           — що зроблено і чому

  prepare_test_data(test_csv, pipeline_pkl, output_dir, ...)
      Бере pipeline.pkl, отриманий на тренувальних даних, і застосовує
      ТІ САМІ трансформації до тестових даних. Нічого не переобчислює.
      Зберігає test.csv у форматі SageMaker XGBoost.

ФОРМАТ ВИХІДНОГО CSV (вимога SageMaker built-in XGBoost):
  таргет = ПЕРША колонка, БЕЗ заголовка, тільки числові значення, NaN = порожньо.

CLI:
  python xgb_dataprep.py train data.csv --target churned --output-dir ./out
  python xgb_dataprep.py test  test.csv --pipeline ./out/pipeline.pkl --output-dir ./out

Залежності: pandas, numpy, scikit-learn
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
# ПРАВИЛА ТА ПОРОГИ (захардкоджено). Змінюйте тут — це єдине джерело правди.
# =============================================================================
HIGH_CARDINALITY_RATIO   = 0.70   # >70% унікальних у категорії -> колонка-ідентифікатор
HIGH_CARDINALITY_ABS_CAP = 10000  # або абсолютний поріг кількості унікальних
QUASI_CONSTANT_SHARE     = 0.99   # одне значення займає >99% рядків -> дропаємо
HIGH_MISSING_RATIO       = 0.70   # >70% пропусків -> дропаємо (зі збереженням is_missing-флага)
ZERO_VARIANCE_STD        = 1e-8   # std нижче -> числова константа
LEAKAGE_TARGET_CORR      = 0.98   # |кореляція з таргетом| вище -> підозра на витік, дроп
MULTICOLLINEAR_CORR      = 0.97   # |кореляція| між фічами вище -> дроп однієї з пари
ONEHOT_MAX_CARDINALITY   = 15     # <=15 унікальних -> one-hot, інакше target/frequency encoding
RARE_CATEGORY_SHARE      = 0.01   # категорія з частотою <1% -> об'єднати в '__other__'
TARGET_SKEW_THRESHOLD    = 1.0    # |skew| таргета вище (регресія, >0) -> log1p-трансформація
IMBALANCE_RATIO          = 5.0    # n_major/n_minor вище -> рахуємо scale_pos_weight
CLASSIFICATION_MAX_UNIQUE = 20    # <=20 унікальних числового таргета -> класифікація (для task='auto')
TARGET_ENC_SMOOTHING     = 10.0   # згладжування target encoding
TARGET_ENC_FOLDS         = 5      # к-сть фолдів для out-of-fold target encoding

MISSING_TOKEN = "__missing__"
OTHER_TOKEN   = "__other__"

# Назви-патерни для автоматичного виявлення проблемних колонок
ID_NAME_PATTERNS    = ["id", "uuid", "guid", "index", "row_number", "rownum", "hash", "key"]
LEAKY_NAME_PATTERNS = ["outcome", "result", "final", "is_fraud", "churned",
                       "closed", "post_", "after_", "_leak"]
BOOLEAN_TOKENS = {"true", "false", "yes", "no", "t", "f", "y", "n", "0", "1"}

# Дефолтні гіперпараметри XGBoost (baseline до тюнінгу)
HP_DEFAULTS = {
    "num_round": 1000, "early_stopping_rounds": 30, "eta": 0.1, "max_depth": 6,
    "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8,
    "gamma": 0, "lambda": 1, "alpha": 0, "verbosity": 1,
}
SAGEMAKER_XGBOOST_VERSION = "1.7-1"


# =============================================================================
# ДОПОМІЖНІ ФУНКЦІЇ ВИЗНАЧЕННЯ ТА КОНВЕРТАЦІЇ ТИПІВ
# =============================================================================
def _try_numeric(series):
    """Конвертує рядкову колонку в число: знімає коми, символи валют, пробіли."""
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
    """True, якщо рядкова колонка впевнено парситься як дата у діапазоні 1900-2100."""
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
    """Робить структуру придатною для json.dump (set->list, numpy->python)."""
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
# ОСНОВНИЙ КЛАС: ФІТИТЬСЯ НА TRAIN, ЗАСТОСОВУЄТЬСЯ ДО TRAIN/VAL/TEST
# =============================================================================
class XGBoostDataPrep:
    """Аналізує дані, фітить усі трансформації та застосовує їх.

    fit(df)            — навчити трансформації на тренувальних даних.
    fit_transform(df)  — навчити + повернути готовий train (з out-of-fold target encoding).
    transform(df)      — застосувати готові трансформації до val/test.
    """

    def __init__(self, target_col, task="auto", time_col=None,
                 group_col=None, random_state=42):
        self.target_col = target_col
        self.task = task                 # 'auto' | 'binary' | 'multiclass' | 'regression'
        self.time_col = time_col
        self.group_col = group_col
        self.random_state = random_state

        # --- стан, що фітиться (це і є "маппер/енкодер") ---
        self.schema = {}                 # колонка -> тип
        self.dropped = {}                # колонка -> причина дропу
        self.added_flag_cols = []        # створені is_missing_* колонки
        self.datetime_cols = []          # datetime-колонки, що залишились (для FE)
        self.rare_maps = {}              # колонка -> set збережених категорій
        self.encoding_strategy = {}      # колонка -> 'onehot' | 'target' | 'frequency'
        self.onehot_categories = {}      # колонка -> впорядкований список категорій
        self.target_enc_maps = {}        # колонка -> {категорія: число}
        self.target_enc_global = {}      # колонка -> глобальне середнє таргета
        self.freq_maps = {}              # колонка -> {категорія: частота}
        self.feature_order = []          # фінальний порядок числових фіч
        # --- таргет ---
        self.resolved_task = None
        self.label_mapping = None        # класифікація: оригінальне значення -> 0..n-1
        self.num_class = None
        self.log_transform_target = False
        self.scale_pos_weight = None
        # --- службове ---
        self.decisions = []              # лог рішень
        self.n_rows_fit = 0
        self.fitted = False

    # ---- логування рішень ---------------------------------------------------
    def _log(self, stage, column, action, detail, reason):
        self.decisions.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage, "column": column, "action": action,
            "detail": detail, "reason": reason,
        })

    # =========================================================================
    # FIT — навчання трансформацій на тренувальних даних
    # =========================================================================
    def fit(self, df):
        if self.target_col not in df.columns:
            raise ValueError(f"Колонки-таргета '{self.target_col}' немає в датасеті.")
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

        # встановлюємо фінальний порядок колонок, зробивши пробну трансформацію
        sample_out = self._transform_features(df, use_oof=False)
        self.feature_order = list(sample_out.columns)
        self.fitted = True
        return self

    def fit_transform(self, df):
        """Фіт + готовий train. Target-encoded колонки рахуються out-of-fold."""
        self.fit(df)
        out = self.transform(df)
        # перерахунок target-encoded колонок через out-of-fold (без витоку таргета)
        oof_cols = [c for c, s in self.encoding_strategy.items() if s == "target"]
        if oof_cols:
            work = self._apply_type_conversions(df.copy())
            y = self._encode_target_series(work[self.target_col])
            for col in oof_cols:
                oof = self._compute_oof_target_encoding(work[col], y, col)
                out[f"te_{col}"] = oof.values
        return out

    # =========================================================================
    # TRANSFORM — застосування готових трансформацій (val / test)
    # =========================================================================
    def transform(self, df):
        if not self.fitted:
            raise RuntimeError("Спочатку викличте fit() або fit_transform().")
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
            # inference-режим: таргета немає, повертаємо лише фічі
            out = features.reset_index(drop=True)
        return out

    # ---- STAGE 1: визначення типів -----------------------------------------
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
                  "класифікація типів потрібна для вибору обробки")

    def _apply_type_conversions(self, df, fitting=True):
        df = df.copy()
        for col, typ in self.schema.items():
            if col not in df.columns:
                continue
            if typ == "numeric_str":
                df[col] = _try_numeric(df[col])
                if fitting:
                    self._log("01_schema", col, "convert_numeric",
                              "число збережене текстом",
                              "інакше було б закодоване як категорія")
            elif typ == "boolean":
                df[col] = _to_boolean(df[col])
                if fitting:
                    self._log("01_schema", col, "convert_boolean",
                              "2 булеві значення", "бінарна ознака має бути числовою")
            elif typ == "datetime":
                df[col] = _try_datetime(df[col])
                if fitting:
                    self._log("01_schema", col, "parse_datetime",
                              "колонка-дата", "потрібна декомпозиція в числові ознаки")
            elif typ == "numeric":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    # ---- STAGE 2: дропи нерепрезентативних колонок --------------------------
    def _decide_drops(self, df):
        n = len(df)
        feature_cols = [c for c in df.columns if c != self.target_col]

        for col in feature_cols:
            s = df[col]
            # константа
            if s.nunique(dropna=False) <= 1:
                self.dropped[col] = "constant_column"
                self._log("02_drop", col, "drop", "1 унікальне значення",
                          "нульова дисперсія — немає сигналу для дерев")
                continue
            # квазі-константа
            top_share = s.value_counts(normalize=True, dropna=False).iloc[0]
            if top_share > QUASI_CONSTANT_SHARE:
                self.dropped[col] = f"quasi_constant ({top_share:.1%})"
                self._log("02_drop", col, "drop", f"{top_share:.1%} одного значення",
                          "майже-константа майже не несе інформації")
                continue
            # багато пропусків
            miss = s.isna().mean()
            if miss > HIGH_MISSING_RATIO:
                self.dropped[col] = f"high_missing ({miss:.1%})"
                self.added_flag_cols.append(col)   # натомість збережемо is_missing-флаг
                self._log("02_drop", col, "drop+flag", f"{miss:.1%} пропусків",
                          "колонка рідко корисна, але факт пропуску зберігаємо як is_missing_*")
                continue
            # нульова дисперсія числових
            if self.schema.get(col) in ("numeric", "numeric_str", "boolean"):
                std = pd.to_numeric(s, errors="coerce").std(skipna=True)
                if std is not None and not np.isnan(std) and std < ZERO_VARIANCE_STD:
                    self.dropped[col] = "zero_variance_numeric"
                    self._log("02_drop", col, "drop", "std ~ 0",
                              "числова константа марна для сплітів")
                    continue
            # висока кардинальність категоріальних -> ідентифікатор
            if self.schema.get(col) == "categorical":
                nun = s.nunique(dropna=True)
                ratio = nun / n if n else 0
                if ratio > HIGH_CARDINALITY_RATIO or nun > HIGH_CARDINALITY_ABS_CAP:
                    self.dropped[col] = f"high_cardinality ({ratio:.1%}, {nun} унік.)"
                    self._log("02_drop", col, "drop",
                              f"{ratio:.1%} унікальних значень ({nun})",
                              "майже-унікальні значення = ідентифікатор, не ознака")
                    continue
            # ID за назвою колонки
            low = col.lower()
            if any(p == low or low.endswith("_" + p) or low.startswith(p + "_")
                   for p in ID_NAME_PATTERNS):
                self.dropped[col] = "identifier_by_name"
                self._log("02_drop", col, "drop", "назва схожа на ідентифікатор",
                          "технічні ID не мають предиктивної цінності")

        # дублікати колонок
        seen = {}
        for col in feature_cols:
            if col in self.dropped:
                continue
            key = tuple(pd.util.hash_pandas_object(
                df[col].astype("object").fillna("__na__"), index=False).values[:5000])
            if key in seen:
                self.dropped[col] = f"duplicate_of '{seen[key]}'"
                self._log("02_drop", col, "drop", f"дублікат '{seen[key]}'",
                          "дублікати спотворюють feature importance")
            else:
                seen[key] = col

    # ---- STAGE 3: target leakage -------------------------------------------
    def _detect_leakage(self, df):
        feature_cols = [c for c in df.columns
                        if c != self.target_col and c not in self.dropped]
        # підозрілі назви
        for col in feature_cols:
            low = col.lower()
            if any(p in low for p in LEAKY_NAME_PATTERNS):
                self.dropped[col] = "leakage_suspicious_name"
                self._log("03_leakage", col, "drop",
                          "назва натякає на пост-фактум дані",
                          "ймовірний витік таргета — даних не буде на момент прогнозу")
        # майже ідеальна кореляція з таргетом
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
                              f"кореляція з таргетом = {corr:.3f}",
                              "майже ідеальна кореляція з таргетом — майже завжди витік")

    # ---- STAGE 3b: мультиколінеарність -------------------------------------
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
                              f"|кореляція| з '{a}' = {val:.3f}",
                              "мультиколінеарність розмиває feature importance")

    def _drop_columns(self, df):
        cols = [c for c in self.dropped if c in df.columns]
        return df.drop(columns=cols)

    # ---- STAGE 4: аналіз таргета -------------------------------------------
    def _fit_target(self, df):
        y = df[self.target_col]
        # визначення типу задачі
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
                      "XGBoost вимагає мітки 0..n-1")
            if self.resolved_task == "binary":
                codes = y.map(self.label_mapping)
                n_pos = int((codes == 1).sum())
                n_neg = int((codes == 0).sum())
                if n_pos > 0 and n_neg > 0:
                    ratio = max(n_pos, n_neg) / min(n_pos, n_neg)
                    if ratio > IMBALANCE_RATIO:
                        self.scale_pos_weight = round(n_neg / n_pos, 4)
                        self._log("04_target", self.target_col, "set_scale_pos_weight",
                                  f"дисбаланс {ratio:.1f}:1 -> scale_pos_weight={self.scale_pos_weight}",
                                  "перебалансовує градієнти для рідкісного класу")
        else:
            y_num = pd.to_numeric(y, errors="coerce")
            skew = y_num.skew()
            if skew is not None and abs(skew) > TARGET_SKEW_THRESHOLD and (y_num.dropna() > 0).all():
                self.log_transform_target = True
                self._log("04_target", self.target_col, "log1p_transform",
                          f"скошеність таргета = {skew:.2f}",
                          "log1p стабілізує помилку для скошеного таргета")

    def _encode_target_series(self, y):
        """Перетворює таргет: label-encode для класифікації, log1p для скошеної регресії."""
        if self.resolved_task in ("binary", "multiclass"):
            codes = y.map(self.label_mapping)
            if codes.isna().any():
                unknown = sorted(set(y.dropna().unique()) - set(self.label_mapping))
                if unknown:
                    print(f"  [УВАГА] невідомі класи таргета в нових даних: {unknown}")
            return codes.astype("Int64")
        y_num = pd.to_numeric(y, errors="coerce")
        return np.log1p(y_num) if self.log_transform_target else y_num

    # ---- STAGE 5: фіт енкодерів категоріальних ------------------------------
    def _fit_categorical_encoders(self, df):
        cat_cols = [c for c in df.columns
                    if c != self.target_col and c not in self.dropped
                    and self.schema.get(c) == "categorical"]
        n = len(df)
        for col in cat_cols:
            s = df[col].astype("object")
            # рідкісні категорії -> __other__
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
                          f"{n_grouped_unique} категорій після групування",
                          "низька кардинальність -> one-hot без ризику витоку")
            elif self.resolved_task in ("binary", "regression"):
                y = self._encode_target_series(df[self.target_col]).astype(float)
                gmean = float(np.nanmean(y))
                self.target_enc_global[col] = gmean
                self.target_enc_maps[col] = self._smoothed_target_map(
                    grouped.astype(str), y, gmean)
                self.encoding_strategy[col] = "target"
                self._log("05_encode", col, "target_encoding",
                          f"{n_grouped_unique} категорій, smoothing={TARGET_ENC_SMOOTHING}",
                          "висока кардинальність -> target encoding (зі згладжуванням, OOF)")
            else:  # multiclass -> frequency encoding (target encoding некоректний)
                fmap = (grouped.astype(str).value_counts() / max(n, 1)).to_dict()
                self.freq_maps[col] = fmap
                self.encoding_strategy[col] = "frequency"
                self._log("05_encode", col, "frequency_encoding",
                          f"{n_grouped_unique} категорій, multiclass-задача",
                          "для multiclass target encoding некоректний -> частотне кодування")

    def _apply_rare(self, series, col):
        """Замінює NaN на __missing__, рідкісні значення на __other__."""
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
        """Out-of-fold target encoding для тренувальних рядків (захист від витоку)."""
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

    # ---- STAGE 6: побудова числових фіч ------------------------------------
    def _transform_features(self, df, use_oof=False):
        """Перетворює всі фічі в числові колонки (без таргета)."""
        out = pd.DataFrame(index=df.index)

        # is_missing-флаги для дропнутих через велику кількість пропусків
        for col in self.added_flag_cols:
            if col in df.columns:
                out[f"is_missing_{col}"] = df[col].isna().astype(int)

        for col in df.columns:
            if col == self.target_col or col in self.dropped:
                continue
            typ = self.schema.get(col)

            if typ in ("numeric", "numeric_str", "boolean"):
                # XGBoost обробляє NaN нативно; масштабування НЕ робимо
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
        """Розкладає дату на числові ознаки + циклічні sin/cos."""
        dt = pd.to_datetime(series, errors="coerce")
        feat = pd.DataFrame(index=series.index)
        feat[f"{col}_year"] = dt.dt.year
        feat[f"{col}_month"] = dt.dt.month
        feat[f"{col}_day"] = dt.dt.day
        feat[f"{col}_dayofweek"] = dt.dt.dayofweek
        feat[f"{col}_dayofyear"] = dt.dt.dayofyear
        feat[f"{col}_hour"] = dt.dt.hour
        feat[f"{col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype("Int64")
        # циклічне кодування періодичних величин
        feat[f"{col}_month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
        feat[f"{col}_month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
        feat[f"{col}_dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
        feat[f"{col}_dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
        feat[f"{col}_hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
        feat[f"{col}_hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
        return feat

    # ---- артефакти ----------------------------------------------------------
    def encoders_summary(self):
        """Людиночитабельний опис усіх енкодерів та мап."""
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
        """Конфіг для тренувальної job SageMaker XGBoost."""
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
            "note": "XGBoost memory-bound; брати general-purpose інстанс, RAM має вміщати датасет",
        }
        cfg["channels"] = {"train": "train.csv", "validation": "validation.csv"}
        cfg["csv_format"] = "таргет=перша колонка, без заголовка, тільки числові значення"
        return _json_safe(cfg)


# =============================================================================
# СПЛІТ TRAIN / VALIDATION (фіт енкодерів — лише на train-частині)
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
    """Записує CSV у форматі SageMaker XGBoost: таргет перший, без заголовка."""
    df.to_csv(path, index=False, header=False)


# =============================================================================
# ФУНКЦІЯ 1 — ПІДГОТОВКА ТРЕНУВАЛЬНИХ ДАНИХ
# =============================================================================
def prepare_training_data(train_csv, target_col, output_dir,
                          task="auto", time_col=None, group_col=None,
                          validation_size=0.15, random_state=42):
    """Аналізує тренувальний CSV, фітить трансформації, зберігає всі артефакти.

    Повертає dict зі шляхами до створених файлів.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=== ПІДГОТОВКА ТРЕНУВАЛЬНИХ ДАНИХ: {train_csv} ===")
    df = pd.read_csv(train_csv)
    print(f"  завантажено: {df.shape[0]} рядків, {df.shape[1]} колонок")

    # попереднє визначення типу задачі (для стратифікації спліту)
    pre = XGBoostDataPrep(target_col, task=task)
    pre._detect_schema(df)
    pre._fit_target(pre._apply_type_conversions(df))
    resolved_task = pre.resolved_task

    # спліт; енкодери фітяться ЛИШЕ на train-частині (захист від витоку)
    train_df, val_df, split_method = _split_train_validation(
        df, target_col, validation_size, resolved_task, time_col, group_col, random_state)
    print(f"  спліт ({split_method}): train={len(train_df)}"
          + (f", validation={len(val_df)}" if val_df is not None else ""))

    pipeline = XGBoostDataPrep(target_col, task=task, time_col=time_col,
                               group_col=group_col, random_state=random_state)
    train_out = pipeline.fit_transform(train_df)

    print(f"  задача: {pipeline.resolved_task}")
    print(f"  видалено колонок: {len(pipeline.dropped)}")
    print(f"  фінальних фіч: {len(pipeline.feature_order)}")
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
    # pipeline.pkl — фітнутий маппер/енкодер для prepare_test_data
    p = os.path.join(output_dir, "pipeline.pkl")
    with open(p, "wb") as fh:
        pickle.dump(pipeline, fh)
    paths["pipeline"] = p
    # encoders.json — людиночитабельний опис
    p = os.path.join(output_dir, "encoders.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(pipeline.encoders_summary(), fh, ensure_ascii=False, indent=2)
    paths["encoders"] = p
    # training_config.json
    p = os.path.join(output_dir, "training_config.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(pipeline.training_config(), fh, ensure_ascii=False, indent=2)
    paths["training_config"] = p
    # feature_schema.json — порядок колонок (для inference)
    p = os.path.join(output_dir, "feature_schema.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(_json_safe({"target": target_col,
                              "feature_order": pipeline.feature_order}),
                  fh, ensure_ascii=False, indent=2)
    paths["feature_schema"] = p
    # decision_log.csv — аудит-трейл
    p = os.path.join(output_dir, "decision_log.csv")
    pd.DataFrame(pipeline.decisions).to_csv(p, index=False)
    paths["decision_log"] = p

    print(f"  артефакти збережено у: {output_dir}")
    return paths


# =============================================================================
# ФУНКЦІЯ 2 — ПІДГОТОВКА ТЕСТОВИХ ДАНИХ (на основі трансформацій з train)
# =============================================================================
def prepare_test_data(test_csv, pipeline_pkl, output_dir,
                       output_name="test.csv"):
    """Застосовує до тестового CSV трансформації, фітнуті на тренувальних даних.

    Нічого не переобчислює: бере pipeline.pkl з prepare_training_data.
    Повертає шлях до підготовленого test.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=== ПІДГОТОВКА ТЕСТОВИХ ДАНИХ: {test_csv} ===")
    with open(pipeline_pkl, "rb") as fh:
        pipeline = pickle.load(fh)
    if not isinstance(pipeline, XGBoostDataPrep):
        raise TypeError("pipeline_pkl не містить об'єкта XGBoostDataPrep.")

    df = pd.read_csv(test_csv)
    print(f"  завантажено: {df.shape[0]} рядків, {df.shape[1]} колонок")
    has_target = pipeline.target_col in df.columns

    test_out = pipeline.transform(df)
    out_path = os.path.join(output_dir, output_name)
    _write_sagemaker_csv(test_out, out_path)

    print(f"  таргет у даних: {'так' if has_target else 'ні (inference-режим)'}")
    print(f"  колонок на виході: {test_out.shape[1]}")
    print(f"  збережено: {out_path}")
    return out_path


# =============================================================================
# CLI
# =============================================================================
def _build_cli():
    parser = argparse.ArgumentParser(
        description="Підготовка даних для XGBoost у SageMaker.")
    sub = parser.add_subparsers(dest="mode", required=True)

    pt = sub.add_parser("train", help="підготувати тренувальні дані")
    pt.add_argument("csv")
    pt.add_argument("--target", required=True)
    pt.add_argument("--output-dir", default="./out")
    pt.add_argument("--task", default="auto",
                    choices=["auto", "binary", "multiclass", "regression"])
    pt.add_argument("--time-col", default=None)
    pt.add_argument("--group-col", default=None)
    pt.add_argument("--validation-size", type=float, default=0.15)
    pt.add_argument("--random-state", type=int, default=42)

    ps = sub.add_parser("test", help="підготувати тестові дані")
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
