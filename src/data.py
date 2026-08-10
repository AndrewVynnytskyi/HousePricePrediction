"""Data loading, cleaning, and preprocessing for the Ames Housing dataset.

Every function here mirrors a specific step of the original exploratory
notebook (notebooks/archive/sample_original.ipynb) so that the transformed
feature matrix and train/test split are numerically identical to what the
notebook produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_raw_data(path: Union[str, Path], id_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.drop(columns=[id_column])


def apply_log_target(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    df = df.copy()
    df[target_column] = np.log1p(df[target_column])
    return df


def impute_missing_values(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Replicates the notebook's imputation order exactly.

    KNNImputer is fit on the full numeric frame (including the already
    log1p-transformed target column) before LotFrontage is filled in. This
    is a leakage pattern inherited from the original notebook and kept
    intentionally so results reproduce; see README "Known limitations".
    """
    impute_cfg = cfg.data.impute
    df = df.copy()

    df_numeric = df.select_dtypes(include=["int64", "float64"])
    imputer = KNNImputer(n_neighbors=impute_cfg.knn_n_neighbors, weights=impute_cfg.knn_weights)
    f_imputed = imputer.fit_transform(df_numeric)

    lot_col = impute_cfg.knn_lotfrontage_column
    df[lot_col] = f_imputed[:, df_numeric.columns.get_loc(lot_col)]

    df["BsmtExposure"] = df["BsmtExposure"].fillna(impute_cfg.bsmt_exposure_fill_value)
    df = df.dropna(subset=list(impute_cfg.dropna_columns))

    cat_cols = list(impute_cfg.categorical_fill_none_columns)
    df[cat_cols] = df[cat_cols].fillna("None")

    num_cols = list(impute_cfg.numeric_fill_zero_columns)
    df[num_cols] = df[num_cols].fillna(0)

    return df


def drop_low_value_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return df.drop(columns=list(columns))


def split_features_target(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def train_test_split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int, shuffle: bool
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)


def build_preprocessor(num_columns: Sequence[str], cat_columns: Sequence[str]) -> ColumnTransformer:
    """StandardScaler on numeric columns, then OneHotEncoder on categorical
    columns. Transformer order matches the notebook's
    pd.concat([num, cat], axis=1) column ordering exactly.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(num_columns)),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(cat_columns)),
        ],
        verbose_feature_names_out=False,
    )


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: List[str]
    preprocessor: ColumnTransformer


def prepare_dataset(cfg: DictConfig) -> PreparedData:
    df = load_raw_data(cfg.data.train_path, cfg.data.id_column)

    if cfg.data.log_transform_target:
        df = apply_log_target(df, cfg.data.target_column)

    df = impute_missing_values(df, cfg)
    df = drop_low_value_columns(df, cfg.data.dropped_columns)

    X, y = split_features_target(df, cfg.data.target_column)
    cat_columns = X.select_dtypes(include=["object"]).columns.tolist()
    num_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split_data(
        X,
        y,
        test_size=cfg.data.split.test_size,
        random_state=cfg.data.split.random_state,
        shuffle=cfg.data.split.shuffle,
    )

    preprocessor = build_preprocessor(num_columns, cat_columns)
    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr = preprocessor.transform(X_test)

    feature_names = list(preprocessor.get_feature_names_out())
    X_train_df = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_arr, columns=feature_names, index=X_test.index)

    return PreparedData(
        X_train=X_train_df,
        X_test=X_test_df,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        preprocessor=preprocessor,
    )
