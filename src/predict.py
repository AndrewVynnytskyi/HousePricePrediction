"""Generate a Kaggle-submission-ready CSV of predictions for data/test.csv.

    python -m src.predict --config configs/train.yaml --output outputs/submission.csv

Reuses the model and preprocessor persisted by `src.train` (models/*.joblib)
rather than refitting anything, so the submission reflects exactly the
model documented in the README.

data/test.csv has missingness the training-time imputation rules don't
cover (Kaggle-known quirks: MSZoning, KitchenQual, SaleType, etc. are
missing only in the test split, never in train). Those rules are still
applied first; any NaNs left over are filled with train-set statistics
(mode for categoricals, median for numerics) -- inference must produce a
prediction for every row, so no row is ever dropped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.impute import KNNImputer

from src.data import drop_low_value_columns, load_raw_data


def _impute_test_data(df: pd.DataFrame, train_df: pd.DataFrame, cfg) -> pd.DataFrame:
    impute_cfg = cfg.data.impute
    df = df.copy()

    df_numeric = df.select_dtypes(include=["int64", "float64"])
    imputer = KNNImputer(n_neighbors=impute_cfg.knn_n_neighbors, weights=impute_cfg.knn_weights)
    f_imputed = imputer.fit_transform(df_numeric)
    lot_col = impute_cfg.knn_lotfrontage_column
    df[lot_col] = f_imputed[:, df_numeric.columns.get_loc(lot_col)]

    df["BsmtExposure"] = df["BsmtExposure"].fillna(impute_cfg.bsmt_exposure_fill_value)

    cat_cols = list(impute_cfg.categorical_fill_none_columns)
    df[cat_cols] = df[cat_cols].fillna("None")

    num_cols = list(impute_cfg.numeric_fill_zero_columns)
    df[num_cols] = df[num_cols].fillna(0)

    # Leftover NaNs are test-set-only quirks not covered by the rules
    # above (they never occur in train, so the training pipeline never
    # needed to handle them). Fill from train statistics; never drop rows.
    for col in df.columns[df.isna().any()]:
        if df[col].dtype == object:
            df[col] = df[col].fillna(train_df[col].mode().iloc[0])
        else:
            df[col] = df[col].fillna(train_df[col].median())

    return df


def generate_submission(cfg, models_dir: Path, output_path: Path) -> pd.DataFrame:
    test_df = load_raw_data(cfg.data.test_path, cfg.data.id_column)
    ids = pd.read_csv(cfg.data.test_path)[cfg.data.id_column]

    train_df = load_raw_data(cfg.data.train_path, cfg.data.id_column)

    test_df = _impute_test_data(test_df, train_df, cfg)
    test_df = drop_low_value_columns(test_df, cfg.data.dropped_columns)

    preprocessor = joblib.load(models_dir / "preprocessor.joblib")
    model = joblib.load(models_dir / "best_model.joblib")

    X_test_arr = preprocessor.transform(test_df)
    X_test = pd.DataFrame(
        X_test_arr, columns=preprocessor.get_feature_names_out(), index=test_df.index
    )
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    submission = pd.DataFrame({cfg.data.id_column: ids, cfg.data.target_column: y_pred})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--output", default="outputs/submission.csv")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    models_dir = Path(args.models_dir or cfg.output.models_dir)

    submission = generate_submission(cfg, models_dir, Path(args.output))
    print(f"Wrote {len(submission)} predictions to {args.output}")
    print(submission.head())


if __name__ == "__main__":
    main()
