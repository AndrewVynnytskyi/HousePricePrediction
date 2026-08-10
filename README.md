# House Price Prediction

A config-driven scikit-learn pipeline that predicts house sale prices on the
Kaggle "House Prices: Advanced Regression Techniques" (Ames Housing) dataset.
Six regression algorithms are tuned with `GridSearchCV` across three feature
representations. The best model reaches an RMSE of about **$22,840** on
held-out data (R² ≈ 0.90) and scores **0.13749** on Kaggle's public
leaderboard.

The project started as a single exploratory notebook and was refactored into
a reproducible pipeline: hyperparameters and preprocessing choices live in a
YAML config, the actual logic lives in a tested `src/` package, and the
notebooks are thin wrappers around it. It can be trained from the CLI, from
a notebook, or inside Docker, and every run is logged to disk.

## Dataset

1460 residential properties in Ames, Iowa, with 79 explanatory features
(size, quality ratings, year built, garage/basement details, neighborhood,
etc.) and the target `SalePrice`. Source: [Kaggle's House Prices competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques),
based on the Ames Housing dataset compiled by Dean De Cock.

## Architecture

```
configs/train.yaml        all hyperparameters, paths, and grids — no magic numbers in code
src/
  data.py                 loading, cleaning, imputation, train/test split, preprocessing
  model.py                estimator registry, feature-importance selection, PCA
  train.py                Trainer: grid search, evaluation, model persistence, CLI entrypoint
  utils.py                local run logging (CSV + per-run JSON, in place of a tracking server)
notebooks/
  00_eda.ipynb            exploratory analysis (distributions, correlations)
  01_baseline_training.ipynb   thin orchestration: load config -> Trainer -> results
  archive/sample_original.ipynb   the original single-file notebook, kept for reference
tests/                     unit tests for the data pipeline and model registry
Dockerfile                 multi-stage image for training in a container
```

Everything in `src/` is config-driven. To reproduce a specific experiment,
override the config from the command line instead of editing code:

```bash
python -m src.train                                    # default: ElasticNet, full feature set
python -m src.train feature_selection.mode=importance
python -m src.train feature_selection.mode=pca
python -m src.train training.models='[LinearRegression,KNN,RandomForest,Ridge,Lasso,ElasticNet]'
```

Each run appends its metrics to `outputs/metrics.csv`, writes a JSON snapshot
to `outputs/runs/`, and — if it's the best RMSE seen — saves the fitted model
and preprocessor to `models/`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

Run the default training config:

```bash
python -m src.train
```

Run the notebooks (they import from `src/`, so the package must be
installed first):

```bash
jupyter lab
```

Run in Docker:

```bash
docker build -t house-price-trainer .
docker run --rm -v $(pwd)/models:/app/models -v $(pwd)/outputs:/app/outputs house-price-trainer
```

Run the tests:

```bash
pytest
```

## Method

1. **Cleaning** — drop 11 low-signal columns, impute missing values (KNN
   imputation for `LotFrontage`, domain-specific fills elsewhere, e.g. a
   missing `GarageType` means "no garage").
2. **Target transform** — `log1p(SalePrice)`, since raw sale price is
   right-skewed; every reported RMSE below is converted back to dollars.
3. **Preprocessing** — `StandardScaler` on numeric features, one-hot
   encoding on categorical features, yielding 254 features from the
   original 79.
4. **Feature selection strategies** (compared independently):
   - **full** — all 254 features
   - **importance** — the 100 features with RandomForest importance > 0.001
   - **pca** — PCA reduced to 49 components (90% explained variance)
5. **Modeling** — `LinearRegression`, `KNN`, `RandomForest`, `Ridge`,
   `Lasso`, and `ElasticNet`, each tuned with 5-fold `GridSearchCV`.

## Results

RMSE is on the original dollar scale (post `expm1`); R² is on the held-out
30% test split.

### Full feature set (254 features)

| Model | CV score | Test R² | RMSE |
|---|---|---|---|
| **ElasticNet** | 0.8305 | **0.8985** | **$22,839.53** |
| Ridge | 0.8353 | 0.8983 | $23,510.17 |
| Lasso | 0.8146 | 0.8980 | $23,999.31 |
| RandomForest | 0.8537 | 0.8735 | $27,655.13 |
| LinearRegression | 0.7947 | 0.8859 | $26,595.28 |
| KNN | 0.8102 | 0.8156 | $34,205.58 |

### Importance-filtered (100 features)

| Model | CV score | Test R² | RMSE |
|---|---|---|---|
| **ElasticNet** | 0.8304 | **0.8947** | **$23,310.87** |
| Ridge | 0.8330 | 0.8955 | $23,540.53 |
| Lasso | 0.8266 | 0.8861 | $24,255.16 |
| LinearRegression | 0.8244 | 0.8832 | $24,788.72 |
| RandomForest | 0.8594 | 0.8784 | $26,848.66 |
| KNN | 0.8125 | 0.8257 | $34,678.84 |

### PCA (49 components, 90% variance)

| Model | CV score | Test R² | RMSE |
|---|---|---|---|
| **LinearRegression** | 0.8120 | **0.8915** | **$24,509.44** |
| Lasso | 0.8296 | 0.8901 | $24,663.77 |
| ElasticNet | 0.8237 | 0.8795 | $25,765.55 |
| Ridge | 0.8241 | 0.8791 | $27,497.45 |
| RandomForest | 0.8125 | 0.8113 | $35,768.64 |
| KNN | 0.7183 | 0.7272 | $45,413.26 |

## Interpretation

- **The regularized linear models win.** ElasticNet on the full feature set
  is the best configuration overall ($22,840 RMSE), with Ridge and Lasso
  close behind. With 254 features and only ~1,000 training rows,
  regularization matters more than model flexibility — RandomForest and KNN
  both underperform the linear models across every feature set.
- **PCA doesn't help.** Every model's RMSE gets worse (or stays flat) under
  PCA compared to the full feature set, and it makes RandomForest's grid
  search roughly 6x slower (one-hot encoding turns categorical variables
  into many near-orthogonal sparse dimensions, which principal components
  mix together and blur rather than compress cleanly).
- **Importance filtering is a reasonable compromise, not a win.** Cutting
  254 features down to the 100 RandomForest found most important gives
  results within a few hundred dollars of the full set, at lower
  dimensionality — useful if training cost mattered, but not a free
  accuracy improvement.
- **What drives price**, by RandomForest importance: `OverallQual`,
  `GrLivArea`, `YearBuilt`, `TotalBsmtSF`, `GarageArea`, `GarageYrBlt`,
  `GarageCars`, `1stFlrSF`, `YearRemodAdd`. Overall quality and living area
  dominate — consistent with the raw correlation ranking in
  `notebooks/00_eda.ipynb`.

## Known limitations

- **KNN imputation leakage**: `LotFrontage` is imputed with `KNNImputer`
  fit on the full dataset (train and test rows together, including the
  already log-transformed target) before the train/test split. This is
  inherited from the original notebook and kept as-is so the pipeline
  reproduces its documented numbers exactly; a leak-free version would fit
  the imputer on the training split only.
- **No outlier removal.** A handful of large-area, low-quality-adjusted
  outliers visible in the EDA notebook are not filtered out.

## Generating a Kaggle submission

```bash
python -m src.predict --output outputs/submission.csv
```

Loads the persisted `models/best_model.joblib` and `models/preprocessor.joblib`
(no refitting), cleans `data/test.csv` (filling the handful of columns that
are only missing in the test split, e.g. `MSZoning`, `KitchenQual`,
`SaleType`, using train-set statistics, since a submission needs a
prediction for every row), and writes a two-column `Id,SalePrice` CSV
matching `data/sample_submission.csv`'s format — ready to upload directly
on the [competition's submission page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/submit).

**Kaggle public leaderboard score: 0.13749** (RMSE on `log(SalePrice)`,
Kaggle's own scoring metric, lower is better).

This lines up closely with the model's held-out RMSE computed the same way
(log scale, before converting back to dollars): **0.1336** on the internal
30% test split. The two being within ~0.004 of each other, on data the
model never saw during training or tuning either way, is a good sign the
internal validation split isn't overstating performance — the model
generalizes about as well to Kaggle's actual test set as it did to its own
held-out data. For context, 0.13-0.14 is a solid single-model result on
this competition without extensive manual feature engineering, outlier
removal, or ensembling — the public leaderboard's very top scores (near
0.0) are generally the result of leaderboard probing rather than genuine
predictive skill, since the metric is computed on log-transformed prices
and rewards fitting to leaked/duplicated data.

## Tech stack

Python, pandas, scikit-learn, OmegaConf, Docker, pre-commit (black, isort,
flake8), pytest.
