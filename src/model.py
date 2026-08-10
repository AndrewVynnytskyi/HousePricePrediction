"""Estimator registry and feature-selection helpers.

There is no neural network in this project — all six candidate models are
scikit-learn regressors, tuned via GridSearchCV, exactly as in the original
notebook's `models_config` dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Type

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor

_ESTIMATOR_REGISTRY: Dict[str, Type[BaseEstimator]] = {
    "LinearRegression": LinearRegression,
    "KNN": KNeighborsRegressor,
    "RandomForest": RandomForestRegressor,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
}


@dataclass
class ModelSpec:
    name: str
    estimator_cls: Type[BaseEstimator]
    param_grid: Dict[str, Any]
    ravel_y: bool


def build_model_registry(cfg: DictConfig) -> Dict[str, ModelSpec]:
    registry: Dict[str, ModelSpec] = {}
    for name in cfg.training.models:
        model_cfg = cfg.models[name]
        registry[name] = ModelSpec(
            name=name,
            estimator_cls=_ESTIMATOR_REGISTRY[name],
            param_grid=OmegaConf.to_container(model_cfg.param_grid, resolve=True),
            ravel_y=bool(model_cfg.ravel_y),
        )
    return registry


def select_features_by_importance(
    X_train: pd.DataFrame, y_train: pd.Series, cfg: DictConfig
) -> List[str]:
    """Fits a RandomForestRegressor and keeps features with
    importance_ > threshold, sorted descending — mirrors the original
    notebook's feature-importance filtering step.
    """
    importance_cfg = cfg.feature_selection.importance
    selector_params = OmegaConf.to_container(importance_cfg.selector_params, resolve=True)
    rfr = RandomForestRegressor(**selector_params)
    rfr.fit(X_train, y_train)

    importance = pd.Series(rfr.feature_importances_, index=X_train.columns)
    importance = importance.sort_values(ascending=False)
    importance = importance[importance > importance_cfg.threshold]
    return importance.index.tolist()


def build_pca_reducer(n_components: float, whiten: bool) -> PCA:
    return PCA(n_components=n_components, whiten=whiten)
