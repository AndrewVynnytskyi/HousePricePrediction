"""Config-driven training entrypoint.

    python -m src.train --config configs/train.yaml
    python -m src.train training.models='[LinearRegression,KNN,RandomForest,Ridge,Lasso,ElasticNet]'

Trainer.fit() runs, per model: GridSearchCV -> refit on best_params_ ->
R2 / cross_val_score / RMSE (on the dollar scale, after undoing the log1p
target transform).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import joblib
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score

from src.data import PreparedData, prepare_dataset
from src.model import (
    ModelSpec,
    build_model_registry,
    build_pca_reducer,
    select_features_by_importance,
)
from src.utils import MetricsLogger, RunMetrics, get_logger

logger = get_logger(__name__)


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.metrics_logger = MetricsLogger(cfg.output.metrics_csv, cfg.output.runs_dir)
        self._prepared: Optional[PreparedData] = None

    @property
    def prepared(self) -> Optional[PreparedData]:
        return self._prepared

    def prepare_data(self) -> None:
        prepared = prepare_dataset(self.cfg)
        mode = self.cfg.feature_selection.mode

        if mode == "importance":
            selected = select_features_by_importance(prepared.X_train, prepared.y_train, self.cfg)
            prepared.X_train = prepared.X_train[selected]
            prepared.X_test = prepared.X_test[selected]
            prepared.feature_names = selected
        elif mode == "pca":
            reducer = build_pca_reducer(
                self.cfg.feature_selection.pca.n_components, self.cfg.feature_selection.pca.whiten
            )
            X_train_pca = reducer.fit_transform(prepared.X_train)
            X_test_pca = reducer.transform(prepared.X_test)
            prepared.X_train = X_train_pca
            prepared.X_test = X_test_pca
            prepared.feature_names = [f"pc_{i}" for i in range(X_train_pca.shape[1])]
        elif mode != "full":
            raise ValueError(f"Unknown feature_selection.mode: {mode}")

        self._prepared = prepared
        logger.info("Prepared data: mode=%s, n_features=%d", mode, len(prepared.feature_names))

    def fit(self) -> List[RunMetrics]:
        if self._prepared is None:
            raise RuntimeError("Call prepare_data() before fit()")

        registry = build_model_registry(self.cfg)
        all_metrics: List[RunMetrics] = []
        best: Optional[Tuple[RunMetrics, Any]] = None

        for name, spec in registry.items():
            logger.info("Running grid search for %s", name)
            model, metrics = self._run_grid_search(spec)
            all_metrics.append(metrics)
            self.metrics_logger.log(metrics)
            logger.info(
                "%s: best_cv_score=%.4f test_r2=%.4f rmse_dollars=%.2f",
                name,
                metrics.best_cv_score,
                metrics.test_r2,
                metrics.rmse_dollars,
            )

            if best is None or metrics.rmse_dollars < best[0].rmse_dollars:
                best = (metrics, model)

        if best is not None:
            self._persist_best_model(best[1])

        return all_metrics

    def _run_grid_search(self, spec: ModelSpec) -> Tuple[Any, RunMetrics]:
        prepared = self._prepared
        assert prepared is not None

        grid = GridSearchCV(
            spec.estimator_cls(),
            param_grid=spec.param_grid,
            n_jobs=self.cfg.training.n_jobs,
            cv=self.cfg.training.cv_folds,
        )
        start_time = time.time()
        grid.fit(prepared.X_train, prepared.y_train)
        grid_search_seconds = time.time() - start_time

        model = spec.estimator_cls(**grid.best_params_)
        model.fit(prepared.X_train, prepared.y_train)

        y_test_eval = prepared.y_test.values.ravel() if spec.ravel_y else prepared.y_test

        cv_scores = cross_val_score(
            model, prepared.X_train, prepared.y_train, cv=self.cfg.training.cv_folds
        )

        metrics = self._evaluate(
            model=model,
            X_test=prepared.X_test,
            y_test=y_test_eval,
            spec=spec,
            grid=grid,
            grid_search_seconds=grid_search_seconds,
            cv_scores=cv_scores,
        )
        return model, metrics

    def _evaluate(
        self, model, X_test, y_test, spec, grid, grid_search_seconds, cv_scores
    ) -> RunMetrics:
        y_pred = model.predict(X_test)
        test_r2 = model.score(X_test, y_test)
        rmse_dollars = mean_squared_error(np.expm1(y_test), np.expm1(y_pred)) ** 0.5

        return RunMetrics(
            model_name=spec.name,
            feature_selection=self.cfg.feature_selection.mode,
            best_params=dict(grid.best_params_),
            grid_search_seconds=grid_search_seconds,
            best_cv_score=float(grid.best_score_),
            test_r2=float(test_r2),
            cv_scores=[float(s) for s in cv_scores],
            cv_score_mean=float(cv_scores.mean()),
            rmse_dollars=float(rmse_dollars),
        )

    def _persist_best_model(self, model: Any) -> None:
        if not self.cfg.output.save_best_only:
            return
        models_dir = Path(self.cfg.output.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, models_dir / "best_model.joblib")
        if self._prepared is not None:
            joblib.dump(self._prepared.preprocessor, models_dir / "preprocessor.joblib")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args, overrides = parser.parse_known_args()

    base_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, OmegaConf.from_dotlist(overrides))

    trainer = Trainer(cfg)
    trainer.prepare_data()
    trainer.fit()


if __name__ == "__main__":
    main()
