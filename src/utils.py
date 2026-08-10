"""Local, file-based run logging (used in place of a tracking server)."""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Union


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@dataclass
class RunMetrics:
    model_name: str
    feature_selection: str
    best_params: Dict[str, Any]
    grid_search_seconds: float
    best_cv_score: float
    test_r2: float
    cv_scores: List[float]
    cv_score_mean: float
    rmse_dollars: float


class MetricsLogger:
    """Appends one row per run to a CSV and writes a per-run JSON snapshot."""

    def __init__(self, metrics_csv: Union[str, Path], runs_dir: Union[str, Path]) -> None:
        self.metrics_csv = Path(metrics_csv)
        self.runs_dir = Path(runs_dir)
        self.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: RunMetrics) -> None:
        row = asdict(metrics)
        row["best_params"] = json.dumps(row["best_params"])
        row["cv_scores"] = json.dumps(row["cv_scores"])

        write_header = not self.metrics_csv.exists()
        with self.metrics_csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        run_id = (
            f"{time.strftime('%Y%m%d-%H%M%S')}_{metrics.model_name}_{metrics.feature_selection}"
        )
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "metrics.json").open("w") as f:
            json.dump(asdict(metrics), f, indent=2)
