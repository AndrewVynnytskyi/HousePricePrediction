from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from src.predict import generate_submission


def test_generate_submission_matches_sample_format(tmp_path):
    cfg = OmegaConf.load("configs/train.yaml")
    output_path = tmp_path / "submission.csv"

    submission = generate_submission(cfg, Path(cfg.output.models_dir), output_path)
    sample = pd.read_csv("data/sample_submission.csv")

    assert list(submission.columns) == list(sample.columns)
    assert len(submission) == len(sample)
    assert (submission["Id"] == sample["Id"]).all()
    assert not submission["SalePrice"].isna().any()
    assert (submission["SalePrice"] > 0).all()
    assert output_path.exists()
