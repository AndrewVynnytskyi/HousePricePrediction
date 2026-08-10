from omegaconf import OmegaConf

from src.data import prepare_dataset


def test_prepare_dataset_shape():
    cfg = OmegaConf.load("configs/train.yaml")
    prepared = prepare_dataset(cfg)

    assert prepared.X_train.shape[1] == 254
    assert prepared.X_test.shape[1] == 254
    assert prepared.X_train.shape[0] + prepared.X_test.shape[0] == len(prepared.y_train) + len(
        prepared.y_test
    )
    assert len(prepared.feature_names) == 254


def test_prepare_dataset_no_missing_values():
    cfg = OmegaConf.load("configs/train.yaml")
    prepared = prepare_dataset(cfg)

    assert not prepared.X_train.isna().any().any()
    assert not prepared.X_test.isna().any().any()
