from omegaconf import OmegaConf

from src.model import build_model_registry

ALL_MODELS = ["LinearRegression", "KNN", "RandomForest", "Ridge", "Lasso", "ElasticNet"]


def test_registry_builds_all_six_models():
    cfg = OmegaConf.load("configs/train.yaml")
    cfg.training.models = ALL_MODELS

    registry = build_model_registry(cfg)

    assert set(registry.keys()) == set(ALL_MODELS)
    for name, spec in registry.items():
        assert spec.name == name
        assert isinstance(spec.param_grid, dict)
        assert len(spec.param_grid) > 0


def test_random_forest_is_the_only_model_that_ravels_y():
    cfg = OmegaConf.load("configs/train.yaml")
    cfg.training.models = ALL_MODELS

    registry = build_model_registry(cfg)

    assert registry["RandomForest"].ravel_y is True
    for name in set(ALL_MODELS) - {"RandomForest"}:
        assert registry[name].ravel_y is False
