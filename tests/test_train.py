import mlflow
import pyarrow as pa
import pytest

from jp_housing_pred.evaluate import METRICS
from jp_housing_pred.models import GlobalMean, GroupMean
from jp_housing_pred.train import build_model, train


# ------------ build_model() ------------
def test_build_model_with_params():
    # Arrange
    entry = {"name": "group_mean", "group": "pref"}

    # Act
    model = build_model(entry)

    # Assert
    assert isinstance(model, GroupMean)
    assert model.group == "pref"

def test_build_model_without_params():
    # Arrange
    entry = {"name": "global_mean"}

    # Act
    model = build_model(entry)

    # Assert
    assert isinstance(model, GlobalMean)

def test_build_model_invalid_name():
    # Arrange
    entry = {"name": "invalid_model"}

    # Act & Assert
    with pytest.raises(KeyError):
        build_model(entry)

# ------------ train() ------------
def test_train_logs_runs_to_mlflow(tmp_path, monkeypatch):
    # Arrange
    def fake_load_split(folder, split):
        return pa.table({
            "log_price": [10.0, 11.0, 12.0 , 13.0],
            "pref": ["13", "13", "27", "27"],
            "year": [2025, 2025, 2026, 2026],
            "quarter": [3, 4, 1, 2],
        })
    
    monkeypatch.setattr("jp_housing_pred.train.load_split", fake_load_split)
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")

    cfg = {
        "experiment": "test_experiment",
        "data": {"folder": "unused", "train_split": "train", "eval_split": "validation"},
        "models": [
            {"name": "global_mean"},
            {"name": "group_mean", "group": "pref"}
        ],
    }

    # Act
    train(cfg)

    # Assert
    runs = mlflow.search_runs(experiment_names=["test_experiment"])
    assert len(runs) == 2
    assert set(runs["params.name"]) == {"global_mean", "group_mean"}
    assert (runs["params.train_period"] == "2025Q3 - 2026Q2").all()
    for metric in METRICS:
        assert f"metrics.{metric}" in runs.columns
