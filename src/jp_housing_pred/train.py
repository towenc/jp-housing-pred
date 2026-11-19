import sys

import mlflow
import yaml

from jp_housing_pred.evaluate import evaluate
from jp_housing_pred.features import PROJECT_ROOT
from jp_housing_pred.models import GlobalMean, GroupMean
from jp_housing_pred.split import load_split, period_range

# YAML config model name -> model class
# Add new models here as needed
MODELS = {
    "global_mean": GlobalMean,
    "group_mean": GroupMean,
}

def build_model(entry):
    """Builds a model instance from a configuration entry

    Args:
        entry (dict): Configuration entry containing the model name and parameters

    Returns:
        An instance of the specified model class with relevant parameters from config file
    """
    cls = MODELS[entry["name"]]
    params = {k: v for k, v in entry.items() if k != "name"}
    return cls(**params)

def train(cfg):
    data_cfg = cfg["data"]
    data_folder = PROJECT_ROOT / data_cfg["folder"]  
    train_data = load_split(data_folder, data_cfg["train_split"])
    validation_data = load_split(data_folder, data_cfg["eval_split"])

    periods = {
        "train_period": period_range(train_data),
        "validation_period": period_range(validation_data)
    }

    print(f"Train split: {len(train_data)} samples ({periods['train_period']})")
    print(f"Validation split: {len(validation_data)} samples ({periods['validation_period']})")

    mlflow.set_experiment(cfg["experiment"])

    for entry in cfg["models"]:
        with mlflow.start_run(run_name=entry["name"]):
            model = build_model(entry)
            metrics = evaluate(model, train_data, validation_data) 

            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"{entry['name']}: {metrics_str}")

            # log parameters, metrics, and config for the current run
            mlflow.log_params(entry)
            mlflow.log_params(data_cfg)
            mlflow.log_params(periods)
            mlflow.log_metrics(metrics)
            mlflow.log_dict(cfg, "config.yaml")

def main():
    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    train(cfg)

if __name__ == "__main__":
    main()
