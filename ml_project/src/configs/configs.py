import yaml

from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

CONFIG_PATH = "configs/config.yaml"


@dataclass
class SplittingParams:
    test_size: float
    random_state: int


@dataclass
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]
    target_col: str


@dataclass
class TrainParams:
    model_type: str
    model_params: Dict[str, Any]
    feature_params: FeatureParams
    splitting_params: SplittingParams


@dataclass
class Config:
    raw_data_path: str
    interim_data_path: str
    predictions_data_path: str
    metrics_data_path: str
    dataset_name: str
    input_data_path: str
    output_model_path: str
    model_params_path: str
    train_params: TrainParams


def read_config(
    config_path: str = Path(__file__).parents[2] / CONFIG_PATH
) -> Config:
    """
    Reads the config file and returns a Config object.
    """
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    with open(Path(__file__).parents[2] / config['model_params_path'], "r", encoding='utf-8') as f:
        model_config = yaml.safe_load(f)

    return Config(
        raw_data_path=config['raw_data_path'],
        interim_data_path=config['interim_data_path'],
        predictions_data_path=config['predictions_data_path'],
        metrics_data_path=config['metrics_data_path'],
        dataset_name=config['dataset_name'],
        input_data_path=config["input_data_path"],
        output_model_path=config["output_model_path"],
        model_params_path=config["model_params_path"],
        train_params=TrainParams(
            model_type=config["train_params"]["model_type"],
            model_params=model_config,
            feature_params=FeatureParams(
                categorical_features=config["feature_params"]["categorical_features"],
                numerical_features=config["feature_params"]["numerical_features"],
                target_col=config["feature_params"]["target_col"],
            ),
            splitting_params=SplittingParams(
                test_size=config["splitting_params"]["test_size"],
                random_state=config["splitting_params"]["random_state"],
            ),
        )
    )


configs = read_config()
