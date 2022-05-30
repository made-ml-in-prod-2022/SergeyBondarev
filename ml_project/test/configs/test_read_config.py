from src.configs.read_config import read_config


CONFIG_PATH = "../../configs/config.yaml"


def test_config_read_with_no_errors():
    read_config()


def test_config_has_all_params():
    config = read_config()
    assert '.csv' in config.input_data_path
    assert 'models' in config.output_model_path
    assert config.train_params.model_type
    assert config.train_params.feature_params.categorical_features
    assert config.train_params.feature_params.numerical_features
    assert config.train_params.feature_params.features_to_drop
