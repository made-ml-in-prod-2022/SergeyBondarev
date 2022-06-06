import os
import json
import logging
import pandas as pd

from src.configs import configs
from src.data.get_data import get_data, get_data_done
from src.features import build_features, build_features_done
from src.models import train_model, train_model_done, evaluate_model, predict


logger = logging.getLogger(__name__)


def run_pipeline():
    """
    Runs the entire pipeline. Includes:
    1. Get data from Kaggle
    2. Build features
    3. Train model
    4. Evaluate model and save metrics
    """
    if not get_data_done(configs.raw_data_path):
        get_data(configs.dataset_name, configs.raw_data_path)

    if not build_features_done():
        build_features()

    if not train_model_done():
        model = train_model()

        X_test = pd.read_csv(os.path.join(configs.interim_data_path, 'X_test.csv'))
        actual = pd.read_csv(os.path.join(configs.interim_data_path, 'y_test.csv'))

        predicted = predict(model)
        metrics = evaluate_model(predicted, actual)
        logger.info("Computed metrics: {}".format(metrics))
        logger.info("Saving computed metrics to {}".format(configs.metrics_data_path))

        with open(configs.metrics_data_path, 'w') as file:
            json.dump(metrics.__dict__, file)


if __name__ == "__main__":
    run_pipeline()
