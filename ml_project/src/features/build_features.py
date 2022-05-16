# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from ml_project.src.utils.read_config import read_config


logger = logging.getLogger(__name__)


def one_hot_encoding(data: pd.DataFrame, cat_features: list) -> pd.DataFrame:
    logger.info("One hot encoding on categorical features...")
    return data.drop(cat_features, axis=1).merge(pd.get_dummies(data[cat_features]))


def remove_outliers(data: pd.DataFrame):
    logger.info("Removing outliers...")
    pass


def scale_data(data: pd.DataFrame):
    logger.info("Scaling data...")
    pass


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('config_filepath', type=click.Path(exists=True))
def build_features(input_filepath, output_filepath, config_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info('making final dataset from raw data')

    config = read_config(config_filepath)

    data = pd.read_csv(input_filepath)

    data = remove_outliers(data)
    data = scale_data(data)
    data = one_hot_encoding(data, config.train_params.feature_params.categorical_features)

    data.to_csv(output_filepath, index=False)

    return data


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())
    build_features()
