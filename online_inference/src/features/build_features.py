# -*- coding: utf-8 -*-
import os
import logging
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split

from src.configs import configs


logger = logging.getLogger(__name__)


def one_hot_encoding(data: pd.DataFrame, cat_features: list) -> pd.DataFrame:
    logger.info("One hot encoding on categorical features...")
    return data.drop(cat_features, axis=1).merge(pd.get_dummies(data[cat_features]), left_index=True, right_index=True)


def remove_outliers(data: pd.DataFrame, target: pd.DataFrame):
    logger.info("Removing outliers...")
    ind = data[(data['trestbps'] > 170)].index
    data = data.drop(ind)
    target = target.drop(ind)

    ind = data[(data['chol'] > 350)].index
    data = data.drop(ind)
    target = target.drop(ind)

    ind = data[(data['thalach'] < 80)].index
    data = data.drop(ind)
    target = target.drop(ind)

    ind = data[(data['oldpeak'] > 4)].index
    data = data.drop(ind)
    target = target.drop(ind)

    return data, target


def scale_data(data: pd.DataFrame, should_scale=True):
    if not should_scale:
        logger.info("No scaling needed")
        return data

    logger.info("Scaling data...")
    num_features = data[configs.train_params.feature_params.numerical_features]

    data[configs.train_params.feature_params.numerical_features] = \
        num_features.apply(lambda x: (x - x.mean()) / x.std())
    return data


def split_to_train_test(dataframe: pd.DataFrame, target: pd.DataFrame):
    logger.info("Splitting data to train and test...")
    return train_test_split(
        dataframe,
        target,
        test_size=configs.train_params.splitting_params.test_size,
        random_state=configs.train_params.splitting_params.random_state,
    )


def build_features():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../predictions).
    """
    logger.info('making final dataset from raw data')

    data = pd.read_csv(configs.input_data_path)
    X_train, X_test, y_train, y_test = split_to_train_test(
        data.drop([configs.train_params.feature_params.target_col], axis=1),
        data[configs.train_params.feature_params.target_col],
    )

    X_train, y_train = remove_outliers(X_train, y_train)

    # no scaling here but can be easily changed by one parameter
    X_train = scale_data(X_train, should_scale=False)
    X_test = scale_data(X_test, should_scale=False)

    X_train = one_hot_encoding(X_train, configs.train_params.feature_params.categorical_features)
    X_test = one_hot_encoding(X_test, configs.train_params.feature_params.categorical_features)

    X_train.to_csv(os.path.join(configs.interim_data_path, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(configs.interim_data_path, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(configs.interim_data_path, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(configs.interim_data_path, 'y_test.csv'), index=False)


def build_features_done():
    return all([
        file in os.listdir(configs.interim_data_path)
        for file in ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']
    ])


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())
    build_features()
