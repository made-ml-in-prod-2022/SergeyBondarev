import os
import logging
import pandas as pd

from dataclasses import dataclass

from joblib import dump

from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.configs import configs


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


logger = logging.getLogger(__name__)


def train_model(save_model=True):
    logger.info('Reading X_train and y_train from {}'.format(configs.interim_data_path))

    X_train = pd.read_csv(os.path.join(configs.interim_data_path, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(configs.interim_data_path, 'y_train.csv'))

    logger.info('Train model {}'.format(configs.train_params.model_type))

    model = None
    if configs.train_params.model_type == 'random_forest':
        logger.info('Training random forest model')
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**configs.train_params.model_params)
    elif configs.train_params.model_type == 'logistic_regression':
        logger.info('Training logistic regression model')
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(**configs.train_params.model_params)
    elif configs.train_params.model_type == 'knn':
        logger.info('Training knn model')
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(**configs.train_params.model_params)
    else:
        logger.error('Model type {} not supported'.format(configs.train_params.model_type))
        raise ValueError('Model type {} not supported'.format(configs.train_params.model_type))

    model.fit(X_train, y_train.values.ravel())

    if save_model:
        logger.info('Save model {}'.format(configs.train_params.model_type))
        dump(model, configs.output_model_path)

    return model


def predict(model: ClassifierMixin, save_predictions=True):
    X_test = pd.read_csv(os.path.join(configs.interim_data_path, 'X_test.csv'))

    predicted = model.predict(X_test)
    if save_predictions:
        pd.DataFrame(predicted).to_csv(configs.predictions_data_path, index=False)

    return predicted


def evaluate_model(actual, predicted):
    metrics = Metrics(
        accuracy=accuracy_score(actual, predicted),
        precision=precision_score(actual, predicted),
        recall=recall_score(actual, predicted),
        f1=f1_score(actual, predicted),
        roc_auc=roc_auc_score(actual, predicted),
    )
    return metrics


def train_model_done():
    # Always retrain the model
    return False
