import logging
import click
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils import read_config


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


logger = logging.getLogger(__name__)


def split_dataset(input_filepath: str, config):
    logger.info('splitting dataset in train and test')

    data = pd.read_csv(input_filepath)
    X = data.drop(config.train_params.feature_params.target_col, axis=1)
    y = data[config.train_params.feature_params.target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.train_params.splitting_params.test_size,
        random_state=config.train_params.splitting_params.random_state,
    )

    return X_train, X_test, y_train, y_test


def train(input_filepath: str, model_output_filepath: str, config_filepath: str):
    logger.info('Reading configs before training')

    config = read_config(config_filepath)

    logger.info('Splitting dataset')

    X_train, X_test, y_train, y_test = split_dataset(input_filepath, config)

    logger.info('Train model', config.train_params.model_type)

    if config.train_params.model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(config.train_params.model_params)
    elif config.train_params.model_type == 'logistic_regression':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(config.train_params.model_params)
    elif config.train_params.model_type == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(config.train_params.model_params)

    model.fit(X_train, y_train)

    logger.info('Save model', config.train_params.model_type)

    model.save(model_output_filepath)

    return model, X_test, y_test


def predict(model: Pipeline, test_data: pd.DataFrame, output_path: str):
    predicted = model.predict(test_data)
    predicted.to_csv(output_path, index=False)
    return predicted


def evaluate_model(predicted, actual):
    metrics = Metrics(
        accuracy=accuracy_score(actual, predicted),
        precision=precision_score(actual, predicted),
        recall=recall_score(actual, predicted),
        f1=f1_score(actual, predicted),
        roc_auc=roc_auc_score(actual, predicted),
    )
    return metrics


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('model_save_filepath', type=click.Path())
@click.argument('metric_save_filepath', type=click.Path())
@click.argument('config_filepath', type=click.Path(exists=True))
def main(
    input_filepath: str,
    output_filepath: str,
    model_save_filepath: str,
    metric_save_filepath: str,
    config_filepath: str,
):
    model, X_test, y_test = train(input_filepath, model_save_filepath, config_filepath)
    predicted = predict(model, X_test, output_filepath)
    metrics = evaluate_model(predicted, y_test)

    logger.info('Saving metrics', metrics)
    with open(metric_save_filepath, 'w') as f:
        f.write(metrics.__dict__)

    logger.info('Saving model', model_save_filepath)
    model.save(model_save_filepath)

    logger.info('Save predictions', output_filepath)
    predicted.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    main()
