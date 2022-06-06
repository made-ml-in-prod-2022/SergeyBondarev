import os
import logging
import zipfile

from src.configs import configs
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi


logger = logging.getLogger(__name__)


def authenticate():
    """
    Authenticate with Kaggle.
    :return: Kaggle API object.
    """
    load_dotenv()
    api = KaggleApi()
    api.authenticate()
    return api


def get_data(dataset_name, dataset_path):
    """
    Get data from Kaggle and unzip it.
    :param dataset_name: Name of the dataset.
    :param dataset_path: Path to the dataset.
    :return: None
    """
    logger.info("Downloading data from Kaggle...")
    api = authenticate()
    api.dataset_download_files(dataset_name, dataset_path)

    logger.info("Unzipping data...")
    path_to_zip_file = os.path.join(dataset_path, 'heart-disease-cleveland-uci.zip')
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(dataset_path)


def get_data_done(dataset_path):
    """
    Checks if the data has been downloaded.
    :return: Boolean.
    """
    for file in os.listdir(dataset_path):
        if file.endswith("heart-disease-cleveland-uci.csv"):
            logger.info("Data has been already downloaded... No need to download again.")
            return True
    return False


if __name__ == '__main__':
    get_data(configs.dataset_name, 'data/raw')
