from ..configs import configs
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi


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
    Get data from Kaggle.
    :param dataset_name: Name of the dataset.
    :param dataset_path: Path to the dataset.
    :return: None
    """
    api = authenticate()
    api.dataset_download_files(dataset_name, dataset_path)


if __name__ == '__main__':
    get_data(configs, 'data/raw')
