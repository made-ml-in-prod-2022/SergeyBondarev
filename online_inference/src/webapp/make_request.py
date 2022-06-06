import logging
import requests


logger = logging.getLogger(__name__)


def make_request(data):
    """
    Makes a request to the given url with the given data.
    """
    logger.info("Health check request")

    response = requests.get("http://localhost:8000/health")
    if response.status_code != 200:
        raise Exception("Health check request failed")

    logger.info("Predict request")

    response = requests.post("http://localhost:8000/predict", json=data)
    if response.status_code != 200:
        raise Exception("Predict request failed")

    logger.info("Predict request successful")
    logger.info("Response: {}".format(response.json()))
    return response.json()


if __name__ == "__main__":
    TEST_DATA = [
        [51, 120, 295, 157, 0.6, 0, 2, 2, 0, 0, 0, 0, 0],
        [60, 140, 185, 155, 3.0, 1, 2, 2, 0, 0, 1, 0, 0]
    ]
    print(
        make_request({'data': TEST_DATA})
    )



