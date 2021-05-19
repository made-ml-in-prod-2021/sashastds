import requests
import json

import pandas as pd

from src.utils import setup_logging


PATH_TO_DATA = "./data/sample.csv"
LOCALHOST = '127.0.0.1'
PORT = 8000
DOMAIN = f'{LOCALHOST}:{PORT}'
ENDPOINT = 'predict'

if __name__ == "__main__":
    logger = setup_logging()

    logger.info("Reading data")
    data = pd.read_csv(PATH_TO_DATA).drop("target", axis=1)
    data["id"] = range(len(data))

    request_data = data.to_dict(orient="records")
    logger.info(f"Request data samples:\n {request_data[::5]}")

    logger.info("Sending post request")
    response = requests.post(
        f"http://{DOMAIN}/{ENDPOINT}",
        json.dumps(request_data)
    )
    logger.info(f"Response status code: {response.status_code}")
    logger.info(f"Response data samples:\n {response.json()[::5]}")