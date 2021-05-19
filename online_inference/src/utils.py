import os
import sys
import json
import pickle
import logging
from typing import List, Dict, Set, Tuple, Union, Iterable, Optional


class CustomException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def setup_logging():
    logger = logging.getLogger(__name__)
    while logger.handlers:
        logger.handlers.pop()
    handler = logging.StreamHandler(sys.stdout)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.propagate = False
    return logger


def load_object(filepath, verbose=False, suffix=".pkl"):
    filepath += suffix
    if verbose:
        print(f"Reading from file {filepath}  ...")
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    if verbose:
        print(f"Got object of type {type(obj)}")
    return obj


def make_binary_prediction(model, X, cutoff):
    y_predicted = model.predict_proba(X)[:, 1]
    y_predicted_binary = (y_predicted >= cutoff).astype(int)
    return y_predicted_binary
