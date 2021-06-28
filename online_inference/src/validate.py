from typing import Tuple, Union

import pandas as pd

from .entities import InputData

CAT_FEATURES = ["sex", "fbs", "restecg", "exang", "slope"]
NUM_FEATURES = ["age", "cp", "trestbps", "chol", "thalach", "oldpeak", "ca", "thal"]
MODEL_FEATURES = CAT_FEATURES + NUM_FEATURES

MODEL_FEATURES_BOUNDS = {
    "sex": [0, 1],
    "fbs": [0, 1],
    "restecg": [0, 2],
    "exang": [0, 1],
    "slope": [0, 2],
    "age": [18, 100],
    "cp": [0, 3],
    "trestbps": [90, 210],
    "chol": [110, 600],
    "thalach": [65, 210],
    "oldpeak": [0, 7],
    "ca": [0, 4],
    "thal": [0, 3],
}


def check_value_in_interval(
    value: Union[int, float, pd.Series],
    lower_bound: Union[int, float],
    upper_bound: Union[int, float],
    name: str,
):
    if isinstance(value, pd.Series):
        min_value, max_value = min(value), max(value)
        if (lower_bound > min_value) or (max_value > upper_bound):
            raise ValueError(
                f"value in '{name}' is out of [{lower_bound}, {upper_bound}] interval"
            )
    else:
        if not (lower_bound <= value <= upper_bound):
            raise ValueError(
                f"value {value} in '{name}' is out of [{lower_bound}, {upper_bound}] interval"
            )


def check_data_valid(data: Tuple[InputData, pd.DataFrame]) -> Tuple[bool, str]:
    try:
        if isinstance(data, pd.DataFrame):

            data = data[MODEL_FEATURES]
            for f, bounds in MODEL_FEATURES_BOUNDS.items():
                check_value_in_interval(data[f], *bounds, f)
        else:
            for f, bounds in MODEL_FEATURES_BOUNDS.items():
                value = data.__getattribute__(f)
                check_value_in_interval(value, *bounds, f)
        return True, "Data is valid"
    except ValueError as error:
        return False, str(error)
