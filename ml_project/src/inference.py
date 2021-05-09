import sys
import os
from typing import List, Dict, Set, Tuple, Union, Iterable, Optional
import pandas as pd
from hydra.utils import to_absolute_path

from .utils import CustomException, load_object, save_object, setup_logging
from .parametrization import load_inference_params, parse_inference_params
from .preprocessing import extract_features, extract_target
from .evaluation import generate_report


def save_predictions(predictions: Iterable, filepath: str, index=None):
    """
    saves model predictions
    :param predictions: predictions
    :param filepath: filepath
    :return: None
    """
    if index is not None:
        save_index = index
    else:
        save_index = range(len(predictions))
    pd.DataFrame(predictions, columns=["prediction"], index=save_index).to_csv(filepath)


def inference_pipeline(path_to_config: Union[str, dict], display_report=False):

    logger = setup_logging()

    logger.info("Inference pipeline started")

    logger.info(f"Reading inference pipeline parameters")
    inference_params = load_inference_params(path_to_config)

    logger.info(f"Parsing config")
    (
        path_dataset,
        path_to_models,
        path_to_predictions,
        binary,
        cutoff,
    ) = parse_inference_params(inference_params)

    logger.info(f"Loading model components from path: {path_to_models}")
    model = load_object(path_to_models + "classifier", verbose=False)
    optimal_cutoff = load_object(path_to_models + "optimal_cutoff", verbose=False)
    transformers = load_object(path_to_models + "transformers", verbose=False)
    num_features = load_object(path_to_models + "num_features", verbose=False)
    cat_features = load_object(path_to_models + "cat_features", verbose=False)

    logger.info(f"Loading data from path: {path_dataset}")
    data = pd.read_csv(path_dataset)
    logger.info(f"Data shape: {data.shape}, Data columns: {list(data.columns)}")

    logger.info(f"Checking dataframe for model consistency")
    for f in num_features + cat_features:
        if f not in data.columns:
            raise CustomException(
                f"cannot find column '{f}' in dataframe which is needed for model inference"
            )
    logger.info(f"Creating features")
    X = extract_features(
        data, cat_features, num_features, transformers, mode="transform"
    )
    logger.info(f"Features shape: {X.shape}")

    logger.info(f"Extracting target if provided")
    if inference_params.target_name:
        target_name = inference_params.target_name
        if target_name in data.columns:
            y = extract_target(data, target_name)
            logger.info(f"Target shape: {y.shape}")
        else:
            y = None
            logger.info(f"Target not found in dataframe")
    else:
        y = None
        logger.info(f"Target not provided")
    logger.info(f"Scoring with model")
    if binary:
        logger.info(f"Binarizing predictions")
        if cutoff is not None:
            logger.info(f"Using cutoff value provided in config: {cutoff:.4f}")
            y_predicted = make_binary_prediction(model, X, cutoff)
        else:
            logger.info(
                f"Using cutoff value determined while training: {optimal_cutoff:.4f}"
            )
            cutoff = optimal_cutoff
            y_predicted = make_binary_prediction(model, X, cutoff)
    else:
        if cutoff is None:
            cutoff = optimal_cutoff  ### for report if will be
        y_predicted = model.predict_proba(X)[:, 1]
    logger.info(f"Saving predictions to {path_to_predictions}")
    save_predictions(y_predicted, path_to_predictions, index=data.index)

    if y is not None and inference_params.path_to_report is not None:
        logger.info(
            f"Generating report and saving to {inference_params.path_to_report}"
        )
        generate_report(
            X=X,
            y=y,
            model=model,
            cutoff=cutoff,
            path_to_report=to_absolute_path(inference_params.path_to_report),
            display_report=display_report,
        )
    logger.info(f"Finished")
