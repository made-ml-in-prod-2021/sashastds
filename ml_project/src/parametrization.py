import yaml
import os
from typing import List, Dict, Set, Union, Tuple, Iterable, Optional
from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema
from hydra.utils import to_absolute_path


@dataclass
class SplittingParams:
    test_size: float = field(default=0.2)
    random_state: int = field(default=8)


@dataclass
class EncoderParams:
    encoder_type: str = field(default="categorical")
    recognize: bool = field(default=False)
    verbose: bool = field(default=False)
    map_unknown_to_na: bool = field(default=True)
    force_nan_and_unknown_category: bool = field(default=True)
    cat_na_fill_value: str = field(default="(UNK)")
    drop: str = field(default=None)


@dataclass
class ClassifierParams:
    classifier_type: str = field(default="lgbm")
    random_state: int = field(default=8)
    n_jobs: int = field(default=2)
    n_estimators: int = field(default=10)
    max_depth: int = field(default=5)
    num_leaves: int = field(default=15)
    min_samples_leaf: int = field(default=10)
    reg_lambda: float = field(default=0.5)
    objective: str = field(default="binary")
    criterion: str = field(default="gini")


@dataclass
class FeatureParams:
    target_name: Optional[str] = field(default=None)
    numerical_features: Optional[List[str]] = field(default=None)
    categorical_features: Optional[List[str]] = field(default=None)
    exclude_features: Optional[List[str]] = field(default=None)


@dataclass
class TrainingParams:
    path_dataset: str
    path_to_models: str
    path_to_report: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    encoder_params: EncoderParams
    classifier_params: ClassifierParams


@dataclass
class InferenceParams:
    path_dataset: str
    path_to_models: str
    path_to_predictions: str
    binary: bool = field(default=False)
    cutoff: float = field(default=None)
    path_to_report: Optional[str] = field(default=None)
    target_name: Optional[str] = field(default=None)


TrainingParamsSchema = class_schema(TrainingParams)
InferenceParamsSchema = class_schema(InferenceParams)


def read_config(config_path: str) -> Dict[str, Union[int, str]]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_training_params(config_path: Union[str, dict]) -> TrainingParams:
    """
    loads parameters for pipeline training
    :param config_path: path to config file
    :return: TrainingParams
    """
    if isinstance(config_path, str):
        params_dict = read_config(config_path)
    else:
        params_dict = config_path
    schema = TrainingParamsSchema()

    return schema.load(params_dict)


def parse_training_params(
    training_params: TrainingParams,
) -> Tuple[str, SplittingParams, EncoderParams, ClassifierParams, FeatureParams]:
    path_dataset = to_absolute_path(training_params.path_dataset)
    path_to_models = to_absolute_path(training_params.path_to_models) + os.sep
    path_to_report = to_absolute_path(training_params.path_to_report) + os.sep
    splitting_params = training_params.splitting_params
    feature_params = training_params.feature_params
    encoder_params = training_params.encoder_params
    classifier_params = training_params.classifier_params
    return (
        path_dataset,
        path_to_models,
        path_to_report,
        splitting_params,
        feature_params,
        encoder_params,
        classifier_params,
    )


def load_inference_params(config_path: Union[str, dict]) -> InferenceParams:
    """
    loads parameters for pipeline training
    :param config_path: path to config file
    :return: TrainingParams
    """
    if isinstance(config_path, str):
        params_dict = read_config(config_path)
    else:
        params_dict = config_path
    schema = InferenceParamsSchema()
    return schema.load(params_dict)


def parse_inference_params(
    inference_params: InferenceParams,
) -> Tuple[str, str, bool, float]:
    path_dataset = to_absolute_path(inference_params.path_dataset)
    path_to_models = to_absolute_path(inference_params.path_to_models) + os.sep
    path_to_predictions = to_absolute_path(inference_params.path_to_predictions)
    binary = inference_params.binary
    cutoff = inference_params.cutoff
    return path_dataset, path_to_models, path_to_predictions, binary, cutoff
