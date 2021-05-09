from typing import List, Dict, Set, Tuple, Union, Iterable, Optional
import warnings
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from hydra.utils import to_absolute_path

from .utils import CustomException, load_object, save_object, setup_logging
from .preprocessing import (
    extract_features,
    extract_target,
    seek_types,
    seek_categorical_from_integer_data,
)
from .transformers import CategoricalEncoder, HotEncoder
from .evaluation import calc_precision_recall_curve_binary, generate_report
from .parametrization import load_training_params, parse_training_params


ClassificationModel = Union[LGBMClassifier, RandomForestClassifier]


def make_split(data, target_name, split_params) -> Tuple[pd.DataFrame, pd.DataFrame]:

    data_train, data_test = train_test_split(
        data,
        stratify=data[target_name],
        test_size=split_params.test_size,
        random_state=split_params.random_state,
    )

    return data_train, data_test


def recognize_features(
    data: pd.DataFrame,
    target_name: str,
    exclude_features: Union[List, Set] = None,
    verbose: bool = True,
) -> Tuple[List[str], List[str]]:

    if not exclude_features:
        exclude_features = []
    if target_name not in exclude_features:
        exclude_features.append(target_name)
    features = [f for f in data.columns if f not in exclude_features]

    feature_data = data.drop(exclude_features, axis=1)

    data_types = seek_types(feature_data)

    cat_features = data_types.get("cat", [])
    if verbose:
        print(f"recognised as categorical: {cat_features}")
    cat_recognized = seek_categorical_from_integer_data(feature_data, data_types["int"])
    if verbose:
        print(f"recognised as categorical from integers: {cat_recognized}")
    cat_features += cat_recognized

    num_features = [f for f in features if f not in cat_features]
    if verbose:
        print(f"recognised as numerical features: {num_features}")
    return cat_features, num_features


def create_encoder(encoder_params):

    if encoder_params.encoder_type == "categorical":
        encoder = CategoricalEncoder(
            recognize=encoder_params.recognize,
            verbose=encoder_params.verbose,
            map_unknown_to_na=encoder_params.map_unknown_to_na,
            cat_na_fill_value=encoder_params.cat_na_fill_value,
        )
    elif encoder_params.encoder_type == "onehot":
        encoder = HotEncoder(
            recognize=encoder_params.recognize,
            verbose=encoder_params.verbose,
            map_unknown_to_na=encoder_params.map_unknown_to_na,
            force_nan_and_unknown_category=encoder_params.force_nan_and_unknown_category,
            cat_na_fill_value=encoder_params.cat_na_fill_value,
            drop=encoder_params.drop,
        )
    else:
        raise CustomException(
            f"got unknown encoder type: {encoder_params.encoder_type}, expected either 'categorical' or 'onehot'"
        )
    return encoder


def create_transformers_pipeline(encoder):
    transformers = {
        "cat": [encoder],
        "num": [],
    }
    return transformers


def create_classifier(classifier_params) -> ClassificationModel:

    if classifier_params.classifier_type == "rf":
        rf_params = {
            "n_estimators": classifier_params.n_estimators,
            "max_depth": classifier_params.max_depth,
            "random_state": classifier_params.random_state,
            "n_jobs": classifier_params.n_jobs,
            "min_samples_leaf": classifier_params.min_samples_leaf,
            "criterion": classifier_params.criterion,
        }
        model = RandomForestClassifier(**rf_params)
    elif classifier_params.classifier_type == "lgbm":

        lgbm_params = {
            "n_estimators": classifier_params.n_estimators,
            "max_depth": classifier_params.max_depth,
            "random_state": classifier_params.random_state,
            "n_jobs": classifier_params.n_jobs,
            "num_leaves": classifier_params.num_leaves,
            "reg_lambda": classifier_params.reg_lambda,
            "objective": classifier_params.objective,
        }
        model = LGBMClassifier(**lgbm_params)
    else:
        raise CustomException(
            f"got unknown classifier type: {classifier_params.classifier_type}, expected either 'lgbm' or 'rf'"
        )
    return model


def get_xy(
    data: pd.DataFrame,
    target_name: str,
    cat_features: List[str],
    num_features: List[str],
    transformers: Dict[str, List],
    mode: "fit",
) -> Tuple[pd.DataFrame, pd.Series]:

    X = extract_features(data, cat_features, num_features, transformers, mode)
    y = extract_target(data, target_name)
    return X, y


def calc_optimal_cutoff(
    prs,
    recs,
    thrs,
    beta: float = 1,
    min_recall_level: float = None,
    second_preference: str = "precision",
):
    """
    takes results of `calc_precision_recall_curve_binary`

    `second_preference` is either 'precision' or 'recall'

    `beta` in arguments is provided not squared and is being squared in calculations
    """
    fss = []

    for pr, rc in zip(prs[:-1], recs[:-1]):
        fss.append((1 + beta ** 2) * pr * rc / ((beta ** 2) * pr + rc))
    res = pd.DataFrame(
        {
            "threshold": thrs,
            "precision": prs[:-1],
            "recall": recs[:-1],
            f"f-{beta}": fss,
        }
    )

    if second_preference not in ["precision", "recall"]:
        print("second_preference should be either 'precision' or 'recall'")
        print(f"got '{second_preference}' instead - falling back to 'precision'")
        second_preference = "precision"
    if min_recall_level is not None:

        warnings.filterwarnings(
            "ignore", category=pd.core.common.SettingWithCopyWarning
        )
        res_ok = res[res["recall"] >= min_recall_level]
        res_ok.sort_values(
            by=[f"f-{beta}", second_preference, "threshold"],
            ascending=[False, False, True],
            inplace=True,
        )

        res_not_ok = res[res["recall"] < min_recall_level]
        res_not_ok[f"f-{beta}"] = -1
        res_not_ok.sort_values(
            by=[second_preference, "threshold"], ascending=[False, True], inplace=True
        )

        res = pd.concat([res_ok, res_not_ok], ignore_index=True)
        warnings.filterwarnings(
            "default", category=pd.core.common.SettingWithCopyWarning
        )
    else:
        res.sort_values(
            by=[f"f-{beta}", second_preference, "threshold"],
            ascending=[False, False, True],
            inplace=True,
        )
        res.reset_index(drop=True, inplace=True)
    return res.loc[0, "threshold"], res


def train_pipeline(path_to_config: Union[str, dict], display_report=False):

    logger = setup_logging()

    logger.info("Training pipeline started")

    logger.info(f"Reading training pipeline parameters")
    training_params = load_training_params(path_to_config)

    logger.info(f"Parsing config")
    (
        path_dataset,
        path_to_models,
        path_to_report,
        splitting_params,
        feature_params,
        encoder_params,
        classifier_params,
    ) = parse_training_params(training_params)

    logger.info(f"Reading data from path: {path_dataset}")
    data = pd.read_csv(path_dataset)
    logger.info(f"Data shape: {data.shape}, Data columns: {list(data.columns)}")

    if not feature_params.target_name:
        raise CustomException("target name was not provided in config")
    if feature_params.target_name not in data.columns:
        raise CustomException(
            f"provided target name: {feature_params.target_name} is not in dataframe"
        )
    target_name = feature_params.target_name

    if not feature_params.exclude_features:
        exclude_features = [target_name]
    else:
        exclude_features = feature_params.exclude_features
        logger.info(f"Excluding following features from analysis: {exclude_features}")
        if target_name not in exclude_features:
            exclude_features.append(target_name)
    logger.info(f"Splitting data with params: {splitting_params}")
    data_train, data_test = make_split(data, target_name, splitting_params)
    logger.info(f"Train data: {data_train.shape}, Test data: {data_test.shape}")

    if not (feature_params.categorical_features and feature_params.numerical_features):
        logger.info(f"Recognizing features with train data")
        cat_features, num_features = recognize_features(
            data=data_train,
            target_name=target_name,
            exclude_features=exclude_features,
            verbose=False,
        )
        logger.info(
            f"Recognized features:\ncategorical: {cat_features}\nnumeric: {num_features}"
        )
    else:
        cat_features, num_features = (
            feature_params.categorical_features,
            feature_params.numerical_features,
        )
        logger.info(
            f"Features from config:\ncategorical: {cat_features}\nnumeric: {num_features}"
        )
    logger.info(f"Creating feature transformer pipelines with params: {encoder_params}")
    transformers = create_transformers_pipeline(create_encoder(encoder_params))

    logger.info(f"Fitting transformers and creating features with train data")
    X_train, y_train = get_xy(
        data=data_train,
        target_name=target_name,
        cat_features=cat_features,
        num_features=num_features,
        transformers=transformers,
        mode="fit",
    )
    logger.info(f"Features shape: {X_train.shape}, Target shape: {y_train.shape}")

    logger.info(f"Creating features for test data")
    X_test, y_test = get_xy(
        data=data_test,
        target_name=target_name,
        cat_features=cat_features,
        num_features=num_features,
        transformers=transformers,
        mode="transform",
    )
    logger.info(f"Features shape: {X_test.shape}, Target shape: {y_test.shape}")

    logger.info(f"Creating classifier with params: {classifier_params}")
    model = create_classifier(classifier_params)

    logger.info(f"Creating additional fitting params for classifier depending on type")
    fit_params = {}
    if (
        isinstance(model, LGBMClassifier)
        and encoder_params.encoder_type == "categorical"
    ):
        fit_params["categorical_feature"] = cat_features
    logger.info(f"Fitting params are: {fit_params}")

    logger.info(f"Fitting classifier")
    model.fit(X_train, y_train, **fit_params)

    logger.info(f"Determining optimal cutoff by maximizing f1-value")
    y_train_predicted = model.predict_proba(X_train)[:, 1]
    prc_results = calc_precision_recall_curve_binary(y_train, y_train_predicted)
    optimal_cutoff, _ = calc_optimal_cutoff(*prc_results)
    logger.info(f"Optimal cutoff is {optimal_cutoff:.4f}")

    logger.info(f"Generating report and saving to {path_to_report}")
    generate_report(
        X=X_test,
        y=y_test,
        model=model,
        cutoff=optimal_cutoff,
        path_to_report=path_to_report,
        display_report=display_report,
    )

    logger.info(f"Saving classifier to {path_to_models + 'classifier'}")
    save_object(model, path_to_models, "classifier", verbose=False)

    logger.info(f"Saving optimal cutoff to {path_to_models + 'optimal_cutoff'}")
    save_object(optimal_cutoff, path_to_models, "optimal_cutoff", verbose=False)

    logger.info(f"Saving transformation pipeline to {path_to_models + 'transformers'}")
    save_object(transformers, path_to_models, "transformers", verbose=False)

    logger.info(
        f"Saving initial categorical feature list to {path_to_models + 'cat_features'}"
    )
    save_object(cat_features, path_to_models, "cat_features", verbose=False)

    logger.info(
        f"Saving initial numerical feature list to {path_to_models + 'num_features'}"
    )
    save_object(num_features, path_to_models, "num_features", verbose=False)

    logger.info(f"Finished")
