from typing import List, Dict, Set, Tuple, Union, Iterable, Optional
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm.notebook import tqdm


def seek_types(dataframe: pd.DataFrame) -> Dict[str, List[str]]:
    """
    returns dict with lists of cat_features, float_features, int_features, date_features, other_features

    Parameters
    ----------
    dataframe: pd.DataFrame
        dataframe, for which columns will by separated by their types

    Returns
    -------
    types_dictionary: Dict[str, List[str]]
        dictionary with lists of features by keys: 'cat', 'float', 'int', 'date', 'other'
    """

    def _get_global_type(t):
        if "obj" in str(t):
            return "cat"
        elif "float" in str(t):
            return "float"
        elif "int" in str(t):
            return "int"
        elif "date" in str(t):
            return "date"
        else:
            return "other"

    found_types = (
        dataframe.dtypes.apply(_get_global_type)
        .reset_index()
        .groupby(0)
        .agg(lambda x: list(x))
    )
    found_types = {k: v for k, v in zip(found_types.index, found_types["index"])}
    return found_types


def seek_categorical_from_integer_data(
    dataframe,
    int_features=[],
    map_binary_to_cat=True,
    N_threshold_low=20,
    N_threshold_high=100,
):
    cat_recognized = []

    if len(int_features) == 0:
        int_features = list(dataframe.columns)
    for f in int_features:
        unq = dataframe[f].unique()
        N_unq = len(unq)
        if N_unq > N_threshold_low:
            if sum(unq < 0) == 1 and N_unq <= N_threshold_high:
                ### one specific negative value (-1  or -99 or smth else)
                cat_recognized.append(f)
            else:
                ### really just integer feature (age for example) or very high cardinality
                pass
        else:
            if sum(unq > N_unq) >= 2:
                ### clearly doesn't lie in range(N_unq) - may be some city codes or smth
                cat_recognized.append(f)
            elif N_unq <= 3:
                ### binary or binary with missings
                if map_binary_to_cat:
                    cat_recognized.append(f)
            else:
                pass
    return cat_recognized


def explore_features(dataframe, target_name, features, figsize=(10, 5)):
    for f in tqdm(features):
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        _ = sns.scatterplot(
            data=dataframe,
            y=target_name,
            x=f,
            s=100,
            palette="coolwarm",
            ax=ax[0],
            alpha=0.1,
        )
        try:
            _ = sns.distplot(dataframe[f], ax=ax[1], color="darkorchid")
        except:
            ax[1].clear()
            _ = sns.distplot(dataframe[f], ax=ax[1], color="darkorchid", kde=False)
        ax[0].grid()
        ax[1].grid()


def calc_data_stats(dataframe: pd.DataFrame) -> pd.DataFrame:
    stats = []
    for col in tqdm(dataframe.columns):
        stats.append(
            (
                col,
                dataframe[col].nunique(),
                dataframe[col].isnull().sum() / dataframe.shape[0],
                dataframe[col].value_counts(normalize=True, dropna=False).values[0],
                dataframe[col].dtype,
            )
        )
    stats_dataframe = pd.DataFrame(
        stats,
        columns=[
            "Feature",
            "Unique_values",
            "Missing values ratio",
            "Largest category rate",
            "Type",
        ],
    )
    stats_dataframe.sort_values("Missing values ratio", ascending=False, inplace=True)
    return stats_dataframe


def extract_features(
    data: pd.DataFrame,
    cat_features: List[str],
    num_features: List[str],
    transformers: Dict[str, List],
    mode: "fit",
) -> pd.DataFrame:

    cat_transformers, num_transformers = transformers.get("cat", []), transformers.get(
        "num", []
    )
    cat_data, num_data = data[cat_features], data[num_features]

    processed_data_parts = []
    for data_part, tf_pipeline in zip(
        (cat_data, num_data), (cat_transformers, num_transformers)
    ):
        data_part_processed = deepcopy(data_part)
        for transformer in tf_pipeline:
            if mode == "fit":
                data_part_processed = transformer.fit_transform(data_part_processed)
            elif mode == "transform":
                data_part_processed = transformer.transform(data_part_processed)
            else:
                raise CustomException(
                    f"got unknown mode for transformers: {mode}, expected either 'fit' or 'transform'"
                )
        processed_data_parts.append(data_part_processed)
    X = pd.concat(processed_data_parts, axis=1)
    return X


def extract_target(dataframe: pd.DataFrame, target_name: str, drop=False) -> pd.Series:
    """
    extracts target from dataframe
    :param dataframe pd.DataFrame: input dataframe
    :param target_name str: name of target column
    :param drop bool: whether to drop target from dataframe after extraction, acts inplace
    :return: pd.Series
    """
    target = dataframe[target_name]

    if drop:
        dataframe.drop(target_name, axis=1, inplace=True)
    return target
