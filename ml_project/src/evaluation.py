import warnings
import os
from pprint import pprint

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from IPython.display import display
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.utils import check_consistent_length, column_or_1d, assert_all_finite
from sklearn.utils.extmath import stable_cumsum
from matplotlib.collections import QuadMesh

from .utils import make_binary_prediction, save_metrics


def calc_gini(y_true, y_score):
    return 2 * roc_auc_score(y_true, y_score) - 1


def calc_permutation_importances(
    model,
    X,
    y_true,
    metric,
    columns=None,
    n_shuffles=10,
    seed=8,
    prediction_method="proba",
    verbose=False,
):
    """
    calculates permutation importances for a given `model`,
    using features data in `X` and true target in `y_true`
    `metric` is a function which takes (y_true, y_pred)
    `columns` is a list of columns for which importance will be calculated
    prediction_method should correspond to a provided `metric`
    """
    if prediction_method in ["proba", "binary"]:
        predict_function = lambda X: model.predict_proba(X)[:, 1]
    elif prediction_method in ["predict", "label", "value"]:
        predict_function = lambda X: model.predict(X)
    else:
        raise ValueError("wrong prediction method")
    baseline = metric(y_true, predict_function(X))
    if verbose:
        print("baseline value: {}\n-------".format(baseline))
    imps = []
    np.random.seed(seed)
    warnings.filterwarnings("ignore", category=pd.core.common.SettingWithCopyWarning)

    if columns is not None and len(columns) > 0 and not isinstance(columns, str):
        col_iterator = columns
    else:
        col_iterator = X.columns
    # for col in tqdm(col_iterator):
    for col in col_iterator:
        m = 0
        for _ in range(n_shuffles):
            saved = X[col].copy()
            X[col] = np.random.permutation(X[col])
            m += metric(y_true, predict_function(X))
            X[col] = saved
        m /= n_shuffles
        imp = baseline - m
        imps.append(imp)
        if verbose:
            print("{0} importance: {1}\n-------".format(col, imp))
    warnings.filterwarnings("default", category=pd.core.common.SettingWithCopyWarning)
    imps = pd.DataFrame(
        data=imps, columns=["Permutation Importance"], index=col_iterator
    )
    imps.sort_values(axis=0, ascending=False, by="Permutation Importance", inplace=True)
    return imps


def plot_permutation_importances(
    importances, n_feat_to_plot=15, plot_type="bar", fontsize=15, title=""
):
    """
    plots results of `calc_permutation_importances` function call
    """

    n_feat_to_plot = min(importances.shape[0], n_feat_to_plot)
    if plot_type == "bar":
        fig = plt.figure(figsize=(20, 10))
        sns.barplot(
            x="Permutation Importance",
            y=importances.index[:n_feat_to_plot],
            data=importances[:n_feat_to_plot],
            color="royalblue",
        )
        plt.xlabel("")
        plt.yticks(fontsize=fontsize)
    else:
        fig = plt.figure(figsize=(20, 10))
        plt.plot(
            list(range(n_feat_to_plot)),
            importances[:n_feat_to_plot],
            color="royalblue",
            marker="o",
            linewidth=3,
        )
        plt.xticks(list(range(n_feat_to_plot)), importances.index[:n_feat_to_plot])
        plt.grid(axis="y")
        plt.xlim(-0.5, n_feat_to_plot - 0.5)
        plt.xticks(rotation=90, fontsize=fontsize)
    sns.set_style("dark")
    plt.title("Permutation Importances {}".format(title), fontsize=fontsize)
    plt.close(fig)
    plt.ioff()

    return fig


def get_confusion_matrix_advanced(
    y_true,
    y_pred,
    classes_dict=None,
    figsize=(16, 10),
    annot_kws={"fontsize": 18},
    xticklabels={"rotation": "horizontal", "fontsize": 18},
    yticklabels={"rotation": "horizontal", "fontsize": 18},
):

    """
    plots confusion matrix for multiclass task with additional informations by classes
    """

    conf_matrix = pd.DataFrame(confusion_matrix(y_true=y_true, y_pred=y_pred))
    plt.figure(figsize=(1, 1))
    palette = sns.color_palette(palette="Blues", n_colors=5)
    ax_init = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=palette)
    plt.close()
    quadmesh = ax_init.findobj(QuadMesh)[0]
    init_facecolors = quadmesh.get_facecolors()

    df_matrix = pd.concat(
        [pd.DataFrame(conf_matrix.sum(axis=0)).transpose(), conf_matrix], axis=0
    )
    df_matrix.index = ["Recognised"] + [*df_matrix.index[1:]]
    df_matrix["Support"] = df_matrix.sum(axis=1)

    fig = plt.figure(figsize=(16, 10))
    ax = sns.heatmap(df_matrix, annot=True, fmt="d", cmap=palette, annot_kws=annot_kws)
    ax.set_xlabel("Predicted", fontsize=15)
    ax.set_ylabel("True", rotation="horizontal", fontsize=15)

    present_classes = sorted(set(y_true).union(y_pred))
    classes_cnt = len(present_classes)
    present_dict = dict(
        enumerate(present_classes)
    )  ### from 0..N-1 index to label in vectors

    if classes_dict is not None:
        if classes_dict == "keep":
            ### keeping what comes
            mapping_dict = present_dict
        else:
            ### verify that we get from classes_dict only classes which are in true or predicted
            mapping_dict = {k: classes_dict[present_dict[k]] for k in present_dict}
    else:
        ### creating 0..N-1 names
        mapping_dict = {l1: l1 for l1 in np.arange(classes_cnt)}
    _ = ax.set_yticks(np.arange(classes_cnt + 1) + 0.5)
    _ = ax.set_xticks(np.arange(classes_cnt + 1) + 0.5)
    _ = ax.set_yticklabels(
        labels=["Recognised as"] + list(mapping_dict.values()), **yticklabels
    )
    _ = ax.set_xticklabels(
        labels=list(mapping_dict.values()) + ["Support"], **xticklabels
    )

    quadmesh = ax.findobj(QuadMesh)[0]

    facecolors = quadmesh.get_facecolors()  # ячейки по строкам

    facecolors[:classes_cnt] = np.array([0.0, 128 / 255, 135 / 255, 1])
    facecolors[np.arange(classes_cnt, len(facecolors), classes_cnt + 1)] = np.array(
        [0.0, 200 / 255, 127 / 255, 1]
    )
    facecolors[classes_cnt] = np.array([0.0, 0.0, 0.0, 1])

    for i in range(0, classes_cnt + 1):
        facecolors[
            ((i + 1) * (classes_cnt + 1)) : (i + 2) * (classes_cnt + 1) - 1
        ] = init_facecolors[i * classes_cnt : (i + 1) * classes_cnt]
    quadmesh.set_facecolors(facecolors)
    plt.tight_layout()
    plt.ioff()
    plt.close(fig)

    conf_matrix.index = mapping_dict.values()
    conf_matrix.columns = mapping_dict.values()

    return conf_matrix, fig


def calc_precision_recall_curve_binary(
    y_true,
    y_score,
    sample_weight=None,
    use_grid=False,
    grid_step=0.01,
    thresholds_grid=None,
    use_sklearn_bounds=False,
):
    """
    changed sklearn version to one where it's possible to set your own thresholds
    """
    check_consistent_length(y_true, y_score)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)

    assert_all_finite(y_true)
    assert_all_finite(y_score)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        assert_all_finite(sample_weight)
    # make y_true a boolean vector
    y_true = y_true == 1.0

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        check_consistent_length(y_true, sample_weight)
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.0
    if use_grid:
        if thresholds_grid is not None:
            thresholds_grid = np.asarray(np.sort(thresholds_grid)[::-1])
        else:
            thresholds_grid = np.arange(0.0, 1.0, grid_step)[
                ::-1
            ]  ### making grid with provided step
        threshold_idxs = []
        for t in thresholds_grid:
            mask = y_score >= t
            if all(mask):
                threshold_idxs.append(y_true.size - 1)
            else:
                threshold_idxs.append(np.max([np.argmin(mask) - 1, 0]))
        threshold_idxs = np.asarray(threshold_idxs)
    else:
        # y_score typically has many tied values. Here we extract
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = np.where(np.diff(y_score))[
            0
        ]  ### contains ix if s[ix+1] - s[ix] != 0
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[
        threshold_idxs
    ]  ### == cumsum on target only in points where it changed
    if sample_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = (
            1 + threshold_idxs - tps
        )  ### amount on every step minus tps, as every step is PredPositive corresponding to threshold
    if use_grid:
        thresholds = thresholds_grid
    else:
        thresholds = y_score[threshold_idxs]
    # fps : A count of false positives, at index i being the number of negative samples assigned a score >= thresholds[i].
    # The total number of negative samples is equal to fps[-1] (thus true negatives are given by fps[-1] - fps).

    # tps: An increasing count of true positives, at index i being the number of positive samples assigned a score >= thresholds[i]
    # The total number of positive samples is equal to tps[-1] (thus false negatives are given by tps[-1] - tps).

    # thresholds: Decreasing score values.

    # The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    # true positives and ``fp`` the number of false positives. The precision is
    # intuitively the ability of the classifier not to label as positive a sample
    # that is negative.

    # The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    # true positives and ``fn`` the number of false negatives. The recall is
    # intuitively the ability of the classifier to find all the positive samples.

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0  ## don't really get when that's gonna happen
    recall = tps / tps[-1]

    if use_sklearn_bounds:
        last_ind = tps.searchsorted(tps[-1])  # stop when full recall attained
    else:
        last_ind = len(tps) - 1
    sl = slice(last_ind, None, -1)  # and reverse the outputs so recall is decreasing

    # The last precision and recall values are 1. and 0. respectively and do not
    # have a corresponding threshold.  This ensures that the graph starts on the y axis.
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def get_classification_report(y_true, y_predicted, digits=4, **kwargs):

    class_report = classification_report(
        y_true, y_predicted, digits=digits, output_dict=True
    )
    target_class_metrics = class_report["1"]
    target_class_metrics["accuracy"] = class_report["accuracy"]
    target_class_metrics.pop("support")

    _, fig = get_confusion_matrix_advanced(y_true, y_predicted, **kwargs)

    return target_class_metrics, fig


def generate_report(X, y, model, cutoff, path_to_report=None, display_report=True):
    permutation_importances = calc_permutation_importances(
        model=model,
        n_shuffles=3,
        X=X,
        y_true=y,
        metric=calc_gini,
        prediction_method="proba",
        verbose=False,
    )

    feature_importances_figure = plot_permutation_importances(permutation_importances)

    y_predicted = model.predict_proba(X)[:, 1]
    y_binary = make_binary_prediction(model, X, cutoff)

    metrics, conf_matrix = get_classification_report(y, y_binary)
    metrics["gini"] = calc_gini(y, y_predicted)

    if path_to_report:
        feature_importances_figure.savefig(
            path_to_report + os.sep + "test_feature_importances"
        )
        conf_matrix.savefig(path_to_report + os.sep + "test_confusion_matrix")
        save_metrics(metrics, path_to_report + os.sep + "test_metrics.json")
    if display_report:
        display(feature_importances_figure)
        display(conf_matrix)
        pprint(metrics)
        print(classification_report(y, y_binary, digits=4, output_dict=False))
