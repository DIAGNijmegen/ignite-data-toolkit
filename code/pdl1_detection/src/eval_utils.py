from typing import List, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import BallTree

EPSILON = 1e-9
PLOT_ALIASES = {
    "reader P1": "$P_1$",
    "reader P2": "$P_2$",
    "reader P3": "$P_3$",
    "reader P4": "$P_4$",
    "reader P5": "$P_5$",
    "reader P6": "$P_6$",
    "reader R1": "$R_i$",
    "reader R2": "$R_{ii}$",
    "reader R3": "$R_{iii}$",
    "reader R4": "$R_{iv}$",
}


def calculate_confusion_matrix(
    *,
    ground_truth: List[Tuple[float, ...]],
    predictions: List[Tuple[float, ...]],
    n_classes: int,
    radius: float = 1.0,
) -> np.array:
    """
    Generates a confusion matrix describing the number of true positives,
    false positives and false negatives for the ground truth points
    given the predicted points and classes.

    If multiple predicted points hit one ground truth point then this is
    considered as 1 true positive, and 0 false negatives.
    If one predicted point is a hit for N ground truth points then this is
    considered as 1 true positive, and N-1 false negatives.
    Parameters
    ----------
    ground_truth
        A list of the ground truth points with corresponding class
    predictions
        A list of the predicted points with corresponding class
    radius
        The maximum distance that two points can be separated by in order to
        be considered a hit
    Returns
    -------
    A tuple containing the number of true positives, false positives and
    false negatives.
    """

    if not n_classes > 0:
        raise ValueError("Number of classes must a positive integer.")

    confusion_matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=np.uint16)
    if len(ground_truth) == 0:
        for point in predictions:
            confusion_matrix[-1, int(point[2])] += 1
        return confusion_matrix
    elif len(predictions) == 0:
        for point in ground_truth:
            confusion_matrix[int(point[2]), -1] += 1
        return confusion_matrix

    hits_for_targets = _find_hits_for_targets(
        targets=np.array(ground_truth)[
            ..., :2
        ],  # Only calculate hits on basis of x,y coords, not class
        predictions=np.array(predictions)[..., :2],
        radius=radius,
    )

    prediction_hit_a_target = np.zeros(len(predictions), dtype=bool)
    target_hit_a_prediction = np.zeros(len(ground_truth), dtype=bool)

    for i, (hits_for_target, target) in enumerate(zip(hits_for_targets, ground_truth)):
        for hit_idx in hits_for_target:
            if not prediction_hit_a_target[hit_idx] and not target_hit_a_prediction[i]:
                prediction_hit_a_target[hit_idx] = True
                target_hit_a_prediction[i] = True
                confusion_matrix[int(target[2]), int(predictions[hit_idx][2])] += 1
                break

    prediction_no_hit_a_target = 1 - prediction_hit_a_target
    idxs = np.argwhere(prediction_no_hit_a_target)
    false_positives = np.transpose(np.array(predictions)[idxs.flatten()])[2]
    fp_class, fp_counts = np.unique(false_positives, return_counts=True)

    for label, count in zip(fp_class, fp_counts):
        confusion_matrix[-1, int(label)] = count

    target_no_hit_a_prediction = 1 - target_hit_a_prediction
    idxs = np.argwhere(target_no_hit_a_prediction)
    false_negatives = np.transpose(np.array(ground_truth)[idxs.flatten()])[2]
    fn_class, fn_counts = np.unique(false_negatives, return_counts=True)

    for label, count in zip(fn_class.astype(int), fn_counts):
        confusion_matrix[int(label), -1] = count

    if not (
        sum(confusion_matrix[:, -1]) == len(false_negatives)
        and sum(confusion_matrix[-1, :]) == len(false_positives)
    ):
        print("fn_class:", fn_class)
        print("fn_counts:", fn_counts)
        print("sum fn counts:", sum(fn_counts))
        print("len ground truth:", len(ground_truth))

        print()
        print("fp_class:", fp_class)
        print("fp_counts:", fp_counts)
        print("sum fp_counts:", sum(fp_counts))
        print("len predictions:", len(predictions))

        print()
        print("tp_counts:", sum(target_hit_a_prediction))
        print("tp_counts:", sum(prediction_hit_a_target))

        print("confusion_matrix:", confusion_matrix)

        raise RuntimeError("Did not find correct sum for false negatives/positives")

    if not all(
        [
            confusion_matrix[:-1].sum() == len(ground_truth),
            confusion_matrix[:, :-1].sum() == len(predictions),
        ]
    ):
        print("len gt:", len(ground_truth))
        print("len yhat:", len(predictions))
        print("sum pred hit target:", prediction_hit_a_target.count(False))
        print("len pred hit target:", len(prediction_hit_a_target))
        print("pred:\n", predictions)
        print("pred hit target:\n", prediction_hit_a_target)
        print(confusion_matrix)
        print(
            [
                confusion_matrix[1:, :].sum() == len(ground_truth),
                confusion_matrix[:, 1:].sum() == len(predictions),
            ]
        )
        raise RuntimeError("Validation failed...")

    return confusion_matrix


def _find_hits_for_targets(
    *,
    targets: List[Tuple[float, ...]],
    predictions: List[Tuple[float, ...]],
    radius: float,
) -> List[Tuple[int, ...]]:
    """
    Generates a list of the predicted points that are within a radius r of the
    targets. The indicies are returned in sorted order, from closest to
    farthest point.
    Parameters
    ----------
    targets
        A list of target points
    predictions
        A list of predicted points
    radius
        The maximum distance that two points can be apart for them to be
        considered a hit
    Returns
    -------
    A list which has the same length as the targets list. Each element within
    this list contains another list that contains the indicies of the
    predictions that are considered hits.
    """
    predictions_tree = BallTree(np.array(predictions))
    hits, _ = predictions_tree.query_radius(
        X=targets, r=radius, return_distance=True, sort_results=True
    )
    return hits


def _get_recall_per_class(cf):
    recalls = np.zeros(cf.shape[0])
    samples = np.sum(cf, axis=1)
    for i in range(recalls.shape[0]):
        recalls[i] = cf[i, i] / (samples[i] + EPSILON)
    return recalls


def _get_f1s_per_class(precisions, recalls):
    f1s = np.zeros(precisions.shape[0])

    for i in range(f1s.shape[0]):
        f1s[i] = (2 * precisions[i] * recalls[i]) / (
            precisions[i] + recalls[i] + EPSILON
        )

    return f1s


def _get_precision_per_class(cf):
    precisions = np.zeros(cf.shape[0])
    n_predicted = np.sum(cf, axis=0)
    for i in range(precisions.shape[0]):
        precisions[i] = cf[i, i] / (n_predicted[i] + EPSILON)
    return precisions


def _get_macro_average(list_of_metric_outputs):
    return float(np.mean(list_of_metric_outputs))


def get_metrics(confusion_matrix, labels):
    outputs = {}
    precisions = _get_precision_per_class(confusion_matrix)
    recalls = _get_recall_per_class(confusion_matrix)
    f1s = _get_f1s_per_class(precisions, recalls)

    for i, label in enumerate(labels):
        outputs[f"precision: {label}"] = precisions[i]
        outputs[f"recall: {label}"] = recalls[i]
        outputs[f"f1: {label}"] = f1s[i]

    outputs["precision macro"] = _get_macro_average(precisions[:-1])
    outputs["recall macro"] = _get_macro_average(recalls[:-1])
    outputs["f1 macro"] = _get_macro_average(f1s[:-1])

    return outputs


def plot_f1_matrix(
    df_f1,
    readers,
    output_path,
    title="F1 scores per reader",
    cmap=plt.get_cmap("Blues"),
):
    triu_mask = np.triu(np.ones((len(df_f1), len(df_f1))))
    inv_diag_mask = ~np.eye(len(df_f1)).astype(bool)  # Fancy diagonal
    ax = plt.axes()
    sns.heatmap(
        df_f1.fillna(0),
        vmin=0.0,
        vmax=1.0,
        annot=True,
        annot_kws={"size": 20},
        fmt=".2f",
        cmap=cmap,
        mask=triu_mask,
        cbar=False,
        linewidth=0.5,
        square=True,
    )
    sns.heatmap(
        df_f1.fillna(0.1),  # Fancy diagonal
        vmin=0.0,
        vmax=1.0,
        mask=inv_diag_mask,
        cmap=plt.get_cmap("Greys"),
        cbar=False,
        linewidth=0.5,
        square=True,
    )

    ax.set_title(title, fontdict={"size": 20, "weight": "bold"}, pad=10)
    tick_fontdict = {"fontsize": 20}
    ticks = [PLOT_ALIASES.get(r, r) for r in readers]
    ax.set_xticklabels(ticks, rotation=0, fontdict=tick_fontdict)
    ax.set_yticklabels(ticks, rotation=0, fontdict=tick_fontdict)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
