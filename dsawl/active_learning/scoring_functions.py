"""
@author: Nikolay Lysenko
"""


from typing import List

import numpy as np
import scipy


def compute_confidences(predicted_probabilities: np.ndarray) -> np.ndarray:
    """
    Compute confidences of classifier at new objects.
    Here confidence at an object means predicted probability
    of the predicted class.

    :param predicted_probabilities:
        predicted by the classifier probabilities of classes for
        each of the new objects, shape = (n_new_objects, n_classes);
        it is recommended to pass calibrated probabilities
    :return:
        confidences of classifier at new objects,
        shape = (n_new_objects,)
    """
    confidences = np.max(predicted_probabilities, axis=1)
    return confidences


def compute_margins(predicted_probabilities: np.ndarray) -> np.ndarray:
    """
    Compute margins of predicted by classifier labels.
    Here margin means the difference between predicted probability
    of the predicted class and predicted probability of the second
    best class.

    :param predicted_probabilities:
        predicted by the classifier probabilities of classes for
        each of the new objects, shape = (n_new_objects, n_classes);
        it is recommended to pass calibrated probabilities
    :return:
        margins of predicted labels at new objects,
        shape = (n_new_objects,)
    """
    sorted_probabilities = np.sort(predicted_probabilities, axis=1)
    margins = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    return margins


def compute_entropy(predicted_probabilities: np.ndarray) -> np.ndarray:
    """
    Compute Shannon entropy of predicted class label distribution
    for each of the new objects.

    :param predicted_probabilities:
        predicted by the classifier probabilities of classes for
        each of the new objects, shape = (n_new_objects, n_classes);
        it is recommended to pass calibrated probabilities
    :return:
        entropy of predictions at new objects,
        shape = (n_new_objects,)
    """
    entropy = scipy.stats.entropy(predicted_probabilities.T)
    return entropy


def compute_committee_divergences(
        list_of_predicted_probabilities: List[np.ndarray]
        ) -> np.ndarray:
    """
    Compute values that indicate how predicted by various classifiers
    probabilities differ from each other.
    Namely, the value for an object is sum over all classifiers of
    Kullback-Leibler divergences between predicted by a classifier
    probabilities and consensus probabilities (i.e., averaged
    probabilities).

    :param list_of_predicted_probabilities:
        list such that its i-th element is predicted by the i-th
        classifier probabilities of classes for new objects;
        all elements have shape (n_new_objects, n_classes);
        it is recommended to calibrate probabilities
    :return:
        sums of Kullback-Leibler divergences,
        shape = (n_new_objects,)
    """
    summed_probabilities = sum(list_of_predicted_probabilities)
    committee_size = len(list_of_predicted_probabilities)
    consensus_probabilities = summed_probabilities / committee_size
    list_of_divergences = []
    for predicted_probabilities in list_of_predicted_probabilities:
        curr_divergences = scipy.stats.entropy(
            predicted_probabilities.T, consensus_probabilities.T
        )
        list_of_divergences.append(curr_divergences)
    divergences = sum(list_of_divergences)
    return divergences


def compute_committee_variances(
        list_of_predictions: List[np.ndarray]
        ) -> np.ndarray:
    """
    Compute values that indicate how predicted by various regressors
    values differ from each other. Namely, the value for an object
    is variance of predictions made for this object.

    :param list_of_predictions:
        list such that its i-th element is predicted by the i-th
        regressor values; all elements have shape (n_new_objects,)
    :return:
        variance of predictions for each new object,
        shape = (n_new_objects,)
    """
    all_predictions = np.hstack(
        [np.array(x).reshape(-1, 1) for x in list_of_predictions]
    )
    variances = np.var(all_predictions, axis=1)
    return variances


def compute_estimations_of_variance(
        predictions: np.ndarray, predictions_of_square: np.ndarray
        ) -> np.ndarray:
    """
    Estimate variance of target variable assuming that one regressor
    predicts mean of the target and another regressor predicts mean of
    the squared target.

    :param predictions:
        estimations of mean of target at new objects,
        shape = (n_new_objects,)
    :param predictions_of_square:
        estimations of mean of squared target at new objects,
        shape = (n_new_objects,)
    :return:
        estimations of target variable variance,
        shape = (n_new_objects,)
    """
    estimations_of_variance = predictions_of_square - predictions ** 2
    estimations_of_variance = np.maximum(estimations_of_variance, 0)
    return estimations_of_variance
