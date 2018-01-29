"""
This module consists of highly-modular tools for pool-based approach
to active learning. Active learning setup assumes that, given a model
and a training set, it is possible to extend the training set with
new labelled examples and the goal is to do it with maximum possible
improvement of model quality. Further, pool-bases sampling means that
new examples come from a fixed and known set of initially unlabelled
examples, i.e., the task is to choose which labels should be explored
and disclosed.

@author: Nikolay Lysenko
"""


from typing import List, Optional


import numpy as np
import scipy

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold

from dsawl.stacking.stackers import FoldType


def make_committee(
        est: BaseEstimator,
        X_train: np.ndarray, y_train: np.ndarray,
        splitter: Optional[FoldType] = None
        ) -> List[BaseEstimator]:
    """
    Make committee from a single estimator by fitting it to
    various folds.

    :param est:
        estimator instance that has method `fit`
    :param X_train:
        feature representation of training objects
    :param y_train:
        target label
    :param splitter:
        instance that can split data into folds
    :return:
        list of fitted instances
    """
    committee = []
    splitter = splitter or StratifiedKFold()
    for train_index, test_index in splitter.split(X_train, y_train):
        X_curr_train = X_train[train_index]
        y_curr_train = y_train[train_index]
        curr_est = clone(est).fit(X_curr_train, y_curr_train)
        committee.append(curr_est)
    return committee


def compute_confidences(
        clf: BaseEstimator,
        X_train: np.ndarray, y_train: np.ndarray,
        X_new: np.ndarray
        ) -> np.ndarray:
    """
    Compute confidence of classifier on new objects.
    Here confidence on an object means predicted probability
    of the predicted class.

    :param clf:
        classifier instance that has methods `fit` and `predict_proba`
    :param X_train:
        feature representation of training objects
    :param y_train:
        target label
    :param X_new:
        feature representation of new objects
    :return:
        confidences of classifier on new objects
    """
    clf.fit(X_train, y_train)
    predicted_probabilities = clf.predict_proba(X_new)
    confidences = np.max(predicted_probabilities, axis=1).reshape((-1, 1))
    return confidences


def compute_margins(
        clf: BaseEstimator,
        X_train: np.ndarray, y_train: np.ndarray,
        X_new: np.ndarray
        ) -> np.ndarray:
    """
    Compute margins of predicted by classifier labels.
    Here margin means the difference between predicted probability
    of the predicted class and predicted probability of the second
    best class.

    :param clf:
        classifier instance that has methods `fit` and `predict_proba`
    :param X_train:
        feature representation of training objects
    :param y_train:
        target label
    :param X_new:
        feature representation of new objects
    :return:
        margins on new objects
    """
    clf.fit(X_train, y_train)
    predicted_probabilities = clf.predict_proba(X_new)
    sorted_probabilities = np.sort(predicted_probabilities, axis=1)
    margins = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    margins = margins.reshape((-1, 1))
    return margins


def compute_entropy(
        clf: BaseEstimator,
        X_train: np.ndarray, y_train: np.ndarray,
        X_new: np.ndarray
        ) -> np.ndarray:
    """
    Compute Shannon entropy of predicted probabilities for each
    of the new objects.

    :param clf:
        classifier instance that has methods `fit` and `predict_proba`
    :param X_train:
        feature representation of training objects
    :param y_train:
        target label
    :param X_new:
        feature representation of new objects
    :return:
        entropy on new objects
    """
    clf.fit(X_train, y_train)
    predicted_probabilities = clf.predict_proba(X_new)
    entropy = scipy.stats.entropy(predicted_probabilities.T).reshape((-1, 1))
    return entropy


def compute_committee_divergences(
        clf: BaseEstimator,
        X_train: np.ndarray, y_train: np.ndarray,
        X_new: np.ndarray,
        splitter: Optional[FoldType] = None
        ) -> np.ndarray:
    """
    Compute a value that indicates how predictions of various fits
    differ from each other.
    Namely, this value is sum over all fits of Kullback-Leibler
    divergences between individual predicted probabilities
    and consensus probabilities.

    :param clf:
        classifier instance that has methods `fit` and `predict_proba`
    :param X_train:
        feature representation of training objects
    :param y_train:
        target label
    :param X_new:
        feature representation of new objects
    :param splitter:
        instance that can split data into folds
    :return:
        sums of Kullback-Leibler divergences
    """
    committee = make_committee(clf, X_train, y_train, splitter)
    all_probabilities = [est.predict_proba(X_new) for est in committee]
    consensus_probabilities = sum(all_probabilities) / len(all_probabilities)
    list_of_divergences = []
    for predicted_probabilities in all_probabilities:
        curr_divergences = scipy.stats.entropy(
            predicted_probabilities.T, consensus_probabilities.T
        ).reshape(-1, 1)
        list_of_divergences.append(curr_divergences)
    divergences = sum(list_of_divergences)
    return divergences
