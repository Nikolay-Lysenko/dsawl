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


from typing import List, Union, Callable, Optional
from abc import ABC, abstractmethod

import numpy as np
import scipy

from sklearn.base import BaseEstimator, clone

from .utils import make_committee


def compute_confidences(predicted_probabilities: np.ndarray) -> np.ndarray:
    """
    Compute confidence of classifier on new objects.
    Here confidence on an object means predicted probability
    of the predicted class.

    :param predicted_probabilities:
        predicted by the classifier probabilities of classes for
        each of the new objects, shape = (n_new_objects, n_classes)
    :return:
        confidences of classifier on new objects
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
        each of the new objects, shape = (n_new_objects, n_classes)
    :return:
        margins on new objects
    """
    sorted_probabilities = np.sort(predicted_probabilities, axis=1)
    margins = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    return margins


def compute_entropy(predicted_probabilities: np.ndarray) -> np.ndarray:
    """
    Compute Shannon entropy of predicted probabilities for each
    of the new objects.

    :param predicted_probabilities:
        predicted by the classifier probabilities of classes for
        each of the new objects, shape = (n_new_objects, n_classes)
    :return:
        entropy on new objects
    """
    entropy = scipy.stats.entropy(predicted_probabilities.T)
    return entropy


def compute_committee_divergences(
        list_of_predicted_probabilities: List[np.ndarray]
        ) -> np.ndarray:
    """
    Compute a value that indicates how predicted by various classifiers
    probabilities differ from each other.
    Namely, this value is sum over all classifiers of Kullback-Leibler
    divergences between predicted by a classifier probabilities
    and consensus probabilities.

    :param list_of_predicted_probabilities:
        list such that its i-th element is predicted by the i-th
        classifier probabilities of classes for new objects;
        all elements have shape (n_new_objects, n_classes)
    :return:
        sums of Kullback-Leibler divergences
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
    Compute a value that indicates how predicted by various regressors
    values differ from each other. Namely, this value is variance.

    :param list_of_predictions:
        list such that its i-th element is predicted by the i-th
        regressor values; all elements have shape (n_new_objects,)
    :return:
        variance of predictions for each new object
    """
    variances = np.var(list_of_predictions, axis=1)
    return variances


def compute_estimations_of_variance(
        predictions: np.ndarray, predictions_of_square: np.ndarray
        ) -> np.ndarray:
    """
    Estimate variance assuming that one regressor predicts mean of
    target and another regressor predicts mean of squared target.

    :param predictions:
        estimations of mean of target on new objects,
        shape = (n_new_objects,)
    :param predictions_of_square:
        estimations of mean of squared target on new objects,
        shape = (n_new_objects,)
    :return:
        estimations of variance
    """
    estimations_of_variance = predictions_of_square - predictions ** 2
    estimations_of_variance = np.max(estimations_of_variance, 0)
    return estimations_of_variance


class BaseScorer(ABC):
    """
    A facade that provides unified interface for various functions
    that score objects from a pool based on usefulness of their
    labels.

    :param scoring_fn:
        function for scoring objects
    :param revert_sign:
        `False` if the most important object has the highest score
        and `True` else
    :param is_classification:
        `True` if it is classification or `False` else
    """

    def __init__(
            self,
            scoring_fn: Callable,
            revert_sign: bool = False,
            is_classification: bool = True
            ):
        self.scoring_fn = scoring_fn
        self.revert_sign = revert_sign
        self.is_classification = is_classification

    @staticmethod
    def _validate_inputs(
            X_train: Optional[np.ndarray], y_train: Optional[np.ndarray]
            ) -> type(None):
        if X_train is None:
            raise ValueError(
                "`X_train` must be passed if `skip_training` == `False`"
            )
        if y_train is None:
            raise ValueError(
                "`y_train` must be passed if `skip_training` == `False`"
            )

    @abstractmethod
    def score(
            self,
            est: BaseEstimator,
            X_new: np.ndarray,
            skip_training: bool = False,
            X_train: Optional[np.ndarray] = None,
            y_train: Optional[np.ndarray] = None,
            *args, **kwargs
            ) -> np.ndarray:
        pass


class UncertaintyScorerForClassification(BaseScorer):
    """
    A scorer working with functions that measure uncertainty in
    predicted class probabilities. Examples of such functions:
    * `compute_confidences`,
    * `compute_margins`,
    * `compute_entropy`.
    """

    def __init__(self, scoring_fn: Callable, revert_sign: bool = False):
        super().__init__(scoring_fn, revert_sign, is_classification=True)

    def score(
            self,
            est: BaseEstimator,
            X_new: np.ndarray,
            skip_training: bool = False,
            X_train: Optional[np.ndarray] = None,
            y_train: Optional[np.ndarray] = None,
            *args, **kwargs
            ) -> np.ndarray:
        """
        Score objects with the highest score standing for the most
        important object.

        :param est:
            classifier that has methods `fit` and `predict_proba`
        :param X_new:
            feature representation of new objects
        :param skip_training:
            `True` if `est` has been fitted already and `False` else
        :param X_train:
            feature representation of training objects
        :param y_train:
            target label
        :return:
            uncertainty scores computed with `self.scoring_fn`
        """
        if not skip_training:
            self._validate_inputs(X_train, y_train)
            est.fit(X_train, y_train)
        predicted_probabilities = est.predict_proba(X_new)
        scores = self.scoring_fn(predicted_probabilities)
        if self.revert_sign:
            scores = -scores
        return scores


class CommitteeScorer(BaseScorer):
    """
    A scorer working with functions that measure degree of disagreement
    in predictions of committee members. Examples of such functions:
    * `compute_committee_divergences`,
    * `compute_committee_variances`.
    """

    def score(
            self,
            est: Union[BaseEstimator, List[BaseEstimator]],
            X_new: np.ndarray,
            skip_training: bool = False,
            X_train: Optional[np.ndarray] = None,
            y_train: Optional[np.ndarray] = None,
            *args, **kwargs
            ) -> np.ndarray:
        """
        Score objects with the highest score standing for the most
        important object.

        :param est:
            estimator if `skip_training` is set to `False` and list of
            already fitted estimators else; method `fit` must be present,
            also method `predict_proba` must be present if it is
            classifier(s) and method `predict` must be present if it is
            regressor(s)
        :param X_new:
            feature representation of new objects
        :param skip_training:
            `True` if `est` is a committee of fitted estimators and
            `False` else
        :param X_train:
            feature representation of training objects
        :param y_train:
            target
        :return:
            discrepancy scores computed with `self.scoring_fn`
        """
        if not skip_training:
            self._validate_inputs(X_train, y_train)
            committee = make_committee(est, X_train, y_train, *args, **kwargs)
        else:
            committee = est
        if self.is_classification:
            list_of_predictions = [
                est.predict_proba(X_new) for est in committee
            ]
        else:
            list_of_predictions = [
                est.predict(X_new) for est in committee
            ]
        scores = self.scoring_fn(list_of_predictions)
        if self.revert_sign:
            scores = -scores
        return scores


class VarianceScorerForRegression(BaseScorer):
    """
    A scorer working with functions that measure estimated variance.
    Examples of such functions:
    * `compute_estimations_of_variance`.
    """

    def __init__(self, scoring_fn: Callable):
        super().__init__(
            scoring_fn, revert_sign=False, is_classification=False
        )

    def score(
            self,
            est: BaseEstimator,
            X_new: np.ndarray,
            skip_training: bool = False,
            X_train: Optional[np.ndarray] = None,
            y_train: Optional[np.ndarray] = None,
            *args, **kwargs
            ) -> np.ndarray:
        """
        Score objects with the highest score standing for the most
        important object.

        :param est:
            regressor that has methods `fit` and `predict`, it must be
            already fitted if at least `X_train` or `y_train` is not
            passed
        :param X_new:
            feature representation of new objects
        :param skip_training:
            `True` if `est` is has been fitted already and `False` else,
            training of squared target prediction is not affected by
            this argument
        :param X_train:
            feature representation of training objects
        :param y_train:
            target variable
        :return:
            estimates of variance computed with `self.scoring_fn`
        """
        if X_train is None or y_train is None:
            raise ValueError("Both `X_train` and `y_train` must be passed")
        if not skip_training:
            est.fit(X_train, y_train)
        predictions = est.predict(X_new)
        second_est = clone(est)
        second_est.fit(X_train, y_train ** 2)
        predictions_of_square = second_est.predict(X_new)
        scores = self.scoring_fn(predictions, predictions_of_square)
        return scores
