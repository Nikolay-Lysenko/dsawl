"""
This file contains highly-modular tools for pool-based approach
to active learning.

Active learning setup assumes that, given a model and a training set,
it is possible to extend the training set with new labelled examples
and the goal is to do it with maximum possible improvement of model
quality subject to constraint on how many new examples can be added.
Further, pool-bases sampling means that new examples come from
a fixed and known set of initially unlabelled examples, i.e., the task
is to choose objects to be studied, not to synthesize them arbitrarily.

@author: Nikolay Lysenko
"""


from typing import List, Dict, Union, Callable, Optional
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import scipy
from sklearn.base import BaseEstimator, clone

from .utils import make_committee


ToolsType = Union[BaseEstimator, List[BaseEstimator], Dict[str, BaseEstimator]]


# Scoring functions.

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
    Compute a value that indicates how predicted by various classifiers
    probabilities differ from each other.
    Namely, this value is sum over all classifiers of Kullback-Leibler
    divergences between predicted by a classifier probabilities
    and consensus probabilities (i.e., averaged probabilities).

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
    Compute a value that indicates how predicted by various regressors
    values differ from each other. Namely, this value at an object
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


# Scorers.

class BaseScorer(ABC):
    """
    A facade that provides unified interface for various functions
    that score objects from a pool by usefulness of their labels.

    :param scoring_fn:
        function for scoring objects
    :param revert_sign:
        `False` if the most important object has the highest score
        and `True` else
    :param is_classification:
        `True` if classification problem is studied and `False` else
    """

    def __init__(
            self,
            scoring_fn: Callable,
            revert_sign: bool = False,
            is_classification: bool = True
            ):
        self._scoring_fn = scoring_fn
        self._revert_sign = revert_sign
        self._is_classification = is_classification

    @abstractmethod
    def get_tools(self) -> ToolsType:
        pass

    @abstractmethod
    def set_tools(self, tools: ToolsType) -> type(None):
        pass

    @abstractmethod
    def update_tools(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            est: Optional[BaseEstimator] = None,
            *args, **kwargs
            ) -> type(None):
        pass

    @abstractmethod
    def score(self, X_new: np.ndarray) -> np.ndarray:
        pass


class UncertaintyScorerForClassification(BaseScorer):
    """
    A scorer working with functions that measure uncertainty in
    predicted class probabilities. Examples of such functions:
    * `compute_confidences`,
    * `compute_margins`,
    * `compute_entropy`.

    :param scoring_fn:
        function for scoring objects
    :param revert_sign:
        `False` if the most important object has the highest score
        and `True` else
    :param clf:
        classifier that has methods `fit` and `predict_proba`,
        it becomes internal classifier of the scorer; it is recommended
        to wrap `clf` in `sklearn.calibration.CalibratedClassifierCV`
        if it does not predict well-calibrated probabilities by default
    """

    def __init__(
            self,
            scoring_fn: Callable,
            revert_sign: bool = False,
            clf: Optional[BaseEstimator] = None,
            ):
        super().__init__(scoring_fn, revert_sign, is_classification=True)
        self.__clf = clf

    def __check_classifier_before_scoring(self) -> type(None):
        # Check that classifier is passed and has proper methods.
        if self.__clf is None:
            raise RuntimeError("Classifier must be passed before scoring.")
        if not hasattr(self.__clf, 'predict_proba'):
            raise ValueError("Classifier must have `predict_proba` method.")

    def __check_classifier_before_update(
            self,
            clf: Optional[BaseEstimator] = None
            ) -> type(None):
        # Check that classifier to be used has proper methods.
        if clf is not None:
            clf_to_be_used = clf
        else:
            clf_to_be_used = self.__clf
        if not hasattr(clf_to_be_used, 'fit'):
            raise ValueError("Classifier must have `fit` method.")

    def get_tools(self) -> BaseEstimator:
        """
        Get internal classifier.

        :return:
            internal classifier
        """
        return self.__clf

    def set_tools(self, tools: BaseEstimator) -> type(None):
        """
        Replace internal classifier with passed instance.

        :param tools:
            classifier that has methods `fit` and `predict_proba`,
            it is assumed that `predict_proba` method returns
            well-calibrated probabilities
        :return:
            None
        """
        self.__clf = tools

    def update_tools(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            est: Optional[BaseEstimator] = None,
            *args, **kwargs
            ) -> type(None):
        """
        Fit internal classifier to passed training data and,
        optionally, before that replace classifier with a new
        instance.

        :param X_train:
            feature representation of training objects
        :param y_train:
            target labels
        :param est:
            classifier that has methods `fit` and `predict_proba`;
            if it is passed, its fitted instance becomes internal
            classifier, so it is assumed that `predict_proba` method
            returns well-calibrated probabilities
        :return:
            None
        """
        if est is not None:
            self.__check_classifier_before_update(est)
            self.__clf = est.fit(X_train, y_train)
        elif self.__clf is not None:
            self.__check_classifier_before_update()
            self.__clf.fit(X_train, y_train)
        else:
            raise RuntimeError(
                "Classifier is not passed neither to initialization "
                "nor to this function."
            )

    def score(self, X_new: np.ndarray) -> np.ndarray:
        """
        Score new objects with the highest score standing for the most
        important object.

        :param X_new:
            feature representation of new objects
        :return:
            uncertainty scores computed with `self.scoring_fn`
        """
        self.__check_classifier_before_scoring()
        predicted_probabilities = self.__clf.predict_proba(X_new)
        scores = self._scoring_fn(predicted_probabilities)
        if self._revert_sign:
            scores = -scores
        return scores


class CommitteeScorer(BaseScorer):
    """
    A scorer working with functions that measure degree of disagreement
    in predictions of committee members. Examples of such functions:
    * `compute_committee_divergences`,
    * `compute_committee_variances`.

    :param scoring_fn:
        function for scoring objects
    :param revert_sign:
        `False` if the most important object has the highest score
        and `True` else
    :param is_classification:
        `True` if it is classification or `False` if it is regression
    :param committee:
        list of instances of the same class fitted to different folds,
        instances must have `predict_proba` method if it is
        classification or `predict` method if it is regression;
        if it is classification, it is assumed that `predict_proba`
        returns well-calibrated probabilities, so consider to wrap
        instances in `sklearn.calibration.CalibratedClassifierCV`
        if they do not predict well-calibrated probabilities
        by default
    """

    def __init__(
            self,
            scoring_fn: Callable,
            revert_sign: bool = False,
            is_classification: bool = True,
            committee: Optional[List[BaseEstimator]] = None,
            ):
        super().__init__(scoring_fn, revert_sign, is_classification)
        self.__committee = committee

    def __check_committee_before_scoring(self) -> type(None):
        # Check that committee is not empty.
        if self.__committee is None:
            raise RuntimeError("Committee must be provided before scoring.")
        if len(self.__committee) == 0:
            raise RuntimeError("Committee has zero length.")

    def __check_estimator_before_update(
            self,
            est: Optional[BaseEstimator] = None
            ) -> type(None):
        # Check that estimator to be cloned for committee has proper methods.
        if est is None:
            est_to_be_cloned = (
                self.__committee[0] if len(self.__committee) > 0 else None
            )
        else:
            est_to_be_cloned = est
        if est_to_be_cloned is None:
            raise RuntimeError("Committee has zero length.")
        if not hasattr(est_to_be_cloned, 'fit'):
            raise ValueError("Estimator must have `fit` method.")

    def get_tools(self) -> List[BaseEstimator]:
        """
        Get internal committee of estimators.

        :return:
            None
        """
        return self.__committee

    def set_tools(self, tools: List[BaseEstimator]) -> type(None):
        """
        Replace internal committee with passed list of estimators.

        :param tools:
            list of instances of the same class fitted to different
            folds, instances must have `predict_proba` method if it
            is classification or `predict` method if it is regression;
            if it is classification, it is assumed that `predict_proba`
            returns well-calibrated probabilities
        :return:
            None
        """
        self.__committee = tools

    def update_tools(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            est: Optional[BaseEstimator] = None,
            *args, **kwargs
            ) -> type(None):
        """
        Fit internal committee to passed training data and,
        optionally, before that replace members of the committee
        with new instances.

        :param X_train:
            feature representation of training objects
        :param y_train:
            target
        :param est:
            estimator that has method `fit`, also it must have method
            `predict_proba` if it is a classifier or it must have
            method `predict` if it is a regressor; if it is passed,
            its clones become members of the committee instead of
            previous members, so it is also assumed that
            `predict_proba` method returns well-calibrated
            probabilities
        :return:
            None
        """
        if est is not None:
            self.__check_estimator_before_update(est)
            self.__committee = make_committee(
                est, X_train, y_train, *args, **kwargs
            )
        elif self.__committee is not None:
            self.__check_estimator_before_update()
            self.__committee = make_committee(
                self.__committee[0], X_train, y_train, *args, **kwargs
            )
        else:
            raise RuntimeError(
                "Committee is not passed neither to initialization "
                "nor to this function."
            )

    def score(self, X_new: np.ndarray) -> np.ndarray:
        """
        Score new objects with the highest score standing for the most
        important object.

        :param X_new:
            feature representation of new objects
        :return:
            discrepancy scores computed with `self.scoring_fn`
        """
        self.__check_committee_before_scoring()
        if self._is_classification:
            list_of_predictions = [
                est.predict_proba(X_new) for est in self.__committee
            ]
        else:
            list_of_predictions = [
                est.predict(X_new) for est in self.__committee
            ]
        scores = self._scoring_fn(list_of_predictions)
        if self._revert_sign:
            scores = -scores  # pragma: no cover
        return scores


class VarianceScorerForRegression(BaseScorer):
    """
    A scorer working with functions that measure estimated variance.
    Examples of such functions:
    * `compute_estimations_of_variance`.

    :param scoring_fn:
        function for scoring objects
    :param rgrs:
        dict with keys 'target' and 'target^2' and values that
        are regressors predicting target variable and squared
        target variable respectively, these regressors must
        have `predict` method for doing so
    """

    def __init__(
            self,
            scoring_fn: Callable,
            rgrs: Optional[Dict[str, BaseEstimator]] = None
            ):
        super().__init__(
            scoring_fn, revert_sign=False, is_classification=False
        )
        self.__rgrs = rgrs

    def __check_regressors_before_scoring(self) -> type(None):
        # Check that regressors are passed and have proper methods.
        if self.__rgrs is None:
            raise RuntimeError("Regressors must be passed before scoring.")
        for key in ['target', 'target^2']:
            if self.__rgrs.get(key, None) is None:
                raise ValueError("{key} must be a key.".format(key=key))
            if not hasattr(self.__rgrs[key], 'predict'):
                raise ValueError("Regressor must have `predict` method.")

    def __check_regressor_before_update(
            self,
            rgr: Optional[BaseEstimator] = None
            ) -> type(None):
        # Check that estimator to be cloned for committee has proper methods.
        if rgr is None:
            first_member = self.__rgrs.get('target', None)
            if first_member is None:
                raise ValueError("Key 'target' is missed.")
            second_member = self.__rgrs.get('target^2', None)
            if second_member is None:
                raise ValueError("Key 'target^2' is missed.")
            rgr_to_be_cloned = first_member
        else:
            rgr_to_be_cloned = rgr
        if not hasattr(rgr_to_be_cloned, 'fit'):
            raise ValueError("Regressor must have `fit` method.")

    def get_tools(self) -> Dict[str, BaseEstimator]:
        """
        Get internal pair of regressors.

        :return:
            internal pair of regressors
        """
        return self.__rgrs

    def set_tools(self, tools: Dict[str, BaseEstimator]) -> type(None):
        """
        Replace internal regressors with passed regressors.

        :param tools:
            dict with keys 'target' and 'target^2' and values that
            are regressors predicting target variable and squared
            target variable respectively, these regressors must
            have method `predict` for doing so
        :return:
            None
        """
        self.__rgrs = tools

    def update_tools(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            rgr: Optional[BaseEstimator] = None,
            *args, **kwargs
            ) -> type(None):
        """
        Fit pair of regressors to passed training data and,
        optionally, before that replace these regressors with new
        instances.

        :param X_train:
            feature representation of training objects
        :param y_train:
            target variable
        :param rgr:
            regressor that has methods `fit` and `predict`; if it
            is passed, it and its clone form a new pair of regressors
        :return:
            None
        """
        if rgr is not None:
            self.__check_regressor_before_update(rgr)
            self.__rgrs = {
                'target': rgr.fit(X_train, y_train),
                'target^2': clone(rgr).fit(X_train, y_train ** 2)
            }
        elif self.__rgrs is not None:
            self.__check_regressor_before_update()
            self.__rgrs = {
                'target': self.__rgrs['target'].fit(X_train, y_train),
                'target^2': self.__rgrs['target^2'].fit(X_train, y_train ** 2)
            }
        else:
            raise RuntimeError(
                "Regressors is not passed neither to initialization "
                "nor to this function."
            )

    def score(self, X_new: np.ndarray) -> np.ndarray:
        """
        Score new objects with the highest score standing for the most
        important object.

        :param X_new:
            feature representation of new objects
        :return:
            estimates of variance computed with `self.scoring_fn`
        """
        self.__check_regressors_before_scoring()
        predictions = self.__rgrs['target'].predict(X_new)
        predictions_of_square = self.__rgrs['target^2'].predict(X_new)
        scores = self._scoring_fn(predictions, predictions_of_square)
        return scores


# Active learning strategies.

class EpsilonGreedyPickerFromPool:
    """
    This class is for picking a random object with a specified
    probability (so called epsilon) or picking object near the
    decision boundary else.

    :param scorer:
        scorer for ranking new objects, it also can be one
        of these strings: 'confidence', 'margin', 'entropy',
        'divergence', 'predictions_variance', 'target_variance'
    :param exploration_probability:
        probability of picking objects at random; if it is a float,
        this value is used always, and if it is a list of floats,
        its i-th element is used when `pick_new_objects` method is
        called for the i-th time, so you can use exploration schedule
        with exploration probability decreasing over time.
    """

    def __init__(
            self,
            scorer: Union[str, BaseScorer],
            exploration_probability: Union[float, List[float]] = 0.1
            ):
        str_to_scorer = defaultdict(
            lambda: scorer,
            confidence=UncertaintyScorerForClassification(
                compute_confidences, revert_sign=True
            ),
            margin=UncertaintyScorerForClassification(
                compute_margins, revert_sign=True
            ),
            entropy=UncertaintyScorerForClassification(
                compute_entropy
            ),
            divergence=CommitteeScorer(
                compute_committee_divergences
            ),
            predictions_variance=CommitteeScorer(
                compute_committee_variances, is_classification=False
            ),
            target_variance=VarianceScorerForRegression(
                compute_estimations_of_variance
            )
        )
        scorer = str_to_scorer[scorer]
        self.__scorer = scorer
        self.__exploration_probability = exploration_probability
        self.n_to_pick = None

    def __exploit(self, X_new: np.ndarray) -> List[int]:
        # Exploit existing knowledge, i.e., pick objects near the current
        # decision boundary and return their indices.
        scores = self.__scorer.score(X_new)
        picked_indices = scores.argsort()[-self.n_to_pick:].tolist()
        return picked_indices

    def __explore(self, n_of_new_objects: int) -> List[int]:
        # Pick objects at random.
        all_indices = np.array(range(n_of_new_objects))
        picked_indices = np.random.choice(
            all_indices, size=self.n_to_pick, replace=False
        )
        return picked_indices

    def __get_current_exploration_probability(self) -> float:
        # Get exploration probability for the current call of
        # `pick_new_objects`.
        if isinstance(self.__exploration_probability, (float, int)):
            return self.__exploration_probability
        elif len(self.__exploration_probability) == 0:
            raise StopIteration("All exploration probabilities are popped.")
        else:
            return self.__exploration_probability.pop(0)

    def pick_new_objects(
            self,
            X_new: np.ndarray,
            n_to_pick: int = 1
            ) -> List[int]:
        """
        Select objects from a fixed pool of objects.

        :param X_new:
            feature representation of new objects
        :param n_to_pick:
            number of objects to pick
        :return:
            indices of the most important objects
        """
        self.n_to_pick = n_to_pick
        outcome = np.random.uniform()
        exploration_probability = self.__get_current_exploration_probability()
        if outcome > exploration_probability:
            picked_indices = self.__exploit(X_new)
        else:
            picked_indices = self.__explore(X_new.shape[0])
        return picked_indices

    def get_tools(self) -> ToolsType:
        """
        Get estimator or ensemble of estimators such that it is used
        for scoring new objects by usefulness of their labels.

        :return:
            internal tools of `self.__scorer`
        """
        return self.__scorer.get_tools()

    def set_tools(self, tools: ToolsType) -> type(None):
        """
        Replace internal tools of scorer with the passed tools.

        :param tools:
            new internal tools of scorer
        :return:
            None
        """
        self.__scorer.set_tools(tools)

    def update_tools(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            est: Optional[BaseEstimator] = None,
            *args, **kwargs
            ) -> type(None):
        """
        Fit internal tools of scorer to passed training data and,
        optionally, before that replace these tools with a new ones
        based on the passed instance of `est`.

        :param X_train:
            feature representation of training objects
        :param y_train:
            target labels
        :param est:
            instance such that new tools are based on it (e.g..,
            if `self.__scorer` is instance of `CommitteeScorer`,
            committee of `est` clones fitted to different folds
            becomes a tools of `self.__scorer`)
        :return:
            None
        """
        self.__scorer.update_tools(X_train, y_train, est, *args, **kwargs)
