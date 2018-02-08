"""
@author: Nikolay Lysenko
"""


from typing import List, Dict, Union, Callable, Optional
from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.mixture.base import BaseMixture

from .utils import make_committee


ToolsType = Union[
    BaseEstimator, List[BaseEstimator], Dict[str, BaseEstimator],
    BaseMixture
]


class BaseScorer(ABC):
    """
    A facade that provides unified interface for various functions
    that score objects by usefulness of their labels.

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
    def get_tools(self) -> Optional[ToolsType]:
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


class RandomScorer(BaseScorer):
    """
    A scorer that scores objects randomly.
    It is needed for making exploratory actions.
    """

    def __init__(self):
        super().__init__(dummy_fn)

    def get_tools(self) -> type(None):
        """
        Get `None` as `RandomScorer` has no tools.

        :return:
            None
        """
        return None

    def set_tools(self, tools: BaseEstimator) -> type(None):
        """
        Do nothing as `RandomScorer` has no tools.

        :param tools:
            anything, its value is not used
        :return:
            None
        """
        return

    def update_tools(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            est: Optional[BaseEstimator] = None,
            *args, **kwargs
            ) -> type(None):
        """
        Do nothing as `RandomScorer` has no tools.

        :param X_train:
            anything, its value is not used
        :param y_train:
            anything, its value is not used
        :param est:
            anything, its value is not used
        :return:
            None
        """
        return

    def score(self, X_new: np.ndarray) -> np.ndarray:
        """
        Score new objects with the highest score standing for the most
        important object.

        :param X_new:
            feature representation of new objects
        :return:
            random values
        """
        scores = np.random.uniform(size=X_new.shape[0])
        return scores


class DensityScorer(BaseScorer):
    """
    A scorer that ranks objects by density estimations. The higher
    density is, the lower object is ranked.
    This scorer is needed for making exploratory actions, because
    it selects objects that looks like outliers.

    :param est:
        density estimator that has methods `fit` and `score_samples`,
        it becomes internal density estimator of the scorer
    """

    def __init__(
            self,
            est: Optional[Union[BaseEstimator, BaseMixture]] = None,
            ):
        super().__init__(dummy_fn)
        self.__est = est

    def __check_estimator_before_scoring(self) -> type(None):
        # Check that estimator is passed and has proper methods.
        if self.__est is None:
            raise RuntimeError("Estimator must be passed before scoring.")
        if not hasattr(self.__est, 'score_samples'):
            raise ValueError("Estimator must have `score` method.")

    def __check_estimator_before_update(
            self,
            est: Optional[Union[BaseEstimator, BaseMixture]] = None
            ) -> type(None):
        # Check that estimator to be used has proper methods.
        if est is not None:
            est_to_be_used = est
        else:
            est_to_be_used = self.__est
        if not hasattr(est_to_be_used, 'fit'):
            raise ValueError("Estimator must have `fit` method.")

    def get_tools(self) -> Union[BaseEstimator, BaseMixture]:
        """
        Get internal density estimator.

        :return:
            internal density estimator
        """
        return self.__est

    def set_tools(
            self, tools: Union[BaseEstimator, BaseMixture]
            ) -> type(None):
        """
        Replace internal density estimator with passed instance.

        :param tools:
            density estimator that has methods `fit` and
            `score_samples`
        :return:
            None
        """
        self.__est = tools

    def update_tools(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            est: Optional[Union[BaseEstimator, BaseMixture]] = None,
            *args, **kwargs
            ) -> type(None):
        """
        Fit internal density estimator to passed training data and,
        optionally, before that replace density estimator with a new
        instance.

        :param X_train:
            feature representation of training objects
        :param y_train:
            target labels
        :param est:
            density estimator that has methods `fit` and
            `score_samples`; if it is passed, its fitted instance
            becomes internal density estimator
        :return:
            None
        """
        if est is not None:
            self.__check_estimator_before_update(est)
            self.__est = est.fit(X_train, y_train)
        elif self.__est is not None:
            self.__check_estimator_before_update()
            self.__est.fit(X_train, y_train)
        else:
            raise RuntimeError(
                "Estimator is not passed neither to initialization "
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
        self.__check_estimator_before_scoring()
        scores = -self.__est.score_samples(X_new)
        return scores


def dummy_fn() -> type(None):
    """
    A placeholder of a function.

    :return:
        None
    """
    pass
