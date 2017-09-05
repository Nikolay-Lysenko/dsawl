"""
The module provides estimators that have compatible with `sklearn` API.
These estimators are learnt with involvement of target-based
out-of-fold generated features (which can replace some of initial features).

@author: Nikolay Lysenko
"""
# TODO: Write more about out-of-fold feature generation.


from typing import List, Dict, Callable, Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from .target_based_features_creator import TargetBasedFeaturesCreator


class OutOfFoldFeaturesEstimator(BaseEstimator):
    """
    Parent class for regression and classification estimators.
    It should not be instantiated.

    :param estimator: internal estimator to be fitted
    :param n_splits: number of folds for feature generation
    :param shuffle: whether to shuffle objects before splitting
    :param random_state: pseudo-random numbers generator seed for shuffling
    :param aggregators: functions that compute aggregates
    """

    def __init__(
            self,
            estimator: BaseEstimator,
            n_splits: int,
            shuffle: bool = False,
            random_state: int = None,
            aggregators: List[Callable] = None
            ):
        self.estimator = estimator
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.aggregators = [np.mean] if aggregators is None else aggregators
        self.features_creator_ = None
        self.drop_source_features_ = None
        self.extended_X_ = None

    def __fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            source_positions: List[int],
            drop_source_features: bool = True,
            estimator_kwargs: Dict[Any, Any] = None,
            save_training_features_as_attr: bool = False
            ) -> 'OutOfFoldFeaturesEstimator':
        self.drop_source_features_ = drop_source_features
        self.features_creator_ = TargetBasedFeaturesCreator(self.aggregators)
        extended_X = self.features_creator_.fit_transform_out_of_fold(
            X,
            y,
            source_positions,
            self.n_splits,
            self.shuffle,
            self.random_state,
            drop_source_features
        )
        self.features_creator_.fit(X, y, source_positions)

        if estimator_kwargs is None:
            estimator_kwargs = dict()
        self.estimator.fit(extended_X, y, **estimator_kwargs)
        if save_training_features_as_attr:
            self.extended_X_ = extended_X
        return self

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            source_positions: List[int],
            drop_source_features: bool = True,
            estimator_kwargs: Dict[Any, Any] = None
            ) -> 'OutOfFoldFeaturesEstimator':
        """
        Fit estimator to a dataset where conditional aggregates of
        target variable are generated and used as features.

        Risk of overfitting is reduced, because for each object
        its own target is not used for generation of its features.

        :param X: features
        :param y: target
        :param source_positions: indices of initial features to be
                                 used as conditions
        :param drop_source_features: drop or keep those of initial
                                     features that are used for
                                     conditioning over them
        :param estimator_kwargs: settings of internal estimator fit
        :return: fitted estimator
        """
        self.__fit(X, y, source_positions, drop_source_features,
                   estimator_kwargs, save_training_features_as_attr=False)
        return self

    def predict(
            self,
            X: np.ndarray
            ) -> np.ndarray:
        """
        Make predictions for objects represented as `X`.

        Note that, ideally, `X` must not overlap with the sample
        that is used for fitting of the current instance, because
        else leakage of information about target occurs.
        If you need in predictions for learning sample, use
        `fit_predict` method.

        :param X: features of objects
        :return: predictions
        """
        extended_X = self.features_creator_.transform(
            X,
            self.drop_source_features_
        )
        predictions = self.estimator.predict(extended_X)
        return predictions

    def fit_predict(
            self,
            X: np.ndarray,
            y: np.ndarray,
            source_positions: List[int],
            drop_source_features: bool = True,
            estimator_kwargs: Dict[Any, Any] = None
            ) -> np.ndarray:
        """
        Train model and make predictions for the training set.

        :param X: features
        :param y: target
        :param source_positions: indices of initial features to be
                                 used as conditions
        :param drop_source_features: drop or keep those of initial
                                     features that are used for
                                     conditioning over them
        :param estimator_kwargs: settings of internal estimator fit
        :return: predictions
        """
        try:
            self.__fit(X, y, source_positions, drop_source_features,
                       estimator_kwargs, save_training_features_as_attr=True)
            predictions = self.estimator.predict(self.extended_X_)
            return predictions
        finally:
            self.extended_X_ = None


class OutOfFoldFeaturesRegressor(
        OutOfFoldFeaturesEstimator, RegressorMixin):
    """
    Regressor that has out-of-fold feature generation before
    training.
    """
    pass


class OutOfFoldFeaturesClassifier(
        OutOfFoldFeaturesEstimator, ClassifierMixin):
    """
    Classifier that has out-of-fold feature generation before
    training.
    """

    def predict_proba(
            self,
            X: np.ndarray
            ) -> np.ndarray:
        """
        Predict class probabilities for objects represented as `X`.

        If you need in probabilities' predictions for learning sample,
        use `fit_predict_proba` method.

        :param X: features of objects
        :return: predicted probabilities
        """
        if not hasattr(self.estimator, "predict_proba"):
            raise NotImplementedError("Estimator has not predict_proba method")
        extended_X = self.features_creator_.transform(
            X,
            self.drop_source_features_
        )
        predicted_probabilities = self.estimator.predict_proba(extended_X)
        return predicted_probabilities

    def fit_predict_proba(
            self,
            X: np.ndarray,
            y: np.ndarray,
            source_positions: List[int],
            drop_source_features: bool = True,
            estimator_kwargs: Dict[Any, Any] = None
            ) -> np.ndarray:
        """
        Train model and predict class probabilities for the
        training set.

        :param X: features
        :param y: target
        :param source_positions: indices of initial features to be
                                 used as conditions
        :param drop_source_features: drop or keep those of initial
                                     features that are used for
                                     conditioning over them
        :param estimator_kwargs: settings of internal estimator fit
        :return: predicted probabilities
        """
        if not hasattr(self.estimator, "predict_proba"):
            raise NotImplementedError("Estimator has not predict_proba method")
        try:
            self.__fit(X, y, source_positions, drop_source_features,
                       estimator_kwargs, save_training_features_as_attr=True)
            predicted_probabilities = \
                self.estimator.predict_proba(self.extended_X_)
            return predicted_probabilities
        finally:
            self.extended_X_ = None
