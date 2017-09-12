"""
The module provides estimators that have compatible with `sklearn` API.
These estimators are learnt with involvement of target-based
out-of-fold generated features (which can replace some of
initial features).
Fitting is implemented in a way that leads to more realistic
cross-validation scores in comparison with plain generation of features
that are aggregates of target value.

@author: Nikolay Lysenko
"""


from typing import List, Dict, Callable, Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from .target_based_features_creator import TargetBasedFeaturesCreator


class BaseOutOfFoldFeaturesEstimator(BaseEstimator):
    """
    Parent class for regression and classification estimators.
    It should not be instantiated.

    :param estimator: internal estimator to be fitted
    :param estimator_kwargs: parameters of internal estimator
    :param n_splits: number of folds for feature generation
    :param shuffle: whether to shuffle objects before splitting
    :param random_state: pseudo-random numbers generator seed for
                         shuffling only, not for training
    :param aggregators: functions that compute aggregates
    :param smoothing_strength: strength of smoothing towards
                               unconditional aggregates for
                               target-based features creation
    :param min_frequency: minimal number of occurrences of a feature's
                          value (if value occurs less times than this
                          parameter, this value is mapped to
                          unconditional aggregate)
    :param drop_source_features: drop or keep at training stage
                                 those of initial features that are
                                 used for conditioning over them
                                 at new features generation stage
    """

    def __init__(
            self,
            estimator: BaseEstimator,
            estimator_kwargs: Dict,
            n_splits: int,
            shuffle: bool = False,
            random_state: int = None,
            aggregators: List[Callable] = None,
            smoothing_strength: float = 0,
            min_frequency: int = 1,
            drop_source_features: bool = True
            ):
        self.estimator = estimator
        self.estimator.set_params(**estimator_kwargs)
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.aggregators = [np.mean] if aggregators is None else aggregators
        self.smoothing_strength = smoothing_strength
        self.min_frequency = min_frequency
        self.drop_source_features = drop_source_features
        self.features_creator_ = None
        self.extended_X_ = None

    def _fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            source_positions: List[int],
            fit_kwargs: Dict[Any, Any] = None,
            save_training_features_as_attr: bool = False
            ) -> 'BaseOutOfFoldFeaturesEstimator':
        self.features_creator_ = TargetBasedFeaturesCreator(
            self.aggregators,
            self.smoothing_strength,
            self.min_frequency,
            self.drop_source_features
        )
        extended_X = self.features_creator_.fit_transform_out_of_fold(
            X,
            y,
            source_positions,
            self.n_splits,
            self.shuffle,
            self.random_state
        )

        fit_kwargs = dict() if fit_kwargs is None else fit_kwargs
        self.estimator.fit(extended_X, y, **fit_kwargs)
        if save_training_features_as_attr:
            self.extended_X_ = extended_X
        return self

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            source_positions: List[int],
            fit_kwargs: Dict[Any, Any] = None
            ) -> 'BaseOutOfFoldFeaturesEstimator':
        """
        Fit estimator to a dataset where conditional aggregates of
        target variable are generated and used as features.

        Risk of overfitting is reduced, because for each object
        its own target is not used for generation of its features.

        :param X: features
        :param y: target
        :param source_positions: indices of initial features to be
                                 used as conditions
        :param fit_kwargs: settings of internal estimator fit
        :return: fitted estimator
        """
        self._fit(X, y, source_positions, fit_kwargs,
                  save_training_features_as_attr=False)
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
        if self.features_creator_ is None:
            raise RuntimeError("Estimator must be trained before predicting")
        extended_X = self.features_creator_.transform(X)
        predictions = self.estimator.predict(extended_X)
        return predictions

    def fit_predict(
            self,
            X: np.ndarray,
            y: np.ndarray,
            source_positions: List[int],
            fit_kwargs: Dict[Any, Any] = None
            ) -> np.ndarray:
        """
        Train model and make predictions for the training set.

        :param X: features
        :param y: target
        :param source_positions: indices of initial features to be
                                 used as conditions
        :param fit_kwargs: settings of internal estimator fit
        :return: predictions
        """
        try:
            self._fit(X, y, source_positions, fit_kwargs,
                      save_training_features_as_attr=True)
            predictions = self.estimator.predict(self.extended_X_)
            return predictions
        finally:
            self.extended_X_ = None


class OutOfFoldFeaturesRegressor(
        BaseOutOfFoldFeaturesEstimator, RegressorMixin):
    """
    Regressor that has out-of-fold feature generation before
    training.
    """
    pass


class OutOfFoldFeaturesClassifier(
        BaseOutOfFoldFeaturesEstimator, ClassifierMixin):
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
            raise NotImplementedError(
                "Internal estimator has not predict_proba method"
            )
        if self.features_creator_ is None:
            raise RuntimeError("Estimator must be trained before predicting")
        extended_X = self.features_creator_.transform(X)
        predicted_probabilities = self.estimator.predict_proba(extended_X)
        return predicted_probabilities

    def fit_predict_proba(
            self,
            X: np.ndarray,
            y: np.ndarray,
            source_positions: List[int],
            fit_kwargs: Dict[Any, Any] = None
            ) -> np.ndarray:
        """
        Train model and predict class probabilities for the
        training set.

        :param X: features
        :param y: target
        :param source_positions: indices of initial features to be
                                 used as conditions
        :param fit_kwargs: settings of internal estimator fit
        :return: predicted probabilities
        """
        if not hasattr(self.estimator, "predict_proba"):
            raise NotImplementedError(
                "Internal estimator has not predict_proba method"
            )
        try:
            self._fit(X, y, source_positions, fit_kwargs,
                      save_training_features_as_attr=True)
            predicted_probabilities = \
                self.estimator.predict_proba(self.extended_X_)
            return predicted_probabilities
        finally:
            self.extended_X_ = None
