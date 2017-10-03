"""
The module provides estimators that have `sklearn`-like API.
Main `sklearn` utilities like grid search cross-validation and
pipelines are supported.
The estimators are learnt with involvement of target-based
out-of-fold generated features (which can replace some of
initial features).
Fitting is implemented in a way that leads to more realistic
cross-validation scores in comparison with plain (i.e., bulk, in-fold)
generation of features that are aggregates of target value.

@author: Nikolay Lysenko
"""


from typing import List, Dict, Callable, Union, Any, Optional

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit
)

from .target_encoder import TargetEncoder


class BaseOutOfFoldTargetEncodingEstimator(BaseEstimator):
    """
    Parent class for regression and classification estimators.
    It should not be instantiated.

    :param estimator:
        internal estimator to be fitted
    :param estimator_kwargs:
        (hyper)parameters of internal estimator
    :param splitter:
        object that splits data into folds, default schema is
        Leave-One-Out
    :param aggregators:
        functions that compute aggregates
    :param smoothing_strength:
        strength of smoothing towards unconditional aggregates for
        target-based features creation (aka target encoding)
    :param min_frequency:
        minimal number of occurrences of a feature's value (if value
        occurs less times than this parameter, this value is mapped to
        unconditional aggregate)
    :param drop_source_features:
        drop or keep at training stage those of initial features
        that are used for conditioning over them at new features'
        generation stage
    """

    def __init__(
            self,
            estimator: BaseEstimator,
            estimator_kwargs: Optional[Dict] = None,
            splitter: Optional[Union[
                KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit
            ]] = None,
            aggregators: Optional[List[Callable]] = None,
            smoothing_strength: Optional[float] = 0,
            min_frequency: Optional[int] = 1,
            drop_source_features: Optional[bool] = True
            ):
        self._can_this_class_have_any_instances()
        self.estimator = estimator
        if estimator_kwargs is None:
            estimator_kwargs = dict()
        self.estimator.set_params(**estimator_kwargs)
        self.splitter = splitter
        self.aggregators = [np.mean] if aggregators is None else aggregators
        self.smoothing_strength = smoothing_strength
        self.min_frequency = min_frequency
        self.drop_source_features = drop_source_features
        self.target_encoder_ = None
        self.extended_X_ = None

    @classmethod
    def _can_this_class_have_any_instances(cls):
        # Make this class abstract.
        raise TypeError('{} must not have any instances.'.format(cls))

    def _fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            source_positions: List[int],
            fit_kwargs: Optional[Dict[Any, Any]] = None,
            save_training_features_as_attr: Optional[bool] = False
            ) -> 'BaseOutOfFoldTargetEncodingEstimator':
        # Run all internal logic of fitting.

        self.target_encoder_ = TargetEncoder(
            self.aggregators,
            self.splitter,
            self.smoothing_strength,
            self.min_frequency,
            self.drop_source_features
        )
        extended_X = self.target_encoder_.fit_transform_out_of_fold(
            X,
            y,
            source_positions
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
            fit_kwargs: Optional[Dict[Any, Any]] = None
            ) -> 'BaseOutOfFoldTargetEncodingEstimator':
        """
        Fit estimator to a dataset where conditional aggregates of
        target variable are generated and used as features.

        Risk of overfitting is reduced, because for each object
        its own target is not used for generation of its new features.

        :param X:
            features
        :param y:
            target
        :param source_positions:
            indices of initial features to be used as conditions
        :param fit_kwargs:
            settings of internal estimator fit
        :return:
            fitted estimator (instance of the class)
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

        :param X:
            features of objects
        :return:
            predictions
        """
        if self.target_encoder_ is None:
            raise RuntimeError("Estimator must be trained before predicting")
        extended_X = self.target_encoder_.transform(X)
        predictions = self.estimator.predict(extended_X)
        return predictions

    def fit_predict(
            self,
            X: np.ndarray,
            y: np.ndarray,
            source_positions: List[int],
            fit_kwargs: Optional[Dict[Any, Any]] = None
            ) -> np.ndarray:
        """
        Train model and make predictions for the training set.

        :param X:
            features
        :param y:
            target
        :param source_positions:
            indices of initial features to be used as conditions
        :param fit_kwargs:
            settings of internal estimator fit
        :return:
            predictions
        """
        try:
            self._fit(X, y, source_positions, fit_kwargs,
                      save_training_features_as_attr=True)
            predictions = self.estimator.predict(self.extended_X_)
            return predictions
        finally:
            self.extended_X_ = None


class OutOfFoldTargetEncodingRegressor(
        BaseOutOfFoldTargetEncodingEstimator, RegressorMixin
        ):
    """
    Regressor that has out-of-fold generation of target-based
    features before training.
    """

    @classmethod
    def _can_this_class_have_any_instances(cls):
        # Allow this class to have instances.
        pass


class OutOfFoldTargetEncodingClassifier(
        BaseOutOfFoldTargetEncodingEstimator, ClassifierMixin
        ):
    """
    Classifier that has out-of-fold generation of target-based
    features before training.
    """

    @classmethod
    def _can_this_class_have_any_instances(cls):
        # Allow this class to have instances.
        pass

    def predict_proba(
            self,
            X: np.ndarray
            ) -> np.ndarray:
        """
        Predict class probabilities for objects represented as `X`.

        If you need in probabilities' predictions for learning sample,
        use `fit_predict_proba` method.

        :param X:
            features of objects
        :return:
            predicted probabilities
        """
        if not hasattr(self.estimator, "predict_proba"):
            raise NotImplementedError(
                "Internal estimator has not predict_proba method"
            )
        if self.target_encoder_ is None:
            raise RuntimeError("Estimator must be trained before predicting")
        extended_X = self.target_encoder_.transform(X)
        predicted_probabilities = self.estimator.predict_proba(extended_X)
        return predicted_probabilities

    def fit_predict_proba(
            self,
            X: np.ndarray,
            y: np.ndarray,
            source_positions: List[int],
            fit_kwargs: Optional[Dict[Any, Any]] = None
            ) -> np.ndarray:
        """
        Train model and predict class probabilities for the
        training set.

        :param X:
            features
        :param y:
            target
        :param source_positions:
            indices of initial features to be used as conditions
        :param fit_kwargs:
            settings of internal estimator fit
        :return:
            predicted probabilities
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
