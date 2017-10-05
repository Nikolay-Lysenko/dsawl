"""
The module provides estimators that have compatible with
`sklearn` API.
The estimators are learnt with involvement of target-based
out-of-fold generated features (which can replace some of
initial features).
Fitting is implemented in a way that leads to more realistic
cross-validation scores in comparison with plain (i.e., bulk, in-fold)
generation of features that are aggregates of target value.

@author: Nikolay Lysenko
"""


from typing import List, Dict, Tuple, Callable, Union, Any, Optional

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder

from .target_encoder import TargetEncoder, FoldType


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
        functions that compute aggregates, default is mean function
    :param smoothing_strength:
        strength of smoothing towards unconditional aggregates for
        target-based features creation (aka target encoding),
        by default there is no smoothing
    :param min_frequency:
        minimal number of occurrences of a feature's value (if value
        occurs less times than this parameter, this value is mapped to
        unconditional aggregate), by default it is 1
    :param drop_source_features:
        to drop or to keep at training stage those of initial features
        that are used for conditioning over them at new features'
        generation stage, default is to drop
    """

    def __init__(
            self,
            estimator: Optional[BaseEstimator] = None,
            estimator_kwargs: Optional[Dict] = None,
            splitter: Optional[FoldType] = None,
            aggregators: Optional[List[Callable]] = None,
            smoothing_strength: Optional[float] = 0,
            min_frequency: Optional[int] = 1,
            drop_source_features: Optional[bool] = True
            ):
        self._can_this_class_have_any_instances()
        self.estimator = estimator
        self.estimator_kwargs = estimator_kwargs
        self.splitter = splitter
        self.aggregators = aggregators
        self.smoothing_strength = smoothing_strength
        self.min_frequency = min_frequency
        self.drop_source_features = drop_source_features
        self._extended_X = None

    @classmethod
    def _can_this_class_have_any_instances(cls):
        # Make this class abstract.
        raise TypeError('{} must not have any instances.'.format(cls))

    def _set_internal_estimator(self):
        # Instantiate nested estimator from initialization parameters.
        pass

    def _do_supplementary_preparations(
            self,
            X: np.ndarray,
            y: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray]:
        # Transform `X` and `y` specially for regression or classification.
        return X, y

    def _fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            source_positions: Optional[List[int]] = None,
            fit_kwargs: Optional[Dict[Any, Any]] = None,
            save_training_features_as_attr: Optional[bool] = False
            ) -> 'BaseOutOfFoldTargetEncodingEstimator':
        # Run all internal logic of fitting.

        X, y = check_X_y(X, y)
        X_, y_ = self._do_supplementary_preparations(X, y)

        self.estimator_ = self._set_internal_estimator()
        self.target_encoder_ = TargetEncoder(
            self.aggregators,
            self.splitter,
            self.smoothing_strength,
            self.min_frequency,
            self.drop_source_features
        )
        extended_X = self.target_encoder_.fit_transform_out_of_fold(
            X,
            y_,
            source_positions
        )

        fit_kwargs = fit_kwargs or dict()
        self.estimator_.fit(extended_X, y, **fit_kwargs)
        if save_training_features_as_attr:
            self._extended_X = extended_X
        return self

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            source_positions: Optional[List[int]] = None,
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
            indices of initial features to be used as conditions,
            default is the last one
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
        check_is_fitted(self, ['target_encoder_'])
        X = check_array(X)

        extended_X = self.target_encoder_.transform(X)
        predictions = self.estimator_.predict(extended_X)
        return predictions

    def fit_predict(
            self,
            X: np.ndarray,
            y: np.ndarray,
            source_positions: Optional[List[int]] = None,
            fit_kwargs: Optional[Dict[Any, Any]] = None
            ) -> np.ndarray:
        """
        Train model and make predictions for the training set.

        :param X:
            features
        :param y:
            target
        :param source_positions:
            indices of initial features to be used as conditions,
            default is the last one
        :param fit_kwargs:
            settings of internal estimator fit
        :return:
            predictions
        """
        try:
            self._fit(X, y, source_positions, fit_kwargs,
                      save_training_features_as_attr=True)
            predictions = self.estimator_.predict(self._extended_X)
            return predictions
        finally:
            self._extended_X = None


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

    def _set_internal_estimator(self) -> BaseEstimator:
        # Instantiate estimator from initialization parameters.
        estimator = self.estimator or LinearRegression()
        estimator_kwargs = self.estimator_kwargs or dict()
        return estimator.set_params(**estimator_kwargs)


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

    def _set_internal_estimator(self) -> BaseEstimator:
        # Instantiate estimator from initialization parameters.
        estimator = self.estimator or LogisticRegression()
        estimator_kwargs = self.estimator_kwargs or dict()
        return estimator.set_params(**estimator_kwargs)

    def _do_supplementary_preparations(
            self,
            X: np.ndarray,
            y: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray]:
        # Take into account nuances of classification.
        self.classes_ = unique_labels(y)
        if len(np.unique(y)) > 2:
            raise ValueError(
                "As of now, only binary classification is supported."
            )
        y_ = LabelEncoder().fit_transform(y)
        return X, y_

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
        check_is_fitted(self, ['target_encoder_'])
        if not hasattr(self.estimator_, "predict_proba"):
            raise NotImplementedError(
                "Internal estimator has not `predict_proba` method."
            )

        extended_X = self.target_encoder_.transform(X)
        predicted_probabilities = self.estimator_.predict_proba(extended_X)
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
            indices of initial features to be used as conditions,
            default is the last one
        :param fit_kwargs:
            settings of internal estimator fit
        :return:
            predicted probabilities
        """
        if not hasattr(self.estimator, "predict_proba"):
            raise NotImplementedError(
                "Internal estimator has not `predict_proba` method."
            )
        try:
            self._fit(X, y, source_positions, fit_kwargs,
                      save_training_features_as_attr=True)
            predicted_probabilities = \
                self.estimator_.predict_proba(self._extended_X)
            return predicted_probabilities
        finally:
            self._extended_X = None
