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


from typing import List, Dict, Callable, Any, Optional

import numpy as np

from .target_encoder import TargetEncoder, FoldType
from dsawl.stacking.stackers import (
    BaseStacking, StackingRegressor, StackingClassifier
)


def _init(
        instance: BaseStacking,
        type_of_instance: type,
        estimator_type: Optional[type] = None,
        estimator_params: Optional[Dict] = None,
        splitter: Optional[FoldType] = None,
        aggregators: Optional[List[Callable]] = None,
        smoothing_strength: float = 0.0,
        min_frequency: int = 1,
        drop_source_features: bool = True
        ) -> BaseStacking:
    # A private function that allows getting rid of code duplication
    # between `__init__` method of `OutOfFoldTargetEncodingRegressor`
    # and `__init__` method of `OutOfFoldTargetEncodingClassifier`.
    instance.aggregators = aggregators
    instance.smoothing_strength = smoothing_strength
    instance.min_frequency = min_frequency
    instance.drop_source_features = drop_source_features

    base_estimators_types = [TargetEncoder]
    base_estimators_params = [
        {
            'aggregators': aggregators,
            'smoothing_strength': smoothing_strength,
            'min_frequency': min_frequency,
            'drop_source_features': drop_source_features
        }
    ]
    super(type_of_instance, instance).__init__(
        base_estimators_types, base_estimators_params,
        estimator_type, estimator_params,
        splitter
    )


def _fit_predict(
        instance: BaseStacking,
        X: np.ndarray,
        y: np.ndarray,
        source_positions: Optional[List[int]] = None,
        fit_kwargs: Optional[Dict[str, Any]] = None,
        return_probabilities: Optional[bool] = False
        ) -> np.ndarray:
    # A private function that allows getting rid of code duplication
    # between `fit_predict` methods. See more in documentation on
    # `fit_predict` of `OutOfFoldTargetEncodingRegressor` or
    # `OfFoldTargetEncodingClassifier`.
    if instance.keep_meta_X is False:
        raise ValueError(
            'Currently, `keep_meta_X` attribute is `False`.'
        )
    instance.fit(
        X, y,
        fit_kwargs={
            TargetEncoder: {'source_positions': source_positions}
        },
        meta_fit_kwargs=fit_kwargs,
    )
    if return_probabilities:
        predictions = instance.meta_estimator_.predict_proba(instance.meta_X_)
    else:
        predictions = instance.meta_estimator_.predict(instance.meta_X_)
    return predictions


class OutOfFoldTargetEncodingRegressor(StackingRegressor):
    """
    Regressor that has out-of-fold generation of target-based
    features before training.
    Internally, it is a stacking such that the first stage model
    transforms data with target encoding and the second stage model
    uses these transformed data.

    :param estimator_type:
        type (class) of internal estimator (i.e., second stage
        estimator)
    :param estimator_params:
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
            estimator_type: Optional[type] = None,
            estimator_params: Optional[Dict] = None,
            splitter: Optional[FoldType] = None,
            aggregators: Optional[List[Callable]] = None,
            smoothing_strength: float = 0.0,
            min_frequency: int = 1,
            drop_source_features: bool = True
            ):
        _init(
            self,
            type(self),
            estimator_type,
            estimator_params,
            splitter,
            aggregators,
            smoothing_strength,
            min_frequency,
            drop_source_features
        )

    def fit_predict(
            self,
            X: np.ndarray,
            y: np.ndarray,
            source_positions: Optional[List[int]] = None,
            fit_kwargs: Optional[Dict[str, Any]] = None
            ) -> np.ndarray:
        """
        Train model and make predictions for the training set.
        This method differs from composition of `fit` and `predict`.

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
        return _fit_predict(self, X, y, source_positions, fit_kwargs)


class OutOfFoldTargetEncodingClassifier(StackingClassifier):
    """
    Classifier that has out-of-fold generation of target-based
    features before training.
    Internally, it is a stacking such that the first stage model
    transforms data with target encoding and the second stage model
    uses these transformed data.

    :param estimator_type:
        type (class) of internal estimator (i.e., second stage
        estimator)
    :param estimator_params:
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
            estimator_type: Optional[type] = None,
            estimator_params: Optional[Dict] = None,
            splitter: Optional[FoldType] = None,
            aggregators: Optional[List[Callable]] = None,
            smoothing_strength: float = 0.0,
            min_frequency: int = 1,
            drop_source_features: bool = True
            ):
        _init(
            self,
            type(self),
            estimator_type,
            estimator_params,
            splitter,
            aggregators,
            smoothing_strength,
            min_frequency,
            drop_source_features
        )

    def fit_predict(
            self,
            X: np.ndarray,
            y: np.ndarray,
            source_positions: Optional[List[int]] = None,
            fit_kwargs: Optional[Dict[str, Any]] = None
            ) -> np.ndarray:
        """
        Train model and make predictions for the training set.
        This method differs from composition of `fit` and `predict`.

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
        return _fit_predict(self, X, y, source_positions, fit_kwargs)

    def fit_predict_proba(
            self,
            X: np.ndarray,
            y: np.ndarray,
            source_positions: Optional[List[int]] = None,
            fit_kwargs: Optional[Dict[str, Any]] = None
            ) -> np.ndarray:
        """
        Train model and predict class probabilities for the training
        set.
        This method differs from composition of `fit` and
        `predict_proba`.

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
            estimated probabilities of classes
        """
        return _fit_predict(
            self, X, y, source_positions, fit_kwargs, return_probabilities=True
        )
