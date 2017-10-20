"""
---

@author: Nikolay Lysenko
"""
# TODO: Write above docstring.


import warnings
from typing import List, Dict, Tuple, Callable, Union, Optional, Any

import numpy as np

from sklearn.base import (
    BaseEstimator,
    RegressorMixin, ClassifierMixin, TransformerMixin,
    clone
)
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression


# For the sake of convenience, define a new type.
FoldType = Union[KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit]


class BaseStacking(BaseEstimator):
    """

    """

    def __init__(
            self,
            base_estimators: Optional[List[Any]] = None,
            base_estimators_params:
                Optional[Dict[Any, List[Dict[str, Any]]]] = None,
            meta_estimator: Optional[Any] = None,
            meta_estimator_params: Optional[Dict[str, Any]] = None,
            splitter: Optional[FoldType] = None,
            keep_meta_X: bool = True
            ):
        self._can_this_class_have_any_instances()
        self.base_estimators = base_estimators
        self.base_estimators_params = base_estimators_params
        self.meta_estimator = meta_estimator
        self.meta_estimator_params = meta_estimator_params
        self.splitter = splitter
        self.keep_meta_X = keep_meta_X

    @classmethod
    def _can_this_class_have_any_instances(cls):
        # Make this class abstract.
        raise TypeError('{} must not have any instances.'.format(cls))

    def _set_base_estimators(self) -> List[BaseEstimator]:
        # Instantiate base estimators from initialization parameters.
        pass

    def _instantiate_base_estimators_from_stubs_and_params(
            self,
            stubs: List[BaseEstimator],
            params: Dict[Any, List[Dict[str, Any]]]
            ) -> List[BaseEstimator]:
        # Create a list of base estimators from a list of their instances and
        # a mapping from their types to all their parameters.

        # Validate input.
        for estimator_type, params_list in params.items():
            if len(params_list) == 0:
                warnings.warn(
                    '{} will be removed from list '.format(estimator_type) +
                    'of base estimators due to empty list of parameters.',
                    RuntimeWarning
                )

        base_estimators = [
            [
                clone(estimator).set_params(**kwargs)
                for kwargs in params.get( type(estimator), [dict()])
            ]
            for estimator in stubs
        ]
        base_estimators = [
            estimator for sublist in base_estimators for estimator in sublist
        ]
        return base_estimators

    def _set_meta_estimator(self) -> BaseEstimator:
        # Instantiate second stage estimator from initialization parameters.
        pass

    @staticmethod
    def __infer_operation(fitted_estimator: Any) -> Callable:
        # Figure out what `fitted_estimator` must do according to its type.

        def predict(estimator: Any, *args, **kwargs) -> np.ndarray:
            return estimator.predict(*args, **kwargs).reshape((-1, 1))

        def predict_proba(estimator: Any, *args, **kwargs) -> np.ndarray:
            # Last column is dropped, because probabilities sum up to 1.
            return estimator.predict_proba(*args, **kwargs)[:, :-1]

        def transform(estimator: Any, *args, **kwargs) -> np.ndarray:
            result = estimator.transform(*args, **kwargs)
            result = (
                result if len(result.shape) > 1 else result.reshape((-1, 1))
            )
            return result

        if isinstance(fitted_estimator, ClassifierMixin):
            if hasattr(fitted_estimator, 'predict_proba'):
                return predict_proba
            else:
                return predict
        elif isinstance(fitted_estimator, RegressorMixin):
            return predict
        elif isinstance(fitted_estimator, TransformerMixin):
            return transform
        else:
            raise ValueError(
                'Invalid type of estimator: {}'.format(type(fitted_estimator))
            )

    @staticmethod
    def __restore_initial_order(
            meta_features: np.array,
            folds: List[Tuple[np.array]]
            ) -> np.array:
        # Rearrange data for the second stage model and get order of rows
        # that corresponds to initial order of objects.
        ordering_column = np.hstack([x[1] for x in folds]).reshape((-1, 1))
        meta_features = np.hstack((meta_features, ordering_column))
        meta_features = meta_features[meta_features[:, -1].argsort(), :-1]
        return meta_features

    def _fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            fit_kwargs: Optional[Dict[Any, Dict[str, Any]]] = None,
            meta_fit_kwargs: Optional[Dict[str, Any]] = None
            ) -> 'BaseStacking':
        # Implement internal logic of fitting.

        X, y = check_X_y(X, y)
        base_estimators = self._set_base_estimators()
        fit_kwargs = fit_kwargs or {x: dict() for x in base_estimators}
        splitter = self.splitter or KFold()

        self.base_estimators_ = []

        folds = list(splitter.split(X))
        meta_features = []
        for estimator in base_estimators:
            apply_fn = self.__infer_operation(estimator)
            current_meta_feature = []
            for fit_indices, hold_out_indices in folds:
                estimator.fit(
                    X[fit_indices, :],
                    y[fit_indices],
                    **fit_kwargs.get(estimator, dict())
                )
                current_meta_feature_on_fold = apply_fn(
                    estimator, X[hold_out_indices, :]
                )
                current_meta_feature.append(current_meta_feature_on_fold)
            current_meta_x = np.vstack(current_meta_feature)
            current_meta_x = self.__restore_initial_order(
                current_meta_x, folds
            )
            meta_features.append(current_meta_x)
            # After all folds are processed, fit `estimator` to whole dataset.
            self.base_estimators_.append(
                estimator.fit(X, y, **fit_kwargs.get(estimator, dict()))
            )
        meta_X = np.hstack(meta_features)

        meta_estimator = self._set_meta_estimator()
        meta_fit_kwargs = meta_fit_kwargs or dict()
        self.meta_estimator_ = meta_estimator.fit(meta_X, y, **meta_fit_kwargs)
        if self.keep_meta_X:
            self.meta_X_ = meta_X
        return self

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            fit_kwargs: Optional[Dict[Any, Dict[str, Any]]] = None,
            meta_fit_kwargs: Optional[Dict[str, Any]] = None
            ) -> 'BaseStacking':
        """
        """
        return self._fit(X, y, fit_kwargs, meta_fit_kwargs)

    def _predict(
            self,
            X: np.array,
            return_probabilities: bool
            ) -> np.array:
        # Implement internal logic of predicting.

        check_is_fitted(self, ['base_estimators_', 'meta_estimator_'])
        X = check_array(X)

        meta_features = []
        for estimator in self.base_estimators_:
            apply_fn = self.__infer_operation(estimator)
            current_meta_feature = apply_fn(estimator, X)
            meta_features.append(current_meta_feature)
        meta_X = np.hstack(meta_features)

        if return_probabilities:
            predictions = self.meta_estimator_.predict_proba(meta_X)
        else:
            predictions = self.meta_estimator_.predict(meta_X)
        return predictions

    def predict(
            self,
            X: np.array,
            ) -> np.array:
        """

        :param X:
        :return:
        """
        return self._predict(X, return_probabilities=False)

    def drop_training_meta_features(self) -> type(None):
        """

        :return:
            None
        """
        self.meta_X_ = None


class StackingRegressor(BaseStacking, RegressorMixin):
    """

    """

    @classmethod
    def _can_this_class_have_any_instances(cls):
        # Allow this class to have instances.
        pass

    def _set_base_estimators(self) -> List[BaseEstimator]:
        # Instantiate base estimators from initialization parameters.

        # Replace `None` values with the corresponding default values.
        default_estimators = [LinearRegression(), KNeighborsRegressor()]
        base_estimators = self.base_estimators or default_estimators
        default_params = {type(x): [dict()] for x in base_estimators}
        base_estimators_params = self.base_estimators_params or default_params

        base_estimators = (
            self._instantiate_base_estimators_from_stubs_and_params(
                base_estimators, base_estimators_params
            )
        )
        return base_estimators

    def _set_meta_estimator(self) -> BaseEstimator:
        # Instantiate second stage estimator from initialization parameters.
        meta_estimator = self.meta_estimator or LinearRegression()
        meta_estimator_params = self.meta_estimator_params or dict()
        meta_estimator.set_params(**meta_estimator_params)
        return meta_estimator


class StackingClassifier(BaseStacking, ClassifierMixin):
    """

    """

    @classmethod
    def _can_this_class_have_any_instances(cls):
        # Allow this class to have instances.
        pass

    def _set_base_estimators(self) -> List[BaseEstimator]:
        # Instantiate base estimators from initialization parameters.

        # Replace `None` values with the corresponding default values.
        default_estimators = [LogisticRegression(), KNeighborsClassifier()]
        base_estimators = self.base_estimators or default_estimators
        default_params = {type(x): [dict()] for x in base_estimators}
        base_estimators_params = self.base_estimators_params or default_params

        base_estimators = (
            self._instantiate_base_estimators_from_stubs_and_params(
                base_estimators, base_estimators_params
            )
        )
        return base_estimators

    def _set_meta_estimator(self) -> BaseEstimator:
        # Instantiate second stage estimator from initialization parameters.
        meta_estimator = self.meta_estimator or LogisticRegression()
        meta_estimator_params = self.meta_estimator_params or dict()
        meta_estimator.set_params(**meta_estimator_params)
        return meta_estimator

    def predict_proba(
            self,
            X: np.array
            ) -> np.array:
        """

        :param X:
        :return:
        """
        if not hasattr(self.meta_estimator_, 'predict_proba'):
            raise NotImplementedError(
                "Second stage estimator has not `predict_proba` method."
            )
        return self._predict(X, return_probabilities=True)
