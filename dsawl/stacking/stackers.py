"""
---

@author: Nikolay Lysenko
"""
# TODO: Write above docstring.


from typing import List, Dict, Tuple, Callable, Union, Optional, Any

import numpy as np

from sklearn.base import (
    BaseEstimator,
    RegressorMixin, ClassifierMixin, TransformerMixin
)
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier


# For the sake of convenience, define a new type.
FoldType = Union[KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit]


class BaseStacking(BaseEstimator):
    """

    :param random_state:
        random state for all estimators and first stage splitting;
        if it is set, it overrides all other random states,
        i.e., the ones that are set in `base_estimators_params`,
        `meta_estimator_params`, and `splitter`.
    """

    def __init__(
            self,
            base_estimators_types: Optional[List[type]] = None,
            base_estimators_params: Optional[List[Dict[str, Any]]] = None,
            meta_estimator_type: Optional[Any] = None,
            meta_estimator_params: Optional[Dict[str, Any]] = None,
            splitter: Optional[FoldType] = None,
            keep_meta_X: bool = True,
            random_state: Optional[int] = None
            ):
        self._can_this_class_have_any_instances()
        self.base_estimators_types = base_estimators_types
        self.base_estimators_params = base_estimators_params
        self.meta_estimator_type = meta_estimator_type
        self.meta_estimator_params = meta_estimator_params
        self.splitter = splitter
        self.keep_meta_X = keep_meta_X
        self.random_state = random_state

    @classmethod
    def _can_this_class_have_any_instances(cls):
        # Make this class abstract.
        raise NotImplementedError(
            '{} must not have any instances.'.format(cls)
        )

    def _create_base_estimators(self) -> List[BaseEstimator]:
        # Instantiate base estimators from initialization parameters.
        pass

    def _create_base_estimators_from_their_types(
            self,
            types: List[type]
            ) -> List[BaseEstimator]:
        # Create a list of base estimators from a list of their types and
        # parameters of `self`.

        # Validate input.
        params = (
            self.base_estimators_params or
            [dict() for _ in range(len(types))]
        )
        if len(types) != len(params):
            raise ValueError(
                (
                    'Lengths mismatch: `base_estimators_types` has length {}, '
                    'whereas `base_estimator_params` has length {}.'
                ).format(len(types), len(params))
            )

        pairs = zip(types, params)
        pairs = [
            (t, p)
            if 'random_state' not in t().get_params().keys()
            else (t, dict(p, **{'random_state': self.random_state}))
            for t, p in pairs
        ]
        base_estimators = [
            estimator_type().set_params(**params)
            for estimator_type, params in pairs
        ]
        return base_estimators

    def _create_meta_estimator(self) -> BaseEstimator:
        # Instantiate second stage estimator from initialization parameters.
        pass

    def _create_meta_estimator_by_its_type(
            self,
            meta_estimator_type: type,
            ) -> BaseEstimator:
        # Instantiate second stage estimator by its type and parameters
        # of `self`.
        meta_estimator_params = self.meta_estimator_params or dict()
        if 'random_state' in meta_estimator_type().get_params().keys():
            meta_estimator_params['random_state'] = self.random_state
        meta_estimator = (
            meta_estimator_type()
            .set_params(**meta_estimator_params)
        )
        return meta_estimator

    def _create_splitter(self) -> FoldType:
        # Create splitter that is used for the first stage of stacking.
        splitter = self.splitter or KFold()
        if hasattr(splitter, 'shuffle') and splitter.shuffle:
            splitter.random_state = self.random_state
        return splitter

    @staticmethod
    def __infer_operation(fitted_estimator: Any) -> Callable:
        # Figure out what `fitted_estimator` must do according to its type.

        def predict(
                estimator: Any,
                X: np.array,
                *args, **kwargs
                ) -> np.ndarray:
            return estimator.predict(X).reshape((-1, 1))

        def predict_proba(
                estimator: Any,
                X: np.array,
                *args, **kwargs
                ) -> np.ndarray:

            def predict_proba_for_all_classes(
                    estimator: Any,
                    X: np.array,
                    train_labels: Union[List[int], range],
                    n_all_labels: int
                    ) -> np.array:
                # Take into consideration that some classes may be not
                # represented on training folds.
                if isinstance(train_labels, range):
                    train_labels = list(train_labels)
                train_labels = [int(x) for x in train_labels]
                preds = np.zeros((X.shape[0], n_all_labels))
                preds[:, train_labels] = estimator.predict_proba(X)
                # Last column is dropped, because probabilities sum up to 1.
                preds = preds[:, :-1]
                return preds

            return predict_proba_for_all_classes(estimator, X, *args, **kwargs)

        def transform(
                estimator: Any,
                X: np.array,
                *args, **kwargs
                ) -> np.ndarray:
            result = estimator.transform(X)
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
            meta_fit_kwargs: Optional[Dict[str, Any]] = None,
            ) -> 'BaseStacking':
        # Implement internal logic of fitting.

        X, y = check_X_y(X, y)
        if self.is_classifier:
            self.classes_, y = np.unique(y, return_inverse=True)

        base_estimators = self._create_base_estimators()
        splitter = self._create_splitter()
        fit_kwargs = fit_kwargs or {x: dict() for x in base_estimators}

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
                    estimator,
                    X[hold_out_indices, :],
                    sorted(np.unique(y[fit_indices]).tolist()),
                    len(self.classes_) if self.is_classifier else 0
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

        meta_estimator = self._create_meta_estimator()
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
            if self.is_classifier:
                current_meta_feature = apply_fn(
                    estimator, X, range(len(self.classes_)), len(self.classes_)
                )
            else:
                current_meta_feature = apply_fn(estimator, X, 0, 0)
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

    def __init__(
            self,
            base_estimators_types: Optional[List[type]] = None,
            base_estimators_params: Optional[List[Dict[str, Any]]] = None,
            meta_estimator_type: Optional[Any] = None,
            meta_estimator_params: Optional[Dict[str, Any]] = None,
            splitter: Optional[FoldType] = None,
            keep_meta_X: bool = True,
            random_state: Optional[int] = None
            ):
        self.is_classifier = False
        # Style of `sklearn` prohibits usage of `*args` and `**kwargs`.
        super().__init__(
            base_estimators_types,
            base_estimators_params,
            meta_estimator_type,
            meta_estimator_params,
            splitter,
            keep_meta_X,
            random_state
        )

    @classmethod
    def _can_this_class_have_any_instances(cls):
        # Allow this class to have instances.
        pass

    def _create_base_estimators(self) -> List[BaseEstimator]:
        # Instantiate base estimators from initialization parameters.
        default_types = [RandomForestRegressor, KNeighborsRegressor]
        types = self.base_estimators_types or default_types
        base_estimators = (
            self._create_base_estimators_from_their_types(types)
        )
        return base_estimators

    def _create_meta_estimator(self) -> BaseEstimator:
        # Instantiate second stage estimator from initialization parameters.
        meta_estimator_type = self.meta_estimator_type or LinearRegression
        meta_estimator = self._create_meta_estimator_by_its_type(
            meta_estimator_type
        )
        return meta_estimator


class StackingClassifier(BaseStacking, ClassifierMixin):
    """

    """

    def __init__(
            self,
            base_estimators_types: Optional[List[type]] = None,
            base_estimators_params: Optional[List[Dict[str, Any]]] = None,
            meta_estimator_type: Optional[Any] = None,
            meta_estimator_params: Optional[Dict[str, Any]] = None,
            splitter: Optional[FoldType] = None,
            keep_meta_X: bool = True,
            random_state: Optional[int] = None
            ):
        self.is_classifier = True
        # Style of `sklearn` prohibits usage of `*args` and `**kwargs`.
        super().__init__(
            base_estimators_types,
            base_estimators_params,
            meta_estimator_type,
            meta_estimator_params,
            splitter,
            keep_meta_X,
            random_state
        )

    @classmethod
    def _can_this_class_have_any_instances(cls):
        # Allow this class to have instances.
        pass

    def _create_base_estimators(self) -> List[BaseEstimator]:
        # Instantiate base estimators from initialization parameters.
        default_types = [RandomForestClassifier, LogisticRegression]
        types = self.base_estimators_types or default_types
        base_estimators = (
            self._create_base_estimators_from_their_types(types)
        )
        return base_estimators

    def _create_meta_estimator(self) -> BaseEstimator:
        # Instantiate second stage estimator from initialization parameters.
        meta_estimator_type = self.meta_estimator_type or LogisticRegression
        meta_estimator = self._create_meta_estimator_by_its_type(
            meta_estimator_type
        )
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