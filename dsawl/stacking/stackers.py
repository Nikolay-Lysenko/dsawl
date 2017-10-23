"""
This module provides tools for stacking a model on top of other
models without information leakage from a target variable to
predictions made by base models.

@author: Nikolay Lysenko
"""


from typing import List, Dict, Tuple, Callable, Union, Optional, Any

import numpy as np

from sklearn.base import (
    BaseEstimator, RegressorMixin, ClassifierMixin
)
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

from .utils import InitablePipeline


# For the sake of convenience, define a new type.
FoldType = Union[KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit]


class BaseStacking(BaseEstimator):
    """
    A parent class for regression and classification stacking.

    :param base_estimators_types:
        list of types of first stage estimators, a type can occur
        multiple times here
    :param base_estimators_params:
        list of (hyper)parameters of first stage estimators such
        that its i-th element relates to the i-th element of
        `base_estimator_types`
    :param meta_estimator_type:
        a type of second stage estimator
    :param meta_estimator_params:
        (hyper)parameters of second stage estimator
    :param splitter:
        an object that splits learning sample into folds for
        first stage estimators training
    :param keep_meta_X:
        if it is `True`, out-of-fold predictions made by first stage
        estimators are stored in the attribute named `meta_X_`
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
            meta_estimator_type: Optional[type] = None,
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
        raise NotImplementedError

    def _create_base_estimators_from_their_types(
            self,
            types: List[type]
            ) -> List[BaseEstimator]:
        # Create a list of base estimators from a list of their types and
        # parameters of `self`.

        # Validate input.
        types = [x if x != Pipeline else InitablePipeline for x in types]
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
        raise NotImplementedError

    def _create_meta_estimator_from_its_type(
            self,
            meta_estimator_type: type,
            ) -> BaseEstimator:
        # Instantiate second stage estimator based on its type and parameters
        # of `self`.
        if meta_estimator_type == Pipeline:
            meta_estimator_type = InitablePipeline
        meta_estimator_params = self.meta_estimator_params or dict()
        # TODO: Delete if all is OK without it.
        # if meta_estimator_type == Pipeline:
        #     meta_estimator = (
        #         Pipeline(steps=[('arbitrary_step', LinearRegression())])
        #         .set_params(**meta_estimator_params)
        #     )
        # else:
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
    def _infer_operation(fitted_estimator: BaseEstimator) -> Callable:
        # Figure out what `fitted_estimator` must do according to its type.
        raise NotImplementedError

    @staticmethod
    def __restore_initial_order(
            meta_features: np.ndarray,
            folds: List[Tuple[np.ndarray]]
            ) -> np.ndarray:
        # Rearrange data for the second stage model and get order of rows
        # that corresponds to initial order of objects.
        ordering_column = np.hstack([x[1] for x in folds]).reshape((-1, 1))
        meta_features = np.hstack((meta_features, ordering_column))
        meta_features = meta_features[meta_features[:, -1].argsort(), :-1]
        return meta_features

    def _preprocess_target_variable(self, y: np.ndarray) -> np.ndarray:
        # Run operations that are specific to regression or classification.
        return y

    def _apply_fitted_base_estimator(
            self,
            apply_fn: Callable,
            estimator: BaseEstimator,
            X: np.ndarray,
            labels_from_training_folds: Optional[List[int]] = None
            ) -> np.ndarray:
        # Use `estimator` on `X` with `apply_fn`.
        # It is a version for `StackingRegressor`.
        # This method is overridden in `StackingClassifier`.
        result = apply_fn(estimator, X)
        return result

    def _fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            base_fit_kwargs: Optional[Dict[type, Dict[str, Any]]] = None,
            meta_fit_kwargs: Optional[Dict[str, Any]] = None,
            ) -> 'BaseStacking':
        # Implement internal logic of fitting.

        X, y = check_X_y(X, y)
        y = self._preprocess_target_variable(y)

        base_estimators = self._create_base_estimators()
        splitter = self._create_splitter()
        base_fit_kwargs = (
            base_fit_kwargs or {x: dict() for x in base_estimators}
        )

        self.base_estimators_ = []

        folds = list(splitter.split(X))
        meta_features = []
        for estimator in base_estimators:
            apply_fn = self._infer_operation(estimator)
            current_meta_feature = []
            for fit_indices, hold_out_indices in folds:
                estimator.fit(
                    X[fit_indices, :],
                    y[fit_indices],
                    **base_fit_kwargs.get(estimator, dict())
                )
                current_meta_feature_on_fold = (
                    self._apply_fitted_base_estimator(
                        apply_fn, estimator, X[hold_out_indices, :],
                        sorted(np.unique(y[fit_indices]).tolist())
                    )
                )
                current_meta_feature.append(current_meta_feature_on_fold)
            current_meta_x = np.vstack(current_meta_feature)
            current_meta_x = self.__restore_initial_order(
                current_meta_x, folds
            )
            meta_features.append(current_meta_x)
            # After all folds are processed, fit `estimator` to whole dataset.
            self.base_estimators_.append(
                estimator.fit(X, y, **base_fit_kwargs.get(estimator, dict()))
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
            base_fit_kwargs: Optional[Dict[type, Dict[str, Any]]] = None,
            meta_fit_kwargs: Optional[Dict[str, Any]] = None
            ) -> 'BaseStacking':
        """
        Train estimators from both stages of stacking.

        :param X:
            features
        :param y:
            target
        :param base_fit_kwargs:
            settings of first stage estimators training, first stage
            estimators are identified by their types, as of now two
            estimators of the same type can not have different
            settings
        :param meta_fit_kwargs:
            settings of second stage estimator training
        :return:
            fitted instance
        """
        return self._fit(X, y, base_fit_kwargs, meta_fit_kwargs)

    def _predict_on_the_second_stage(
            self,
            meta_X: np.ndarray,
            return_probabilities: Optional[bool] = False
            ) -> np.ndarray:
        # Make predictions with meta-estimator.
        # It is a version for `StackingRegressor`.
        # This method is overridden in `StackingClassifier`.
        predictions = self.meta_estimator_.predict(meta_X)
        return predictions

    def _predict(
            self,
            X: np.ndarray,
            return_probabilities: Optional[bool] = False
            ) -> np.ndarray:
        # Implement internal logic of predicting.

        check_is_fitted(self, ['base_estimators_', 'meta_estimator_'])
        X = check_array(X)

        meta_features = []
        for estimator in self.base_estimators_:
            apply_fn = self._infer_operation(estimator)
            current_meta_feature = self._apply_fitted_base_estimator(
                apply_fn, estimator, X,
                list(range(len(self.classes_)))
                if hasattr(self, 'classes_')
                else []
            )
            meta_features.append(current_meta_feature)
        meta_X = np.hstack(meta_features)

        predictions = self._predict_on_the_second_stage(
            meta_X, return_probabilities
        )
        return predictions

    def predict(
            self,
            X: np.ndarray,
            ) -> np.ndarray:
        """
        Predict target variable on a new dataset.

        :param X:
            features
        :return:
            predictions
        """
        return self._predict(X, return_probabilities=False)

    def drop_training_meta_features(self) -> type(None):
        """
        Delete a sample on which second stage estimator was trained.

        :return:
            None
        """
        self.meta_X_ = None


class StackingRegressor(BaseStacking, RegressorMixin):
    """
    A class that allows training a regressor on predictions made by
    other regressors and/or transformations made by transformers.
    Information does not leak through predictions and transformations,
    because all of them are made in an out-of-fold manner.
    """

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
        meta_estimator = self._create_meta_estimator_from_its_type(
            meta_estimator_type
        )
        return meta_estimator

    @staticmethod
    def _infer_operation(fitted_estimator: BaseEstimator) -> Callable:
        # Figure out what `fitted_estimator` must do according to its type.

        def predict(estimator: BaseEstimator, X: np.ndarray) -> np.ndarray:
            return estimator.predict(X).reshape((-1, 1))

        def transform(estimator: BaseEstimator, X: np.ndarray) -> np.ndarray:
            result = estimator.transform(X)
            result = (
                result if len(result.shape) > 1 else result.reshape((-1, 1))
            )
            return result

        if hasattr(fitted_estimator, 'predict'):
            return predict
        elif hasattr(fitted_estimator, 'transform'):
            return transform
        else:
            raise ValueError(
                'Invalid type of estimator: {}'.format(type(fitted_estimator))
            )


class StackingClassifier(BaseStacking, ClassifierMixin):
    """
    A class that allows training a classifier on predictions made by
    other classifiers and/or transformations made by transformers.
    Information does not leak through predictions and transformations,
    because all of them are made in an out-of-fold manner.
    """

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
        meta_estimator = self._create_meta_estimator_from_its_type(
            meta_estimator_type
        )
        return meta_estimator

    @staticmethod
    def _infer_operation(fitted_estimator: BaseEstimator) -> Callable:
        # Figure out what `fitted_estimator` must do according to its type.

        def predict(
                estimator: BaseEstimator,
                X: np.ndarray,
                *args, **kwargs
                ) -> np.ndarray:
            return estimator.predict(X).reshape((-1, 1))

        def predict_proba(
                estimator: BaseEstimator,
                X: np.ndarray,
                *args, **kwargs
                ) -> np.ndarray:

            def predict_proba_for_all_classes(
                    estimator: BaseEstimator,
                    X: np.ndarray,
                    train_labels: List[int],
                    n_all_labels: int
                    ) -> np.ndarray:
                # Take into consideration that some classes may be not
                # represented on training folds.
                preds = np.zeros((X.shape[0], n_all_labels))
                preds[:, train_labels] = estimator.predict_proba(X)
                # Last column is dropped, because probabilities sum up to 1.
                preds = preds[:, :-1]
                return preds

            return predict_proba_for_all_classes(estimator, X, *args, **kwargs)

        def transform(
                estimator: BaseEstimator,
                X: np.ndarray,
                *args, **kwargs
                ) -> np.ndarray:
            result = estimator.transform(X)
            result = (
                result if len(result.shape) > 1 else result.reshape((-1, 1))
            )
            return result

        if hasattr(fitted_estimator, 'predict_proba'):
            return predict_proba
        elif hasattr(fitted_estimator, 'predict'):
            return predict
        elif hasattr(fitted_estimator, 'transform'):
            return transform
        else:
            raise ValueError(
                'Invalid type of estimator: {}'.format(type(fitted_estimator))
            )

    def _preprocess_target_variable(self, y: np.ndarray) -> np.ndarray:
        # Convert class labels to dense integers.
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        return y

    def _apply_fitted_base_estimator(
            self,
            apply_fn: Callable,
            estimator: BaseEstimator,
            X: np.ndarray,
            labels_from_training_folds: Optional[List[int]] = None
            ) -> np.ndarray:
        # Use `estimator` on `X` with `apply_fn`.
        result = apply_fn(
            estimator,
            X,
            labels_from_training_folds,
            len(self.classes_)
        )
        return result

    def _predict_on_the_second_stage(
            self,
            meta_X: np.ndarray,
            return_probabilities: Optional[bool] = False
            ) -> np.ndarray:
        # Make predictions with meta-estimator.
        if return_probabilities:
            if not hasattr(self.meta_estimator_, 'predict_proba'):
                raise NotImplementedError(
                    "Second stage estimator has not `predict_proba` method."
                )
            predictions = self.meta_estimator_.predict_proba(meta_X)
        else:
            raw_predictions = self.meta_estimator_.predict(meta_X)
            predictions = np.apply_along_axis(
                lambda x: self.classes_[x],
                axis=0,
                arr=raw_predictions
            )
        return predictions

    def predict_proba(
            self,
            X: np.ndarray
            ) -> np.ndarray:
        """
        Predict probabilities of classes on a new dataset.

        :param X:
            features
        :return:
            estimated probabilities of classes
        """
        return self._predict(X, return_probabilities=True)
