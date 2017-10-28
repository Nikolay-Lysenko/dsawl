"""
This module is for both in-fold and out-of-fold generation of
features that are derived in the following manner:
1) group target variable by a particular initial feature;
2) apply aggregating function to each group;
3) for each object, use corresponding aggregated value
   as a new feature.
This trick is sometimes called target encoding.

@author: Nikolay Lysenko
"""


from typing import List, Tuple, Callable, Optional

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import KFold, TimeSeriesSplit

from dsawl.stacking.stackers import FoldType


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    This class can augment datasets with new features such that
    each of them is conditional value of an aggregating function
    applied to target variable where conditioning is taken over
    a specified initial feature.

    For example, this class can replace a categorical feature
    (unsuitable for lots of machine learning algorithms) with
    within-group mean of target variable where grouping is by
    this categorical feature.

    In order to handle rare values of initial source features,
    smoothing towards unconditional (i.e., population, general)
    aggregates can be applied.

    Also this class allows avoiding overfitting by generating
    new features based only on out-of-fold values of target.

    NB: As of now, multi-label classification is not fully supported,
    only binary classification and regression are fully supported.
    Results for multi-label classification are adequate if class
    labels are ordered and equally spaced (e.g., a difference between
    'A' and 'B' grades is equal to a difference between 'B' and 'C'
    grades). Since it can not be automatically detected whether these
    conditions hold true, it is up to user to verify passed target
    and there are no warnings. Passing classification target with more
    than two distinct values such that they are not ordered and equally
    spaced results in encoding that is not as good as union of encodings
    based on binary indicators of the classes.

    :param aggregators:
        functions that compute aggregates, default is mean function
    :param splitter:
        object that splits data into folds for out-of-fold
        transformation, default schema is Leave-One-Out.
    :param smoothing_strength:
        strength of smoothing towards unconditional aggregates,
        by default there is no smoothing
    :param min_frequency:
        minimal number of occurrences of a feature's value (if value
        occurs less times than this parameter, this value is mapped to
        unconditional aggregate), by default it is 1
    :param drop_source_features:
        to drop or to keep those of initial features that are used for
        conditioning over them, default is to drop
    """

    def __init__(
            self,
            aggregators: Optional[List[Callable]] = None,
            splitter: Optional[FoldType] = None,
            smoothing_strength: float = 0,
            min_frequency: int = 1,
            drop_source_features: bool = True
            ):
        self.aggregators = aggregators
        self.splitter = splitter
        self.smoothing_strength = smoothing_strength
        self.min_frequency = min_frequency
        self.drop_source_features = drop_source_features

    def __process_raw_aggregator(
            self,
            aggregator: Callable,
            target: np.ndarray,
            ) -> Callable:
        # Make `aggregator` smoothing towards unconditional aggregates
        # according to parameters of the current instance.

        def compute_aggregate(ser):
            n_occurrences = len(ser.index)
            if n_occurrences < self.min_frequency:
                return aggregator(target)
            else:
                numerator = (
                    n_occurrences * aggregator(ser) +
                    self.smoothing_strength * aggregator(target)
                )
                denominator = n_occurrences + self.smoothing_strength
                return numerator / denominator

        func = compute_aggregate
        # `gb.agg(funcs)` requires unique names of all functions from `funcs`.
        func.__name__ = aggregator.__name__
        return func

    def __handle_negative_indices(
            self,
            source_positions: List[int]
            ) -> List[int]:
        # Allow writing indices like `arr[-1]` instead of `arr[len(arr) - 1]`.
        source_positions = list(map(
            lambda x: x + self.n_columns_ if x < 0 else x,
            source_positions
        ))
        return source_positions

    @staticmethod
    def __coalesce(
            aggregators: Optional[List[Callable]] = None,
            source_positions: Optional[List[int]] = None,
            splitter: Optional[FoldType] = None,
            n_splits: int = 3
            ) -> Tuple[List[Callable], List[int], FoldType]:
        # Fill missed values with corresponding defaults.
        aggregators = aggregators or [np.mean]
        source_positions = source_positions or [-1]
        splitter = splitter or KFold(n_splits)
        return aggregators, source_positions, splitter

    def __drop_source_features(self, transformed_X: np.ndarray) -> np.ndarray:
        # Remove from `X` features that has been used for conditioning.
        relevant_columns = [
            x
            for x in range(transformed_X.shape[1])
            if x not in self.mappings_.keys()
        ]
        return transformed_X[:, relevant_columns]

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            source_positions: Optional[List[int]] = None
            ) -> 'TargetEncoder':
        """
        Fit to a whole dataset of `X` and `y`.
        In other words, memorize mappings from initial values
        of selected columns to conditional aggregates.

        :param X:
            features
        :param y:
            target
        :param source_positions:
            indices of initial features to be used as conditions,
            default is the last one column
        :return:
            fitted instance
        """
        X, y = check_X_y(X, y)
        # `n_columns_` attribute is created predominantly for the
        # sake of full compatibility with `sklearn`.
        self.n_columns_ = X.shape[1]
        self.mappings_ = dict()
        aggregators, source_positions, _ = self.__coalesce(
            self.aggregators, source_positions
        )

        for position in self.__handle_negative_indices(source_positions):
            feature = X[:, position].reshape((-1, 1))
            target = y.reshape((-1, 1))
            df = pd.DataFrame(np.hstack((feature, target)), columns=['x', 'y'])
            mapping = df.groupby('x')['y'].agg(
                [self.__process_raw_aggregator(agg, target)
                 for agg in aggregators]
            )
            mapping.reset_index(inplace=True)
            mapping.columns = (
                [str(position)] +
                ['agg_{}'.format(x) for x in range(len(aggregators))]
            )
            # '__unseen__' is a reserved key for unseen values.
            mapping.loc['__unseen__'] = [np.nan] + [
                agg(y) for agg in aggregators
            ]
            self.mappings_[position] = mapping
        return self

    def transform(
            self,
            X: np.ndarray
            ) -> np.ndarray:
        """
        Augment `X` with learnt conditional aggregates.

        :param X:
            feature representation to be augmented
        :return:
            transformed data
        """
        check_is_fitted(self, ['mappings_', 'n_columns_'])
        X = check_array(X)
        if X.shape[1] != self.n_columns_:
            raise ValueError(
                'Shape of input is different from what was seen in `fit`.'
            )

        transformed_df = pd.DataFrame(
            X, columns=[str(x) for x in range(X.shape[1])]
        )
        for position, mapping in self.mappings_.items():
            transformed_df = transformed_df.merge(
                mapping, on=str(position), how='left'
            )
            n_new_columns = len(mapping.columns) - 1  # Only aggregates
            default_values = mapping.loc['__unseen__'].values[1:]
            transformed_df.loc[
                pd.isnull(transformed_df.iloc[:, -1]), -n_new_columns:
            ] = default_values
        transformed_X = transformed_df.values
        if self.drop_source_features:
            transformed_X = self.__drop_source_features(transformed_X)
        return transformed_X

    def fit_transform_out_of_fold(
            self,
            X: np.ndarray,
            y: np.ndarray,
            source_positions: Optional[List[int]] = None
            ) -> np.ndarray:
        """
        Enrich `X` with features based on `y` in a manner that
        reduces risk of overfitting.
        This property holds true, because for each object its own
        value of target variable is not used for generation of
        its new features.

        :param X:
            feature representation to be augmented
        :param y:
            target to be incorporated in new features
        :param source_positions:
            indices of initial features to be used as conditions,
            default is the last one
        :return:
            transformed feature representation
        """
        X, y = check_X_y(X, y)
        aggregators, source_positions, splitter = self.__coalesce(
            self.aggregators, source_positions, self.splitter, X.shape[0]
        )

        new_n_columns = (X.shape[1] +
                         len(aggregators) * len(source_positions) -
                         self.drop_source_features * len(source_positions))
        transformed_X = np.full((X.shape[0], new_n_columns), np.nan)
        for fit_indices, transform_indices in splitter.split(X):
            self.fit(
                X[fit_indices, :],
                y[fit_indices],
                source_positions
            )
            transformed_X[transform_indices, :] = self.transform(
                X[transform_indices, :]
            )
        if isinstance(self.splitter, TimeSeriesSplit):
            # Drop rows from the earliest fold.
            transformed_X = transformed_X[
                ~np.isnan(transformed_X).any(axis=1), :
            ]
        self.fit(X, y, source_positions)  # Finally, fit to whole dataset.
        return transformed_X
