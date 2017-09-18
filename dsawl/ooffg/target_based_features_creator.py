"""
This module is for both in-fold and out-of-fold generation of
features that are derived in the following manner:
1) group target variable by a particular initial feature;
2) apply aggregating function to each group;
3) for each object, use corresponding aggregated value
   as a new feature.

@author: Nikolay Lysenko
"""


from typing import List, Callable, Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit
)


class TargetBasedFeaturesCreator(BaseEstimator, TransformerMixin):
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

    :param aggregators:
        functions that compute aggregates
    :param splitter:
        object that splits data into folds for out-of-fold
        transformation
    :param smoothing_strength:
        strength of smoothing towards unconditional aggregates
    :param min_frequency:
        minimal number of occurrences of a feature's value (if value
        occurs less times than this parameter, this value is mapped to
        unconditional aggregate)
    :param drop_source_features:
        drop or keep those of initial features that are used for
        conditioning over them
    """

    def __init__(
            self,
            aggregators: List[Callable] = None,
            splitter: Union[
                KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit
            ] = None,
            smoothing_strength: float = 0,
            min_frequency: int = 1,
            drop_source_features: bool = True
            ):
        self.aggregators = [np.mean] if aggregators is None else aggregators
        self.splitter = KFold() if splitter is None else splitter
        self.smoothing_strength = smoothing_strength
        self.min_frequency = min_frequency
        self.drop_source_features = drop_source_features
        self.mappings_ = dict()

    def __process_raw_aggregator(
            self,
            aggregator: Callable,
            target: np.ndarray,
            ) -> Callable:
        # Make `aggregator` smoothing towards unconditional aggregates
        # according to class parameters.

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

        # `gb.agg(funcs)` requires unique names of all functions from `funcs`.
        func = compute_aggregate
        func.__name__ = aggregator.__name__
        return func

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            source_positions: List[int]
            ) -> 'TargetBasedFeaturesCreator':
        """
        Fit to a whole dataset of `X` and `y`.
        In other words, memorize mappings from initial values
        of selected columns to conditional aggregates.

        :param X:
            features
        :param y:
            target
        :param source_positions:
            indices of initial features to be used as conditions
        :return:
            fitted instance
        """
        for position in source_positions:
            feature = X[:, position].reshape((-1, 1))
            target = y.reshape((-1, 1))
            df = pd.DataFrame(np.hstack((feature, target)), columns=['x', 'y'])
            mapping = df.groupby('x')['y'].agg(
                [self.__process_raw_aggregator(agg, target)
                 for agg in self.aggregators]
            )
            mapping = pd.DataFrame(
                np.hstack(
                    (mapping.index.values.reshape((-1, 1)), mapping.values)
                ),
                columns=[str(position)] +
                        ['agg_{}'.format(x)
                         for x in range(len(self.aggregators))]
            )
            # '__unseen__' is a reserved key for unseen values.
            mapping.loc['__unseen__'] = [np.nan] + [
                agg(y) for agg in self.aggregators
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
        transformed_df = pd.DataFrame(
            X,
            columns=[str(x) for x in range(X.shape[1])]
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
            relevant_columns = list(filter(
                lambda x: x not in self.mappings_.keys(),
                range(transformed_X.shape[1])
            ))
            transformed_X = transformed_X[:, relevant_columns]
        return transformed_X

    def fit_transform_out_of_fold(
            self,
            X: np.ndarray,
            y: np.ndarray,
            source_positions: List[int]
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
            indices of initial features to be used as conditions
        :return:
            transformed feature representation
        """
        new_n_columns = (X.shape[1] +
                         len(self.aggregators) * len(source_positions) -
                         self.drop_source_features * len(source_positions))
        transformed_X = np.full((X.shape[0], new_n_columns), np.nan)
        for fit_indices, transform_indices in self.splitter.split(X):
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
