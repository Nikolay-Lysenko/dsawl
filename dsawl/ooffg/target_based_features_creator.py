"""
This module is for both in-fold and out-of-fold generation of
features that are derived in the following manner:
1) group target variable by a particular initial feature;
2) apply aggregating function to each group;
3) for each object, use corresponding aggregated value
   as a new feature.

@author: Nikolay Lysenko
"""


from typing import List, Callable

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold


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

    :param aggregators: functions that compute aggregates
    :param smoothing_strength: strength of smoothing towards
                               unconditional aggregates
    :param min_frequency: minimal number of occurrences of a feature's
                          value (if value occurs less times than this
                          parameter, this value is mapped to
                          unconditional aggregate)
    :param drop_source_features: drop or keep those of initial
                                 features that are used for
                                 conditioning over them
    """

    def __init__(
            self,
            aggregators: List[Callable] = None,
            smoothing_strength: float = 0,
            min_frequency: int = 1,
            drop_source_features: bool = True
            ):
        self.aggregators = [np.mean] if aggregators is None else aggregators
        self.smoothing_strength = smoothing_strength
        self.min_frequency = min_frequency
        self.drop_source_features = drop_source_features
        self.mappings_ = dict()

    def __compute_aggregate(
            self,
            mask: np.array,
            target: np.array,
            aggregator: Callable
            ) -> float:
        n_occurrences = sum(mask)
        if n_occurrences < self.min_frequency:
            return aggregator(target)
        else:
            numerator = (
                n_occurrences * aggregator(target[mask]) +
                self.smoothing_strength * aggregator(target)
            )
            denominator = n_occurrences + self.smoothing_strength
            return numerator / denominator

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

        :param X: features
        :param y: target
        :param source_positions: indices of initial features to be
                                 used as conditions
        :return: fitted instance
        """
        for position in source_positions:
            feature = X[:, position]
            self.mappings_[position] = {
                k: [self.__compute_aggregate(feature == k, y, agg)
                    for agg in self.aggregators]
                for k in np.unique(feature)
            }
            # `None` is a reserved key for unseen values.
            self.mappings_[position][None] = [
                agg(y) for agg in self.aggregators
            ]
        return self

    def transform(
            self,
            X: np.ndarray
            ) -> np.ndarray:
        """
        Augment `X` with learnt conditional aggregates.

        :param X: feature representation to be augmented
        :return: transformed data
        """
        transformed_X = X.copy()
        for position, mappings in self.mappings_.items():
            feature = X[:, position]
            n_new = len(next(iter(mappings.values())))
            new_features = np.full((X.shape[0], n_new), np.nan)
            for value, conditional_aggregates in mappings.items():
                if value is None:
                    continue
                new_features[[feature == value]] = \
                    np.tile(conditional_aggregates, (sum(feature == value), 1))
            new_features[np.isnan(new_features[:, 0]), :] = mappings[None]
            transformed_X = np.hstack((transformed_X, new_features))
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
            source_positions: List[int],
            n_splits: int,
            shuffle: bool = False,
            random_state: int = None
            ) -> np.ndarray:
        """
        Enrich `X` with features based on `y` in a manner that
        reduces risk of overfitting.
        This property holds true, because for each object its own
        value of target variable is not used for generation of
        its new features.

        :param X: feature representation to be augmented
        :param y: target to be incorporated in new features
        :param source_positions: indices of initial features to be
                                 used as conditions
        :param n_splits: number of folds for feature generation
        :param shuffle: whether to shuffle objects before splitting
        :param random_state: pseudo-random numbers generator seed
                             for shuffling
        :return: transformed feature representation
        """
        new_n_columns = (X.shape[1] +
                         len(self.aggregators) * len(source_positions) -
                         self.drop_source_features * len(source_positions))
        transformed_X = np.full((X.shape[0], new_n_columns), np.nan)
        kf = KFold(n_splits, shuffle, random_state)
        for fit_indices, transform_indices in kf.split(X):
            self.fit(
                X[fit_indices, :],
                y[fit_indices],
                source_positions
            )
            transformed_X[transform_indices, :] = self.transform(
                X[transform_indices, :]
            )
        self.fit(X, y, source_positions)
        return transformed_X
