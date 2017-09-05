"""
This module is for both in-fold and out-of-fold generation of
features that are derived in the following manner:
1) group target variable by a particular initial feature;
2) apply aggregating function to each group;
3) for each object, use corresponding aggregated value
   as a new feature.

@author: Nikolay Lysenko
"""


from typing import List, Tuple, Callable

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

    Also this class allows avoiding overfitting by generating
    new features based only on out-of-fold values of target.

    :param aggregators: functions that compute aggregates
    """

    def __init__(self, aggregators: List[Callable] = None):
        self.aggregators = [np.mean] if aggregators is None else aggregators
        # TODO: Implement `min_support` argument and/or smoothing.
        self.mappings_ = dict()

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
                k: [agg(y[feature == k]) for agg in self.aggregators]
                for k in np.unique(feature)
            }
        return self

    def transform(
            self,
            X: np.ndarray,
            drop_source_features: bool = True
            ) -> np.ndarray:
        """
        Augment `X` with learnt conditional aggregates.

        :param X: feature representation to be augmented
        :param drop_source_features: drop or keep those of initial
                                     features that are used for
                                     conditioning over them
        :return: transformed data
        """
        transformed_X = X.copy()
        for position, mappings in self.mappings_.items():
            feature = X[:, position]
            n_new = len(next(iter(mappings.values())))
            new_features = np.full((X.shape[0], n_new), np.nan)
            for value, conditional_aggregates in mappings.items():
                new_features[[feature == value]] = \
                    np.tile(conditional_aggregates, (sum(feature == value), 1))
            transformed_X = np.hstack((transformed_X, new_features))
        if drop_source_features:
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
            random_state: int = None,
            drop_source_features: bool = True
            ) -> Tuple[np.ndarray, np.ndarray]:
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
        :param drop_source_features: drop or keep those of initial
                                     features that are used for
                                     conditioning over them
        :return: transformed feature representation
        """
        new_n_columns = (X.shape[1] +
                         len(self.aggregators) * len(source_positions) -
                         drop_source_features * len(source_positions))
        transformed_X = np.full((X.shape[0], new_n_columns), np.nan)
        kf = KFold(n_splits, shuffle, random_state)
        for fit_indices, transform_indices in kf.split(X):
            self.fit(
                X[fit_indices],
                y[fit_indices],
                source_positions
            )
            transformed_X[transform_indices, :] = self.transform(
                X[transform_indices],
                drop_source_features
            )
        self.mappings_ = dict()  # A sort of cleaning up.
        return transformed_X
