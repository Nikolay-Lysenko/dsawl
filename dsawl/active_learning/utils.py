"""
This file contains some auxiliaries.

@author: Nikolay Lysenko
"""


from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold

from dsawl.stacking.stackers import FoldType


def make_committee(
        est: BaseEstimator,
        X_train: np.ndarray, y_train: np.ndarray,
        splitter: Optional[FoldType] = None
        ) -> List[BaseEstimator]:
    """
    Make committee from a single estimator by fitting it to
    various folds.

    :param est:
        estimator instance that has method `fit`
    :param X_train:
        feature representation of training objects
    :param y_train:
        target label
    :param splitter:
        instance that can split data into folds
    :return:
        list of fitted instances
    """
    committee = []
    splitter = splitter or StratifiedKFold()
    for train_index, test_index in splitter.split(X_train, y_train):
        X_curr_train = X_train[train_index]
        y_curr_train = y_train[train_index]
        curr_est = clone(est).fit(X_curr_train, y_curr_train)
        committee.append(curr_est)
    return committee
