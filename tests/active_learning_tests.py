"""
This module contains tests of code from `../dsawl/active_learning`
directory. Code that is tested here provides functionality for
extension of learning sample in an efficient way.

@author: Nikolay Lysenko
"""


import unittest
from typing import Tuple

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from dsawl.active_learning.pool_based_sampling import (
    compute_confidences, compute_margins, compute_entropy,
    compute_committee_divergences
)


def get_dataset_and_pool() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get dataset with categorical target variable and two features
    and get feature representation of new unlabelled objects.

    :return:
        train features, train target, new features
    """
    dataset = np.array(
        [[0, 0, 1],
         [1, 0, 1],
         [0, 1, 1],
         [3, 0, 2],
         [2, 1, 2],
         [3, 1, 2],
         [0, 3, 3],
         [1, 2, 3],
         [1, 3, 3]],
        dtype=float)
    X_train = dataset[:, :-1]
    y_train = dataset[:, -1]
    X_new = np.array(
        [[2, 0],
         [0, 2],
         [2, 2],
         [1, 1],
         [1, -4]]
    )
    return X_train, y_train, X_new


class TestRankingFunctions(unittest.TestCase):
    """
    Tests of functions for ranking in pool-based sampling.
    """

    def test_compute_confidence(self) -> type(None):
        """
        Test that confidences are computed correctly.

        :return:
            None
        """
        clf = KNeighborsClassifier()
        X_train, y_train, X_new = get_dataset_and_pool()
        execution_result = compute_confidences(clf, X_train, y_train, X_new)
        true_answer = np.array([0.6, 0.6, 0.4, 0.6, 0.6]).reshape(-1, 1)
        self.assertTrue(np.array_equal(execution_result, true_answer))

    def test_compute_margins(self) -> type(None):
        """
        Test that margins are computed correctly.

        :return:
            None
        """
        clf = KNeighborsClassifier()
        X_train, y_train, X_new = get_dataset_and_pool()
        execution_result = compute_margins(clf, X_train, y_train, X_new)
        true_answer = np.array([0.2, 0.2, 0, 0.4, 0.2]).reshape(-1, 1)
        self.assertTrue(np.allclose(execution_result, true_answer))

    def test_compute_entropy(self) -> type(None):
        """
        Test that entropy is computed correctly.

        :return:
            None
        """
        clf = KNeighborsClassifier()
        X_train, y_train, X_new = get_dataset_and_pool()
        clf.fit(X_train, y_train)
        execution_result = compute_entropy(clf, X_train, y_train, X_new)
        true_answer = np.array(
            [[0.67301167],
             [0.67301167],
             [1.05492017],
             [0.95027054],
             [0.67301167]]
        )
        self.assertTrue(np.allclose(execution_result, true_answer))

    def test_compute_committee_divergences(self) -> type(None):
        """
        Test that KL divergences are computed correctly.

        :return:
            None
        """
        clf = KNeighborsClassifier()
        X_train, y_train, X_new = get_dataset_and_pool()
        splitter = StratifiedKFold()
        execution_result = compute_committee_divergences(
            clf, X_train, y_train, X_new, splitter
        )
        true_answer = np.array(
            [[0],
             [0],
             [0],
             [0.09080533],
             [0]]
        )
        self.assertTrue(np.allclose(execution_result, true_answer))


def main():
    test_loader = unittest.TestLoader()
    suites_list = []
    testers = [
        TestRankingFunctions()
    ]
    for tester in testers:
        suite = test_loader.loadTestsFromModule(tester)
        suites_list.append(suite)
    overall_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner().run(overall_suite)


if __name__ == '__main__':
    main()
