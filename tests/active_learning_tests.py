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
from dsawl.active_learning.utils import make_committee


def get_data_for_classification() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def get_predictions_of_knn_classifier() -> np.ndarray:
    """
    Get predictions of class probabilities made by
    `KNeighborsClassifier()` trained on data that are returned
    by `get_dataset_and_pool` function and applied to `X_new` from
    its output.

    :return:
        predicted probabilities for new objects
    """
    clf = KNeighborsClassifier()
    X_train, y_train, X_new = get_data_for_classification()
    clf.fit(X_train, y_train)
    predicted_probabilities = clf.predict_proba(X_new)
    return predicted_probabilities


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
        predicted_probabilities = get_predictions_of_knn_classifier()
        execution_result = compute_confidences(predicted_probabilities)
        true_answer = np.array([0.6, 0.6, 0.4, 0.6, 0.6])
        self.assertTrue(np.array_equal(execution_result, true_answer))

    def test_compute_margins(self) -> type(None):
        """
        Test that margins are computed correctly.

        :return:
            None
        """
        predicted_probabilities = get_predictions_of_knn_classifier()
        execution_result = compute_margins(predicted_probabilities)
        true_answer = np.array([0.2, 0.2, 0, 0.4, 0.2])
        self.assertTrue(np.allclose(execution_result, true_answer))

    def test_compute_entropy(self) -> type(None):
        """
        Test that entropy is computed correctly.

        :return:
            None
        """
        predicted_probabilities = get_predictions_of_knn_classifier()
        execution_result = compute_entropy(predicted_probabilities)
        true_answer = np.array(
            [0.67301167, 0.67301167, 1.05492017, 0.95027054, 0.67301167]
        )
        self.assertTrue(np.allclose(execution_result, true_answer))

    def test_compute_committee_divergences(self) -> type(None):
        """
        Test that KL divergences are computed correctly.

        :return:
            None
        """
        clf = KNeighborsClassifier()
        X_train, y_train, X_new = get_data_for_classification()
        splitter = StratifiedKFold()
        committee = make_committee(clf, X_train, y_train, splitter)
        list_of_predicted_probabilities = [
            clf.predict_proba(X_new) for clf in committee
        ]
        execution_result = compute_committee_divergences(
            list_of_predicted_probabilities
        )
        true_answer = np.array([0, 0, 0, 0.09080533, 0])
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
