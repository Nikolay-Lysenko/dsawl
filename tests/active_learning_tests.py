"""
This module contains tests of code from `../dsawl/active_learning`
directory. Code that is tested here provides functionality for
extension of learning sample in an efficient way.

@author: Nikolay Lysenko
"""


import unittest
from typing import Tuple

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from dsawl.active_learning.pool_based_sampling import (
    compute_confidences, compute_margins, compute_entropy,
    compute_committee_divergences, compute_committee_variances,
    compute_estimations_of_variance,
    UncertaintyScorerForClassification, CommitteeScorer,
    VarianceScorerForRegression,
    EpsilonGreedyPickerFromPool
)


def get_dataset_and_pool() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get dataset with target variable that can be considered both
    categorical or numerical and two numerical features, also
    get feature representation of new unlabelled objects.

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
    by `get_dataset_and_pool` function and applied to `X_new` that
    also comes from this function's output.

    :return:
        predicted probabilities for new objects
    """
    clf = KNeighborsClassifier()
    X_train, y_train, X_new = get_dataset_and_pool()
    clf.fit(X_train, y_train)
    predicted_probabilities = clf.predict_proba(X_new)
    return predicted_probabilities


class TestUncertaintyScorerForClassification(unittest.TestCase):
    """
    Tests of `UncertaintyScorerForClassification` class.
    """

    def test_score_with_confidences(self) -> type(None):
        """
        Test that `score` method` works correctly if
        scoring is based on confidences.

        :return:
            None
        """
        X_train, y_train, X_new = get_dataset_and_pool()
        clf = KNeighborsClassifier()
        clf.fit(X_train, y_train)
        scorer = UncertaintyScorerForClassification(
            compute_confidences,
            revert_sign=True,
            clf=clf
        )
        execution_result = scorer.score(X_new)
        true_answer = np.array([-0.6, -0.6, -0.4, -0.6, -0.6])
        self.assertTrue(np.array_equal(execution_result, true_answer))

    def test_score_with_margins(self) -> type(None):
        """
        Test that `score` method` works correctly if
        scoring is based on margins.

        :return:
            None
        """
        X_train, y_train, X_new = get_dataset_and_pool()
        clf = KNeighborsClassifier()
        clf.fit(X_train, y_train)
        scorer = UncertaintyScorerForClassification(
            compute_margins,
            revert_sign=True,
            clf=clf
        )
        execution_result = scorer.score(X_new)
        true_answer = np.array([-0.2, -0.2, 0, -0.4, -0.2])
        self.assertTrue(np.allclose(execution_result, true_answer))

    def test_score_with_entropy(self) -> type(None):
        """
        Test that `score` method` works correctly if
        scoring is based on entropy.

        :return:
            None
        """
        X_train, y_train, X_new = get_dataset_and_pool()
        clf = KNeighborsClassifier()
        clf.fit(X_train, y_train)
        scorer = UncertaintyScorerForClassification(
            compute_entropy,
            clf=clf
        )
        execution_result = scorer.score(X_new)
        true_answer = np.array(
            [0.67301167, 0.67301167, 1.05492017, 0.95027054, 0.67301167]
        )
        self.assertTrue(np.allclose(execution_result, true_answer))

    def test_set_tools(self) -> type(None):
        """
        Test that `set_tools` method works correctly.

        :return:
            None
        """
        X_train, y_train, X_new = get_dataset_and_pool()
        clf = KNeighborsClassifier()
        clf.fit(X_train, y_train)
        scorer = UncertaintyScorerForClassification(
            compute_entropy,
            clf=clf
        )
        another_clf = KNeighborsClassifier()
        another_clf.fit(X_train[:-1, :], y_train[:-1])
        scorer.set_tools(another_clf)
        predictions = scorer.clf.predict(X_new)
        another_predictions = another_clf.predict(X_new)
        self.assertTrue(np.array_equal(predictions, another_predictions))

    def test_update_tools(self) -> type(None):
        """
        Test that `update_tools` method works correctly.

        :return:
            None
        """
        X_train, y_train, X_new = get_dataset_and_pool()
        clf = KNeighborsClassifier()
        scorer = UncertaintyScorerForClassification(
            compute_entropy,
            clf=clf
        )
        scorer.update_tools(X_train, y_train)
        execution_result = scorer.clf.predict(X_new)
        true_answer = np.array([2, 3, 2, 1, 1])
        self.assertTrue(np.array_equal(execution_result, true_answer))
        scorer.update_tools(X_train[:-1, :], y_train[:-1], clf)
        execution_result = scorer.clf.predict(X_new)
        true_answer = np.array([2, 1, 2, 1, 1])
        self.assertTrue(np.array_equal(execution_result, true_answer))


class TestCommitteeScorer(unittest.TestCase):
    """
    Tests of `CommitteeScorer` class.
    """

    def test_score_with_divergences(self) -> type(None):
        """
        Test that `score` method` works correctly if it is
        a classification problem and scoring is based on
        Kullback-Leibler divergence.

        :return:
            None
        """
        X_train, y_train, X_new = get_dataset_and_pool()
        scorer = CommitteeScorer(
            compute_committee_divergences
        )
        scorer.update_tools(X_train, y_train, KNeighborsClassifier())
        execution_result = scorer.score(X_new)
        true_answer = np.array([0, 0, 0, 0.09080533, 0])
        self.assertTrue(np.allclose(execution_result, true_answer))

    def test_score_with_variances(self) -> type(None):
        """
        Test that `score` method` works correctly if it is
        a regression problem and scoring is based on variance of
        predictions.

        :return:
            None
        """
        X_train, y_train, X_new = get_dataset_and_pool()
        scorer = CommitteeScorer(
            compute_committee_variances,
            is_classification=False
        )
        scorer.update_tools(X_train, y_train, KNeighborsRegressor())
        execution_result = scorer.score(X_new)
        true_answer = np.array([0, 0, 0, 0.008888889, 0])
        self.assertTrue(np.allclose(execution_result, true_answer))

    def test_set_tools(self) -> type(None):
        """
        Test that `set_tools` method works correctly.

        :return:
            None
        """
        X_train, y_train, X_new = get_dataset_and_pool()
        clf = KNeighborsClassifier()
        clf.fit(X_train, y_train)
        scorer = CommitteeScorer(
            compute_entropy,
            committee=[clf]
        )
        another_clf = KNeighborsClassifier()
        another_clf.fit(X_train[:-1, :], y_train[:-1])
        scorer.set_tools([another_clf])
        predictions = scorer.committee[0].predict(X_new)
        another_predictions = another_clf.predict(X_new)
        self.assertTrue(np.array_equal(predictions, another_predictions))

    def test_update_tools(self) -> type(None):
        """
        Test that `update_tools` method works correctly.

        :return:
            None
        """
        X_train, y_train, X_new = get_dataset_and_pool()
        clf = KNeighborsClassifier()
        scorer = CommitteeScorer(
            compute_entropy,
            committee=[clf]
        )
        scorer.update_tools(X_train, y_train)
        execution_result = [clf.predict(X_new) for clf in scorer.committee]
        true_answer = [np.array([1, 1, 2, 1, 1]) for _ in range(3)]
        for result, answer in zip(execution_result, true_answer):
            self.assertTrue(np.array_equal(result, answer))
        scorer.update_tools(
            np.vstack((X_train, X_train[1, :])),
            np.hstack((y_train, y_train[1])),
            clf
        )
        execution_result = [clf.predict(X_new) for clf in scorer.committee]
        true_answer = [np.array([1, 1, 2, 1, 1]) for _ in range(3)]
        for result, answer in zip(execution_result, true_answer):
            self.assertTrue(np.array_equal(result, answer))


class TestVarianceScorerForRegression(unittest.TestCase):
    """
    Tests of `VarianceScorerForRegression` class.
    """

    def test_score_with_estimation_of_target_variance(self) -> type(None):
        """
        Test that `score` method` works correctly if scoring
        is based on estimation of target variable's variance.

        :return:
            None
        """
        X_train, y_train, X_new = get_dataset_and_pool()
        scorer = VarianceScorerForRegression(
            compute_estimations_of_variance
        )
        scorer.update_tools(X_train, y_train, KNeighborsRegressor())
        execution_result = scorer.score(X_new)
        true_answer = np.array([0.24, 0.96, 0.56, 0.64, 0.24])
        self.assertTrue(np.allclose(execution_result, true_answer))

    def test_set_tools(self) -> type(None):
        """
        Test that `set_tools` method works correctly.

        :return:
            None
        """
        X_train, y_train, X_new = get_dataset_and_pool()
        rgrs = {
            'target': KNeighborsRegressor().fit(X_train, y_train),
            'target^2': KNeighborsRegressor().fit(X_train, y_train ** 2)
        }
        scorer = VarianceScorerForRegression(
            compute_entropy,
            rgrs=rgrs
        )
        another_rgrs = {
            'target': KNeighborsRegressor().fit(X_train[:-1, :], y_train[:-1]),
            'target^2': KNeighborsRegressor().fit(
                X_train[:-1, :], y_train[:-1] ** 2
            )
        }
        scorer.set_tools(another_rgrs)
        predictions = scorer.rgrs['target'].predict(X_new)
        another_predictions = another_rgrs['target'].predict(X_new)
        self.assertTrue(np.array_equal(predictions, another_predictions))

    def test_update_tools(self) -> type(None):
        """
        Test that `update_tools` method works correctly.

        :return:
            None
        """
        X_train, y_train, X_new = get_dataset_and_pool()
        rgrs = {
            'target': KNeighborsRegressor(),
            'target^2': KNeighborsRegressor()
        }
        scorer = VarianceScorerForRegression(
            compute_entropy,
            rgrs=rgrs
        )
        scorer.update_tools(X_train, y_train)
        execution_result = scorer.rgrs['target'].predict(X_new)
        true_answer = np.array([1.6, 2.2, 2.2, 1.6, 1.4])
        self.assertTrue(np.array_equal(execution_result, true_answer))
        scorer.update_tools(
            X_train[:-1, :], y_train[:-1], KNeighborsRegressor()
        )
        execution_result = scorer.rgrs['target'].predict(X_new)
        true_answer = np.array([1.6, 1.8, 2.4, 1.6, 1.4])
        self.assertTrue(np.array_equal(execution_result, true_answer))


class TestEpsilonGreedyPickerFromPool(unittest.TestCase):
    """
    Tests of `EpsilonGreedyPickerFromPool` class.
    """

    def test_pick_new_objects_with_zero_exploration(self) -> type(None):
        """
        Test that `pick_new_objects` method works correctly
        when there are no exploration.

        :return:
            None
        """
        X_train, y_train, X_new = get_dataset_and_pool()
        clf = KNeighborsClassifier()
        clf.fit(X_train, y_train)
        scorer = UncertaintyScorerForClassification(
            compute_confidences, revert_sign=True, clf=clf
        )
        picker = EpsilonGreedyPickerFromPool(scorer, exploration_probability=0)
        execution_result = picker.pick_new_objects(X_new)
        true_answer = [2]
        self.assertTrue(execution_result == true_answer)

    def test_pick_new_objects_with_full_exploration(self) -> type(None):
        """
        Test that `pick_new_objects` method works correctly
        when there are no exploitation.

        :return:
            None
        """
        X_train, y_train, X_new = get_dataset_and_pool()
        clf = KNeighborsClassifier()
        clf.fit(X_train, y_train)
        scorer = UncertaintyScorerForClassification(
            compute_confidences, revert_sign=True, clf=clf
        )
        picker = EpsilonGreedyPickerFromPool(scorer, exploration_probability=1)
        picked_indices = picker.pick_new_objects(X_new, n_to_pick=2)
        self.assertTrue(len(picked_indices) == 2)
        for index in picked_indices:
            self.assertTrue(0 <= index < len(X_new))


def main():
    test_loader = unittest.TestLoader()
    suites_list = []
    testers = [
        TestUncertaintyScorerForClassification(),
        TestCommitteeScorer(),
        TestVarianceScorerForRegression(),
        TestEpsilonGreedyPickerFromPool()
    ]
    for tester in testers:
        suite = test_loader.loadTestsFromModule(tester)
        suites_list.append(suite)
    overall_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner().run(overall_suite)


if __name__ == '__main__':
    main()