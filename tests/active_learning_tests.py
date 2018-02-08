"""
This module contains tests of code from `../dsawl/active_learning`
directory. Code that is tested here provides functionality for
extension of learning sample in an efficient way.

@author: Nikolay Lysenko
"""


import unittest
from typing import Tuple

import numpy as np
from sklearn.neighbors import (
    KNeighborsClassifier, KNeighborsRegressor, KernelDensity
)

from dsawl.active_learning.scoring_functions import (
    compute_confidences, compute_margins, compute_entropy,
    compute_committee_divergences, compute_committee_variances,
    compute_estimations_of_variance
)
from dsawl.active_learning.scorers import (
    UncertaintyScorerForClassification, CommitteeScorer,
    VarianceScorerForRegression, RandomScorer, DensityScorer,
)
from dsawl.active_learning.pool_based_sampling import (
    CombinedSamplerFromPool
)


def get_predictions() -> np.ndarray:
    """
    Get data that can be treated both as predicted probabilities of
    class labels or as predictions of multiple continuous values.

    :return:
        predictions
    """
    predictions = np.array(
        [[0.2, 0.5, 0.3],
         [0.9, 0.1, 0],
         [0.33, 0.34, 0.33],
         [0.5, 0.5, 0]]
    )
    return predictions


class TestScoringFunctions(unittest.TestCase):
    """
    Tests of scoring functions.
    """

    def test_compute_confidences(self) -> type(None):
        """
        Test that `compute_confidences` function works correctly.

        :return:
            None
        """
        predicted_probabilities = get_predictions()
        execution_result = compute_confidences(predicted_probabilities)
        true_answer = np.array([0.5, 0.9, 0.34, 0.5])
        self.assertTrue(np.allclose(execution_result, true_answer))

    def test_compute_margins(self) -> type(None):
        """
        Test that `compute_margins` function works correctly.

        :return:
            None
        """
        predicted_probabilities = get_predictions()
        execution_result = compute_margins(predicted_probabilities)
        true_answer = np.array([0.2, 0.8, 0.01, 0.])
        self.assertTrue(np.allclose(execution_result, true_answer))

    def test_compute_entropy(self) -> type(None):
        """
        Test that `compute_entropy` function works correctly.

        :return:
            None
        """
        predicted_probabilities = get_predictions()
        execution_result = compute_entropy(predicted_probabilities)
        true_answer = np.array(
            [1.02965301, 0.32508297, 1.09851262, 0.69314718]
        )
        self.assertTrue(np.allclose(execution_result, true_answer))

    def test_compute_committee_divergences(self) -> type(None):
        """
        Test that `compute_committee_divergences` function
        works correctly.

        :return:
            None
        """
        stub = get_predictions()
        list_of_predicted_probabilities = [stub[:2, :], stub[2:, :]]
        execution_result = compute_committee_divergences(
            list_of_predicted_probabilities
        )
        true_answer = np.array([0.0321534, 0.20349845])
        self.assertTrue(np.allclose(execution_result, true_answer))

    def test_compute_committee_variances(self) -> type(None):
        """
        Test that `compute_committee_variances` function
        works correctly.

        :return:
            None
        """
        stub = get_predictions()
        list_of_predictions = [stub[:2, 0], stub[2:, 0]]
        execution_result = compute_committee_variances(list_of_predictions)
        true_answer = np.array([0.004225, 0.04])
        self.assertTrue(np.allclose(execution_result, true_answer))

    def test_compute_estimations_of_variance(self) -> type(None):
        """
        Test that `compute_estimations_of_variance` function
        works correctly.

        :return:
            None
        """
        predictions = np.array([0.1, 0.2, 0.3])
        predictions_of_square = np.array([0.2, 0.3, 0.4])
        execution_result = compute_estimations_of_variance(
            predictions, predictions_of_square
        )
        true_answer = np.array([0.19, 0.26, 0.31])
        self.assertTrue(np.allclose(execution_result, true_answer))


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
        Test that `score` method works correctly if
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
        Test that `score` method works correctly if
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
        Test that `score` method works correctly if
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

    def test_get_tools_and_set_tools(self) -> type(None):
        """
        Test that `get_tools` and `set_tools` methods work
        correctly.

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
        predictions = scorer.get_tools().predict(X_new)
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
        execution_result = scorer.get_tools().predict(X_new)
        true_answer = np.array([2, 3, 2, 1, 1])
        self.assertTrue(np.array_equal(execution_result, true_answer))
        scorer.update_tools(X_train[:-1, :], y_train[:-1], clf)
        execution_result = scorer.get_tools().predict(X_new)
        true_answer = np.array([2, 1, 2, 1, 1])
        self.assertTrue(np.array_equal(execution_result, true_answer))


class TestCommitteeScorer(unittest.TestCase):
    """
    Tests of `CommitteeScorer` class.
    """

    def test_score_with_divergences(self) -> type(None):
        """
        Test that `score` method works correctly if it is
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
        Test that `score` method works correctly if it is
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

    def test_get_tools_and_set_tools(self) -> type(None):
        """
        Test that `get_tools` and `set_tools` methods work
        correctly.

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
        predictions = scorer.get_tools()[0].predict(X_new)
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
        execution_result = [clf.predict(X_new) for clf in scorer.get_tools()]
        true_answer = [np.array([1, 1, 2, 1, 1]) for _ in range(3)]
        for result, answer in zip(execution_result, true_answer):
            self.assertTrue(np.array_equal(result, answer))
        scorer.update_tools(
            np.vstack((X_train, X_train[1, :])),
            np.hstack((y_train, y_train[1])),
            clf
        )
        execution_result = [clf.predict(X_new) for clf in scorer.get_tools()]
        true_answer = [np.array([1, 1, 2, 1, 1]) for _ in range(3)]
        for result, answer in zip(execution_result, true_answer):
            self.assertTrue(np.array_equal(result, answer))


class TestVarianceScorerForRegression(unittest.TestCase):
    """
    Tests of `VarianceScorerForRegression` class.
    """

    def test_score_with_estimation_of_target_variance(self) -> type(None):
        """
        Test that `score` method works correctly if scoring
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

    def test_get_tools_and_set_tools(self) -> type(None):
        """
        Test that `get_tools` and `set_tools` methods work
        correctly.

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
        predictions = scorer.get_tools()['target'].predict(X_new)
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
        execution_result = scorer.get_tools()['target'].predict(X_new)
        true_answer = np.array([1.6, 2.2, 2.2, 1.6, 1.4])
        self.assertTrue(np.array_equal(execution_result, true_answer))
        scorer.update_tools(
            X_train[:-1, :], y_train[:-1], KNeighborsRegressor()
        )
        execution_result = scorer.get_tools()['target'].predict(X_new)
        true_answer = np.array([1.6, 1.8, 2.4, 1.6, 1.4])
        self.assertTrue(np.array_equal(execution_result, true_answer))


class TestRandomScorer(unittest.TestCase):
    """
    Tests of `RandomScorer` class.
    """

    def test_score(self) -> type(None):
        """
        Test that `score` method runs.

        :return:
            None
        """
        X_train, y_train, X_new = get_dataset_and_pool()
        scorer = RandomScorer()
        execution_result = scorer.score(X_new)
        self.assertTrue(len(execution_result) == len(X_new))

    def test_get_tools_and_set_tools(self) -> type(None):
        """
        Test that `get_tools` and `set_tools` methods work
        correctly.

        :return:
            None
        """
        scorer = RandomScorer()
        clf = KNeighborsClassifier()
        scorer.set_tools(clf)
        execution_result = scorer.get_tools()
        self.assertTrue(execution_result is None)

    def test_update_tools(self) -> type(None):
        """
        Test that `update_tools` method works correctly.

        :return:
            None
        """
        X_train, y_train, X_new = get_dataset_and_pool()
        scorer = RandomScorer()
        scorer.update_tools(X_train, y_train)
        execution_result = scorer.get_tools()
        self.assertTrue(execution_result is None)


class TestDensityScorer(unittest.TestCase):
    """
    Tests of `DensityScorer` class.
    """

    def test_score(self) -> type(None):
        """
        Test that `score` method works correctly.

        :return:
            None
        """
        X_train, y_train, X_new = get_dataset_and_pool()
        est = KernelDensity()
        est.fit(X_train, y_train)
        scorer = DensityScorer(est)
        execution_result = scorer.score(X_new)
        true_answer = np.array(
            [3.12072551, 3.12072551, 3.20416149, 2.86297789, 11.471556]
        )
        self.assertTrue(np.allclose(execution_result, true_answer))

    def test_get_tools_and_set_tools(self) -> type(None):
        """
        Test that `get_tools` and `set_tools` methods work
        correctly.

        :return:
            None
        """
        X_train, y_train, X_new = get_dataset_and_pool()
        est = KernelDensity()
        est.fit(X_train, y_train)
        scorer = DensityScorer(est)
        another_est = KernelDensity()
        another_est.fit(X_train[:-1, :], y_train[:-1])
        scorer.set_tools(another_est)
        estimates = scorer.get_tools().score_samples(X_new)
        another_estimates = another_est.score_samples(X_new)
        self.assertTrue(np.array_equal(estimates, another_estimates))

    def test_update_tools(self) -> type(None):
        """
        Test that `update_tools` method works correctly.

        :return:
            None
        """
        X_train, y_train, X_new = get_dataset_and_pool()
        est = KernelDensity()
        scorer = DensityScorer(est)
        scorer.update_tools(X_train, y_train)
        execution_result = scorer.score(X_new)
        true_answer = np.array(
            [3.12072551, 3.12072551, 3.20416149, 2.86297789, 11.471556]
        )
        self.assertTrue(np.allclose(execution_result, true_answer))
        scorer.update_tools(
            X_train[:-1, :], y_train[:-1], KernelDensity()
        )
        execution_result = scorer.score(X_new)
        true_answer = np.array(
            [3.00564647, 3.16244687, 3.26104477, 2.78801309, 11.353773]
        )
        self.assertTrue(np.allclose(execution_result, true_answer))


class TestCombinedSamplerFromPool(unittest.TestCase):
    """
    Tests of `CombinedSamplerFromPool` class.
    """

    def test_pick_new_objects_with_one_scorer(self) -> type(None):
        """
        Test that `pick_new_objects` method works correctly
        when there is only one scorer.

        :return:
            None
        """
        X_train, y_train, X_new = get_dataset_and_pool()
        clf = KNeighborsClassifier()
        clf.fit(X_train, y_train)
        scorer = UncertaintyScorerForClassification(
            compute_confidences, revert_sign=True, clf=clf
        )
        sampler = CombinedSamplerFromPool([scorer])
        execution_result = sampler.pick_new_objects(X_new)
        true_answer = [2]
        self.assertTrue(execution_result == true_answer)

    def test_pick_new_objects_with_two_scorers(self) -> type(None):
        """
        Test that `pick_new_objects` method works correctly
        when there are two scorers with non-zero probabilities.

        :return:
            None
        """
        X_train, y_train, X_new = get_dataset_and_pool()
        clf = KNeighborsClassifier()
        clf.fit(X_train, y_train)
        scorers = [
            UncertaintyScorerForClassification(
                compute_confidences, revert_sign=True, clf=clf
            ),
            RandomScorer()
        ]
        sampler = CombinedSamplerFromPool(
            scorers, scorers_probabilities=[0.2, 0.8]
        )
        picked_indices = sampler.pick_new_objects(X_new, n_to_pick=2)
        self.assertTrue(len(picked_indices) == 2)
        for index in picked_indices:
            self.assertTrue(0 <= index < len(X_new))
        self.assertTrue(picked_indices[0] != picked_indices[1])

    def test_get_tools(self) -> type(None):
        """
        Test that `get_tools` method works correctly.

        :return:
            None
        """
        X_train, y_train, X_new = get_dataset_and_pool()
        clf = KNeighborsClassifier()
        clf.fit(X_train, y_train)
        scorer = CommitteeScorer(
            compute_confidences, revert_sign=True, committee=[clf]
        )
        sampler = CombinedSamplerFromPool([scorer])
        another_clf = sampler.get_tools(0)[0]
        predictions = clf.predict(X_new)
        another_predictions = another_clf.predict(X_new)
        self.assertTrue(np.array_equal(predictions, another_predictions))
        yet_another_clf = sampler.get_tools()[0][0]
        yet_another_predictions = yet_another_clf.predict(X_new)
        self.assertTrue(np.array_equal(predictions, yet_another_predictions))

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
            compute_confidences, revert_sign=True, clf=clf
        )
        another_scorer = UncertaintyScorerForClassification(
            compute_confidences, revert_sign=True
        )
        sampler = CombinedSamplerFromPool([scorer])
        another_sampler = CombinedSamplerFromPool([another_scorer])
        another_sampler.set_tools(clf, scorer_id=0)
        execution_result = sampler.pick_new_objects(X_new)
        another_execution_result = another_sampler.pick_new_objects(X_new)
        self.assertTrue(execution_result == another_execution_result)

    def test_update_tools(self) -> type(None):
        """
        Test that `set_tools` method works correctly.

        :return:
            None
        """
        X_train, y_train, X_new = get_dataset_and_pool()
        clf = KNeighborsClassifier()
        clf.fit(X_train, y_train)
        scorer = UncertaintyScorerForClassification(
            compute_confidences, revert_sign=True, clf=clf
        )
        another_clf = KNeighborsClassifier()
        another_scorer = UncertaintyScorerForClassification(
            compute_confidences, revert_sign=True, clf=another_clf
        )
        sampler = CombinedSamplerFromPool([scorer])
        another_sampler = CombinedSamplerFromPool([another_scorer])
        another_sampler.update_tools(X_train, y_train, scorer_id=0)
        execution_result = sampler.pick_new_objects(X_new)
        another_execution_result = another_sampler.pick_new_objects(X_new)
        self.assertTrue(execution_result == another_execution_result)
        another_sampler.update_tools(X_train, y_train)
        yet_another_execution_result = another_sampler.pick_new_objects(X_new)
        self.assertTrue(execution_result == yet_another_execution_result)


def main():
    test_loader = unittest.TestLoader()
    suites_list = []
    testers = [
        TestScoringFunctions(),
        TestUncertaintyScorerForClassification(),
        TestCommitteeScorer(),
        TestVarianceScorerForRegression(),
        TestRandomScorer(),
        TestDensityScorer(),
        TestCombinedSamplerFromPool()
    ]
    for tester in testers:
        suite = test_loader.loadTestsFromModule(tester)
        suites_list.append(suite)
    overall_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner().run(overall_suite)


if __name__ == '__main__':
    main()
