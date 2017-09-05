"""
This module contains tests of code from `dsawl/ooffg` directory.
It is code that provides out-of-fold feature generation functionality.

@author: Nikolay Lysenko
"""


import unittest
from typing import Tuple

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from dsawl.ooffg.target_based_features_creator import (
    TargetBasedFeaturesCreator
)
from dsawl.ooffg.estimators import (
    OutOfFoldFeaturesRegressor, OutOfFoldFeaturesClassifier
)


def get_dataset_for_features_creation() -> Tuple[np.ndarray, np.ndarray]:
    """
    Get dataset with one numerical feature, one label-encoded
    categorical feature, and a numerical target variable.

    :return: X and y
    """
    dataset = np.array(
        [[1, 0, 1],
         [2, 0, 2],
         [3, 0, 3],
         [4, 0, 4],
         [10, 0, 10],
         [1, 1, 3],
         [2, 1, 4],
         [3, 1, 5],
         [4, 1, 6],
         [10, 1, 12],
         [1, -1, 5],
         [2, -1, 6],
         [3, -1, 7],
         [4, -1, 8],
         [10, -1, 14]],
        dtype=float)
    X = dataset[:, :-1]
    y = dataset[:, -1]
    return X, y


class TestTargetBasedFeaturesCreator(unittest.TestCase):
    """
    Tests of `TargetBasedFeaturesCreator` class.
    """

    def test_fit_transform(self) -> type(None):
        """
        Test in-fold combination of `fit` and `transform`.

        :return: None
        """
        X, y = get_dataset_for_features_creation()
        fg = TargetBasedFeaturesCreator(aggregators=[np.mean, np.median])
        execution_result = fg.fit_transform(
            X,
            y,
            source_positions=[1]
        )
        true_answer = np.array(
            [[1, 4, 3],
             [2, 4, 3],
             [3, 4, 3],
             [4, 4, 3],
             [10, 4, 3],
             [1, 6, 5],
             [2, 6, 5],
             [3, 6, 5],
             [4, 6, 5],
             [10, 6, 5],
             [1, 8, 7],
             [2, 8, 7],
             [3, 8, 7],
             [4, 8, 7],
             [10, 8, 7]],
            dtype=float
        )
        self.assertTrue(np.array_equal(execution_result, true_answer))

    def test_fit_transform_out_of_fold(self) -> type(None):
        """
        Test `fit_transform_out_of_fold` method.

        :return: None
        """
        X, y = get_dataset_for_features_creation()
        fg = TargetBasedFeaturesCreator(aggregators=[np.mean, np.median])
        execution_result = fg.fit_transform_out_of_fold(
            X,
            y,
            source_positions=[1],
            n_splits=5
        )
        true_answer = np.array(
            [[1, 7, 7],
             [2, 7, 7],
             [3, 7, 7],
             [4, 2, 2],
             [10, 2, 2],
             [1, 6.75, 5.5],
             [2, 7.5, 7.5],
             [3, 7.5, 7.5],
             [4, 7.5, 7.5],
             [10, 4.5, 4.5],
             [1, 29 / 3, 8],
             [2, 29 / 3, 8],
             [3, 5.5, 5.5],
             [4, 5.5, 5.5],
             [10, 5.5, 5.5]],
            dtype=float
        )
        self.assertTrue(np.allclose(execution_result, true_answer))


def get_dataset_for_regression() -> Tuple[np.ndarray, np.ndarray]:
    """
    Get dataset with one numerical feature, one label-encoded
    categorical feature, and a numerical target variable.

    :return: X and y
    """
    dataset = np.array(
        [[3, 0, 6],
         [1, 0, 2],
         [2, 0, 4],
         [1, 1, 1],
         [2, 1, 2],
         [3, 1, 3]],
        dtype=float)
    X = dataset[:, :-1]
    y = dataset[:, -1]
    return X, y


class TestOutOfFoldFeaturesRegressor(unittest.TestCase):
    """
    Tests of `OutOfFoldFeaturesRegressor` class.
    """

    def test_fit(self) -> type(None):
        """
        Test `fit` method.

        :return: None
        """
        X, y = get_dataset_for_regression()
        ooffr = OutOfFoldFeaturesRegressor(
            LinearRegression(),
            n_splits=3
        )
        ooffr.fit(X, y, source_positions=[1])
        learnt_slopes = ooffr.estimator.coef_
        true_answer = np.array([1.8, 0.8])
        self.assertTrue(np.allclose(learnt_slopes, true_answer))

    def test_fit_and_predict(self) -> type(None):
        """
        Test `predict` method as well as `fit` method.
        Predominantly, it is `test` of `predict` method, but
        correct work of `fit` method is required.
        Also note that `fit_predict` must produce different result.

        :return: None
        """
        X, y = get_dataset_for_regression()
        ooffr = OutOfFoldFeaturesRegressor(
            LinearRegression(),
            n_splits=3
        )
        ooffr.fit(X, y, source_positions=[1])
        result = ooffr.predict(X)
        true_answer = np.array([5.8, 2.2, 4, 0.6, 2.4, 4.2])
        self.assertTrue(np.allclose(result, true_answer))

    def test_fit_predict(self) -> type(None):
        """
        Test `fit_predict` method.

        :return: None
        """
        X, y = get_dataset_for_regression()
        ooffr = OutOfFoldFeaturesRegressor(
            LinearRegression(),
            n_splits=3
        )
        result = ooffr.fit_predict(X, y, source_positions=[1])
        true_answer = np.array([5.8, 2.2, 4, 1, 1.6, 3.4])
        self.assertTrue(np.allclose(result, true_answer))


def get_dataset_for_classification() -> Tuple[np.ndarray, np.ndarray]:
    """
    Get dataset with one numerical feature, one label-encoded
    categorical feature, and a binary class label.

    :return: X and y
    """
    dataset = np.array(
        [[3, 0, 1],
         [1, 0, 0],
         [2, 0, 1],
         [1, 1, 0],
         [2, 1, 0],
         [3, 1, 1]],
        dtype=float)
    X = dataset[:, :-1]
    y = dataset[:, -1]
    return X, y


class TestOutOfFoldFeaturesClassifier(unittest.TestCase):
    """
    Tests of `OutOfFoldFeaturesClassifier` class.
    """

    def test_fit(self) -> type(None):
        """
        Test `fit` method.

        :return: None
        """
        X, y = get_dataset_for_classification()
        ooffc = OutOfFoldFeaturesClassifier(
            LogisticRegression(random_state=361),
            n_splits=3
        )
        ooffc.fit(X, y, source_positions=[1])
        learnt_slopes = ooffc.estimator.coef_
        true_answer = np.array([0.48251806, -0.16291334])
        self.assertTrue(np.allclose(learnt_slopes, true_answer))

    def test_fit_and_predict_proba(self) -> type(None):
        """
        Test `predict_proba` method as well as `fit` method.
        Predominantly, it is `test` of `predict` method, but
        correct work of `fit` method is required.
        Also note that `fit_predict` must produce different result.

        :return: None
        """
        X, y = get_dataset_for_classification()
        ooffc = OutOfFoldFeaturesClassifier(
            LogisticRegression(random_state=361),
            n_splits=3
        )
        ooffc.fit(X, y, source_positions=[1])
        result = ooffc.predict_proba(X)[:, 1]
        true_answer = np.array([0.69413293, 0.46368326, 0.58346035,
                                0.47721111, 0.59659543, 0.70553938])
        self.assertTrue(np.allclose(result, true_answer))

    def test_fit_predict_proba(self) -> type(None):
        """
        Test `fit_predict_proba` method.

        :return: None
        """
        X, y = get_dataset_for_classification()
        ooffc = OutOfFoldFeaturesClassifier(
            LogisticRegression(random_state=361),
            n_splits=3
        )
        result = ooffc.fit_predict_proba(X, y, source_positions=[1])[:, 1]
        true_answer = np.array([0.68248347, 0.45020866, 0.59004395,
                                0.47044176, 0.60959347, 0.71669408])
        self.assertTrue(np.allclose(result, true_answer))


def main():
    test_loader = unittest.TestLoader()
    suites_list = []
    testers = [
        TestTargetBasedFeaturesCreator(),
        TestOutOfFoldFeaturesRegressor(),
        TestOutOfFoldFeaturesClassifier()
    ]
    for tester in testers:
        suite = test_loader.loadTestsFromModule(tester)
        suites_list.append(suite)
    overall_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner().run(overall_suite)


if __name__ == '__main__':
    main()
