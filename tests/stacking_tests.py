"""
This module contains tests of code from `../dsawl/stacking`
directory. Code that is tested here provides functionality for
stacking machine learning models on top of other machine learning
models.

@author: Nikolay Lysenko
"""


import unittest
from typing import Tuple

import numpy as np

from sklearn.utils.estimator_checks import check_estimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from dsawl.stacking.stackers import StackingRegressor, StackingClassifier


def get_dataset_for_regression() -> Tuple[np.ndarray, np.ndarray]:
    """
    Get dataset with numerical target variable and two
    numerical features.

    :return:
        features array and target array
    """
    dataset = np.array(
        [[1, 2, 7],
         [2, 3, 11],
         [3, 4, 15],
         [4, 5, 19],
         [5, 6, 23],
         [6, 1, 8],
         [7, 2, 14],
         [8, 3, 16],
         [9, 4, 22],
         [1, 5, 15],
         [2, 6, 21],
         [3, 7, 23],
         [4, 8, 29],
         [5, 9, 31],
         [6, 2, 13]],
        dtype=float
    )
    X = dataset[:, :-1]
    y = dataset[:, -1]
    return X, y


class TestStackingRegressor(unittest.TestCase):
    """
    Tests of `StackingRegressor` class.
    """

    def test_compatibility_with_sklearn(self) -> type(None):
        """
        Test that `sklearn` API is fully supported.

        :return:
            None
        """
        check_estimator(StackingRegressor)

    def test_fit(self) -> type(None):
        """
        Test `fit` method.

        :return:
            None
        """
        X, y = get_dataset_for_regression()
        rgr = StackingRegressor(
            base_estimators_types=[LinearRegression, KNeighborsRegressor],
            base_estimators_params=[dict(), {'n_neighbors': 1}],
            keep_meta_X=True
        )
        rgr.fit(X, y)
        true_meta_X_ = np.array(
            [[6.69395712, 15.0],
             [10.76647173, 15.0],
             [14.83898635, 15.0],
             [18.91150097, 21.0],
             [22.98401559, 23.0],
             [9.74141049, 13.0],
             [13.70235081, 13.0],
             [17.66329114, 13.0],
             [21.62423146, 13.0],
             [15.94394213, 21.0],
             [19.8032967, 15.0],
             [23.92527473, 19.0],
             [28.04725275, 23.0],
             [32.16923077, 23.0],
             [11.94542125, 8.0]]
        )
        self.assertTrue(np.allclose(rgr.meta_X_, true_meta_X_))
        true_coefs_of_base_lr = np.array([1.05304994, 2.97421767])
        self.assertTrue(
            np.allclose(
                rgr.base_estimators_[0].coef_,
                true_coefs_of_base_lr
            )
        )
        true_coefs_of_meta_estimator = np.array([1.01168028, -0.04313311])
        self.assertTrue(
            np.allclose(
                rgr.meta_estimator_.coef_,
                true_coefs_of_meta_estimator
            )
        )

    def test_fit_with_defaults(self) -> type(None):
        """
        Test that `fit` method with all arguments left untouched runs
        (more things can not be tested, because random seeds are not
        set by default).

        :return:
            None
        """
        X, y = get_dataset_for_regression()
        rgr = StackingRegressor()
        rgr.fit(X, y)

    def test_that_fit_raises_on_wrong_lengths(self) -> type(None):
        """
        Test that `fit` method raises an exception if lengths of
        base estimators' types and parameters mismatch.

        :return:
            None
        """
        X, y = get_dataset_for_regression()
        rgr = StackingRegressor(
            base_estimators_types=[LinearRegression, KNeighborsRegressor],
            base_estimators_params=[{'n_neighbors': 1}]
        )
        message = ''
        try:
            rgr.fit(X, y)
        except ValueError as e:
            message = str(e)
        self.assertTrue('Lengths mismatch' in message)

    def test_predict(self) -> type(None):
        """
        Test `predict` method.

        :return:
            None
        """
        X_test = np.array(
            [[1, 5],
             [-3, -4],
             [7, 9],
             [2, 1]],
            dtype=float
        )
        rgr = StackingRegressor(
            base_estimators_types=[LinearRegression, KNeighborsRegressor],
            base_estimators_params=[dict(), {'n_neighbors': 1}]
        )

        # Fit `StackingRegressor` manually.
        X, y = get_dataset_for_regression()
        lr = LinearRegression().fit(X, y)
        kn = KNeighborsRegressor(n_neighbors=1).fit(X, y)
        rgr.base_estimators_ = [lr, kn]
        meta_estimator = LinearRegression()
        meta_estimator.coef_ = np.array([1.01168028, -0.04313311])
        meta_estimator.intercept_ = 0.392229628617
        rgr.meta_estimator_ = meta_estimator

        result = rgr.predict(X_test)
        true_answer = np.array(
            [15.73572972, -15.2612211, 33.47352854, 5.11031499]
        )
        self.assertTrue(np.allclose(result, true_answer))


class TestStackingClassifier(unittest.TestCase):
    """
    Tests of `StackingRegressor` class.
    """

    def test_compatibility_with_sklearn(self) -> type(None):
        """
        Test that `sklearn` API is fully supported.

        :return:
            None
        """
        check_estimator(StackingClassifier)


def main():
    test_loader = unittest.TestLoader()
    suites_list = []
    testers = [
        TestStackingRegressor(),
        TestStackingClassifier()
    ]
    for tester in testers:
        suite = test_loader.loadTestsFromModule(tester)
        suites_list.append(suite)
    overall_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner().run(overall_suite)


if __name__ == '__main__':
    main()
