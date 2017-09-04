"""
This module contains tests of code from `src/ooffg` directory.
It is code that provides out-of-fold feature generation functionality.

@author: Nikolay Lysenko
"""


import unittest
from typing import Tuple

import numpy as np

from dsawl.ooffg.features_generator import FeaturesGenerator


def get_regression_dataset() -> Tuple[np.ndarray, np.ndarray]:
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


class TestFeaturesGenerator(unittest.TestCase):
    """
    Tests of `FeatureGenerator` class.
    """

    def test_fit_transform_in_fold(self) -> type(None):
        """
        Test combination of `fit` and `transform`.

        :return: None
        """
        X, y = get_regression_dataset()
        fg = FeaturesGenerator(aggregators=[np.mean, np.median])
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
        X, y = get_regression_dataset()
        fg = FeaturesGenerator(aggregators=[np.mean, np.median])
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


def main():
    test_loader = unittest.TestLoader()
    suites_list = []
    test_classes = [TestFeaturesGenerator()]
    for test_class in test_classes:
        suite = test_loader.loadTestsFromModule(test_class)
        suites_list.append(suite)
    overall_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner().run(overall_suite)


if __name__ == '__main__':
    main()
