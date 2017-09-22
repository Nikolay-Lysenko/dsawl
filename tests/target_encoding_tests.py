"""
This module contains tests of code from `../dsawl/target_encoding`
directory. Code that is tested here provides functionality for
out-of-fold feature generation from target variable.

@author: Nikolay Lysenko
"""


import unittest
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, LogisticRegression

from dsawl.target_encoding.target_encoder import TargetEncoder
from dsawl.target_encoding.estimators import (
    OutOfFoldTargetEncodingRegressor, OutOfFoldTargetEncodingClassifier
)


def get_dataset_for_features_creation() -> Tuple[np.ndarray, np.ndarray]:
    """
    Get dataset with numerical target variable and two features
    such that they can be interpreted both as numerical or as
    categorical.

    :return:
        features array and target array
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


class TestTargetEncoder(unittest.TestCase):
    """
    Tests of `TargetEncoder` class.
    """

    def test_fit_transform(self) -> type(None):
        """
        Test in-fold combination of `fit` and `transform` on data
        with one numerical feature and one categorical feature.

        :return:
            None
        """
        X, y = get_dataset_for_features_creation()
        target_encoder = TargetEncoder(aggregators=[np.mean, np.median])
        execution_result = target_encoder.fit_transform(
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

    def test_fit_transform_with_more_source_positions(self) -> type(None):
        """
        Test in-fold combination of `fit` and `transform` on data
        with two categorical features.

        :return:
            None
        """
        X, y = get_dataset_for_features_creation()
        target_encoder = TargetEncoder(aggregators=[np.mean, np.median])
        execution_result = target_encoder.fit_transform(
            X,
            y,
            source_positions=[0, 1]
        )
        true_answer = np.array(
            [[3, 3, 4, 3],
             [4, 4, 4, 3],
             [5, 5, 4, 3],
             [6, 6, 4, 3],
             [12, 12, 4, 3],
             [3, 3, 6, 5],
             [4, 4, 6, 5],
             [5, 5, 6, 5],
             [6, 6, 6, 5],
             [12, 12, 6, 5],
             [3, 3, 8, 7],
             [4, 4, 8, 7],
             [5, 5, 8, 7],
             [6, 6, 8, 7],
             [12, 12, 8, 7]],
            dtype=float
        )
        self.assertTrue(np.array_equal(execution_result, true_answer))

    def test_fit_transform_with_keeping_of_sources(self) -> type(None):
        """
        Test in-fold combination of `fit` and `transform` on data
        with one numerical feature and one categorical feature
        that must be kept.

        :return:
            None
        """
        X, y = get_dataset_for_features_creation()
        target_encoder = TargetEncoder(
            aggregators=[np.mean],
            drop_source_features=False
        )
        execution_result = target_encoder.fit_transform(
            X,
            y,
            source_positions=[1]
        )
        true_answer = np.array(
            [[1, 0, 4],
             [2, 0, 4],
             [3, 0, 4],
             [4, 0, 4],
             [10, 0, 4],
             [1, 1, 6],
             [2, 1, 6],
             [3, 1, 6],
             [4, 1, 6],
             [10, 1, 6],
             [1, -1, 8],
             [2, -1, 8],
             [3, -1, 8],
             [4, -1, 8],
             [10, -1, 8]],
            dtype=float
        )
        self.assertTrue(np.array_equal(execution_result, true_answer))

    def test_fit_transform_with_smoothing(self) -> type(None):
        """
        Test in-fold combination of `fit` and `transform`
        with smoothing.

        :return:
            None
        """
        X, y = get_dataset_for_features_creation()
        target_encoder = TargetEncoder(
            aggregators=[np.mean, np.median],
            smoothing_strength=5
        )
        execution_result = target_encoder.fit_transform(
            X,
            y,
            source_positions=[1]
        )
        true_answer = np.array(
            [[1, 5, 4],
             [2, 5, 4],
             [3, 5, 4],
             [4, 5, 4],
             [10, 5, 4],
             [1, 6, 5],
             [2, 6, 5],
             [3, 6, 5],
             [4, 6, 5],
             [10, 6, 5],
             [1, 7, 6],
             [2, 7, 6],
             [3, 7, 6],
             [4, 7, 6],
             [10, 7, 6]],
            dtype=float
        )
        self.assertTrue(np.array_equal(execution_result, true_answer))

    def test_fit_transform_with_min_frequency(self) -> type(None):
        """
        Test in-fold combination of `fit` and `transform`
        with threshold on number of occurrences.

        :return:
            None
        """
        X, y = get_dataset_for_features_creation()
        X = X[:-1, :]
        y = y[:-1]
        target_encoder = TargetEncoder(
            aggregators=[np.mean, np.median],
            min_frequency=5
        )
        execution_result = target_encoder.fit_transform(
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
             [1, 76 / 14, 5],
             [2, 76 / 14, 5],
             [3, 76 / 14, 5],
             [4, 76 / 14, 5]],
            dtype=float
        )
        self.assertTrue(np.allclose(execution_result, true_answer))

    def test_fit_transform_out_of_fold(self) -> type(None):
        """
        Test `fit_transform_out_of_fold` method on data
        with one numerical feature and one categorical feature.

        :return:
            None
        """
        X, y = get_dataset_for_features_creation()
        target_encoder = TargetEncoder(
            aggregators=[np.mean, np.median],
            splitter=KFold(n_splits=5)
        )
        execution_result = target_encoder.fit_transform_out_of_fold(
            X,
            y,
            source_positions=[1]
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

    def test_fit_transform_out_of_fold_with_more_sources(self) -> type(None):
        """
        Test `fit_transform_out_of_fold` method on data
        with two categorical features.

        :return:
            None
        """
        X, y = get_dataset_for_features_creation()
        target_encoder = TargetEncoder(
            aggregators=[np.mean, np.median],
            splitter=KFold(n_splits=5)
        )
        execution_result = target_encoder.fit_transform_out_of_fold(
            X,
            y,
            source_positions=[0, 1]
        )
        true_answer = np.array(
            [[4, 4, 7, 7],
             [5, 5, 7, 7],
             [6, 6, 7, 7],
             [7, 7, 2, 2],
             [13, 13, 2, 2],
             [3, 3, 6.75, 5.5],
             [4, 4, 7.5, 7.5],
             [5, 5, 7.5, 7.5],
             [6, 6, 7.5, 7.5],
             [12, 12, 4.5, 4.5],
             [2, 2, 29 / 3, 8],
             [3, 3, 29 / 3, 8],
             [4, 4, 5.5, 5.5],
             [5, 5, 5.5, 5.5],
             [11, 11, 5.5, 5.5]],
            dtype=float
        )
        self.assertTrue(np.allclose(execution_result, true_answer))

    def test_fit_transform_out_of_fold_with_keeping_of_sources(
            self
            ) -> type(None):
        """
        Test `fit_transform_out_of_fold` method on data
        with one numerical feature and one categorical feature
        that should be kept.

        :return:
            None
        """
        X, y = get_dataset_for_features_creation()
        X[0, 1] = 2  # Make test more comprehensive, add unseen values.
        target_encoder = TargetEncoder(
            aggregators=[np.mean, np.median],
            splitter=KFold(n_splits=5),
            drop_source_features=False
        )
        execution_result = target_encoder.fit_transform_out_of_fold(
            X,
            y,
            source_positions=[1]
        )
        true_answer = np.array(
            [[1, 2, 7, 6],
             [2, 0, 7, 7],
             [3, 0, 7, 7],
             [4, 0, 2.5, 2.5],
             [10, 0, 2.5, 2.5],
             [1, 1, 6.75, 5.5],
             [2, 1, 7.5, 7.5],
             [3, 1, 7.5, 7.5],
             [4, 1, 7.5, 7.5],
             [10, 1, 4.5, 4.5],
             [1, -1, 29 / 3, 8],
             [2, -1, 29 / 3, 8],
             [3, -1, 5.5, 5.5],
             [4, -1, 5.5, 5.5],
             [10, -1, 5.5, 5.5]],
            dtype=float
        )
        self.assertTrue(np.allclose(execution_result, true_answer))

    def test_fit_transform_out_of_fold_on_ordered_data(self) -> type(None):
        """
        Test `fit_transform_out_of_fold` method on data
        with ordered observations. Ordering implies that
        `TimeSeriesSplit` must be used.

        :return:
            None
        """
        X, y = get_dataset_for_features_creation()
        target_encoder = TargetEncoder(
            aggregators=[np.mean, np.median],
            splitter=TimeSeriesSplit(n_splits=5)
        )
        execution_result = target_encoder.fit_transform_out_of_fold(
            X,
            y,
            source_positions=[1]
        )
        true_answer = np.array(
            [[1, 4, 3],
             [2, 4, 3],
             [3, 3.5, 3.5],
             [4, 3.5, 3.5],
             [10, 4.5, 4.5],
             [1, 38 / 9, 4],
             [2, 5, 5],
             [3, 5, 5],
             [4, 6, 6],
             [10, 6, 6]],
            dtype=float
        )
        self.assertTrue(np.allclose(execution_result, true_answer))

    def test_correct_work_with_rare_values(self) -> type(None):
        """
        Test that `fit_transform_out_of_fold` does not produce
        `np.nan` or other missing placeholders if a value does not
        occur in folds other than a current one. It must fill it with
        unconditional aggregate instead of missing placeholder.

        :return:
            None
        """
        X = np.array(
            [[4, 1],
             [5, 2],
             [6, 3]],
            dtype=float
        )
        y = np.array([2, 4, 6], dtype=float)
        target_encoder = TargetEncoder(
            aggregators=[np.mean, np.median]
        )
        execution_result = target_encoder.fit_transform_out_of_fold(
            X,
            y,
            source_positions=[1]
        )
        true_answer = np.array(
            [[4, 5, 5],
             [5, 4, 4],
             [6, 3, 3]],
            dtype=float
        )
        self.assertTrue(np.allclose(execution_result, true_answer))


def get_dataset_for_regression() -> Tuple[np.ndarray, np.ndarray]:
    """
    Get dataset with one numerical feature, one label-encoded
    categorical feature, and a numerical target variable.

    :return:
        features array and target array
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


class TestOutOfFoldTargetEncodingRegressor(unittest.TestCase):
    """
    Tests of `OutOfFoldTargetEncodingRegressor` class.
    """

    def test_fit(self) -> type(None):
        """
        Test `fit` method.

        :return:
            None
        """
        X, y = get_dataset_for_regression()
        rgr = OutOfFoldTargetEncodingRegressor(
            LinearRegression(),
            estimator_kwargs=dict()
        )
        rgr.fit(X, y, source_positions=[1])
        learnt_slopes = rgr.estimator.coef_
        true_answer = np.array([1.8, 0.8])
        self.assertTrue(np.allclose(learnt_slopes, true_answer))

    def test_predict(self) -> type(None):
        """
        Test `predict` method.
        Note that `fit_predict` must produce different result.

        :return:
            None
        """
        X, y = get_dataset_for_regression()
        rgr = OutOfFoldTargetEncodingRegressor(
            LinearRegression(),
            estimator_kwargs=dict()
        )

        # Fit it manually.
        rgr.estimator.coef_ = np.array([1.8, 0.8])
        rgr.estimator.intercept_ = -2.8
        rgr.target_encoder_ = TargetEncoder()
        rgr.target_encoder_.mappings_ = {
            1: pd.DataFrame(
                [[0.0, 4.0],
                 [1.0, 2.0],
                 [np.nan, 3.0]],
                columns=['1', 'agg_0'],
                index=[0, 1, '__unseen__']
            )
        }

        result = rgr.predict(X)
        true_answer = np.array([5.8, 2.2, 4, 0.6, 2.4, 4.2])
        self.assertTrue(np.allclose(result, true_answer))

    def test_fit_predict(self) -> type(None):
        """
        Test `fit_predict` method.

        :return:
            None
        """
        X, y = get_dataset_for_regression()
        rgr = OutOfFoldTargetEncodingRegressor(
            LinearRegression(),
            estimator_kwargs=dict()
        )
        result = rgr.fit_predict(X, y, source_positions=[1])
        true_answer = np.array([5.8, 2.2, 4, 1, 1.6, 3.4])
        self.assertTrue(np.allclose(result, true_answer))


def get_dataset_for_classification() -> Tuple[np.ndarray, np.ndarray]:
    """
    Get dataset with one numerical feature, one label-encoded
    categorical feature, and a binary class label.

    :return:
        features array and target array
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


class TestOutOfFoldTargetEncodingClassifier(unittest.TestCase):
    """
    Tests of `OutOfFoldTargetEncodingClassifier` class.
    """

    def test_fit(self) -> type(None):
        """
        Test `fit` method.

        :return:
            None
        """
        X, y = get_dataset_for_classification()
        clf = OutOfFoldTargetEncodingClassifier(
            LogisticRegression(),
            estimator_kwargs={'random_state': 361}
        )
        clf.fit(X, y, source_positions=[1])
        learnt_slopes = clf.estimator.coef_
        true_answer = np.array([[0.48251806, -0.16291334]])
        self.assertTrue(np.allclose(learnt_slopes, true_answer))

    def test_predict_proba(self) -> type(None):
        """
        Test `predict_proba` method.
        Note that `fit_predict_proba` must produce different result.

        :return:
            None
        """
        X, y = get_dataset_for_classification()
        clf = OutOfFoldTargetEncodingClassifier(
            LogisticRegression(),
            estimator_kwargs={'random_state': 361}
        )

        # Fit it manually.
        clf.estimator.coef_ = np.array([[0.48251806, -0.16291334]])
        clf.estimator.intercept_ = [-0.51943239]
        clf.target_encoder_ = TargetEncoder()
        clf.target_encoder_.mappings_ = {
            1: pd.DataFrame(
                [[0.0, 2 / 3],
                 [1.0, 1 / 3],
                 [np.nan, 0.5]],
                columns=['1', 'agg_0'],
                index=[0, 1, '__unseen__']
            )
        }

        result = clf.predict_proba(X)[:, 1]
        true_answer = np.array([0.69413293, 0.46368326, 0.58346035,
                                0.47721111, 0.59659543, 0.70553938])
        self.assertTrue(np.allclose(result, true_answer))

    def test_fit_predict_proba(self) -> type(None):
        """
        Test `fit_predict_proba` method.

        :return:
            None
        """
        X, y = get_dataset_for_classification()
        clf = OutOfFoldTargetEncodingClassifier(
            LogisticRegression(),
            estimator_kwargs={'random_state': 361}
        )
        result = clf.fit_predict_proba(X, y, source_positions=[1])[:, 1]
        true_answer = np.array([0.68248347, 0.45020866, 0.59004395,
                                0.47044176, 0.60959347, 0.71669408])
        self.assertTrue(np.allclose(result, true_answer))


def main():
    test_loader = unittest.TestLoader()
    suites_list = []
    testers = [
        TestTargetEncoder(),
        TestOutOfFoldTargetEncodingRegressor(),
        TestOutOfFoldTargetEncodingClassifier()
    ]
    for tester in testers:
        suite = test_loader.loadTestsFromModule(tester)
        suites_list.append(suite)
    overall_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner().run(overall_suite)


if __name__ == '__main__':
    main()
