"""
This module contains tests of code from `../dsawl/stacking`
directory. Code that is tested here provides functionality for
stacking machine learning models on top of other machine learning
models.

@author: Nikolay Lysenko
"""


import unittest
from typing import Tuple
from copy import copy

import numpy as np

from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from dsawl.stacking.stackers import (
    BaseStacking, StackingRegressor, StackingClassifier
)


class TestBaseStacking(unittest.TestCase):
    """
    Tests of `BaseStacking` class.
    """

    def test_that_this_class_is_abstract(self) -> type(None):
        """
        Test that `BaseStacking` class can not have any instances.

        :return:
            None
        """
        message = ''
        try:
            BaseStacking()
        except TypeError as e:
            message = str(e)
        self.assertTrue('abstract class' in message)


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

    def test_fit_with_pipelines_as_base_estimators(self) -> type(None):
        """
        Test `fit` with objects of class `sklearn.pipeline.Pipeline`
        as first stage estimators.

        :return:
            None
        """
        X, y = get_dataset_for_regression()
        rgr = StackingRegressor(
            base_estimators_types=[Pipeline, Pipeline],
            base_estimators_params=[
                {
                    'steps': [('lin_reg', LinearRegression())]
                },
                {
                    'steps': [('neighbors', KNeighborsRegressor())],
                    'neighbors__n_neighbors': 1
                }
            ],
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
                rgr.base_estimators_[0].named_steps.lin_reg.coef_,
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

    def test_fit_with_pipeline_as_meta_estimator(self) -> type(None):
        """
        Test `fit` with object of class `sklearn.pipeline.Pipeline`
        as second stage estimator.

        :return:
            None
        """
        X, y = get_dataset_for_regression()
        rgr = StackingRegressor(
            base_estimators_types=[LinearRegression, KNeighborsRegressor],
            base_estimators_params=[dict(), {'n_neighbors': 1}],
            meta_estimator_type=Pipeline,
            meta_estimator_params={
                'steps': [('lin_reg', LinearRegression())]
            },
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
                rgr.meta_estimator_.named_steps.lin_reg.coef_,
                true_coefs_of_meta_estimator
            )
        )

    def test_fit_with_random_state(self) -> type(None):
        """
        Test that several calls of `fit` produce the same results if
        random state is set.

        :return:
            None
        """
        X, y = get_dataset_for_regression()
        rgr = StackingRegressor(
            base_estimators_types=[RandomForestRegressor, KNeighborsRegressor],
            base_estimators_params=[{'n_estimators': 3}, {'n_neighbors': 1}],
            splitter=KFold(shuffle=True),
            random_state=361
        )
        rgr.fit(X, y)
        first_meta_X = copy(rgr.meta_X_)
        first_importances = rgr.base_estimators_[0].feature_importances_
        rgr.fit(X, y)
        second_meta_X = copy(rgr.meta_X_)
        second_importances = rgr.base_estimators_[0].feature_importances_
        self.assertTrue(np.array_equal(first_meta_X, second_meta_X))
        self.assertTrue(np.array_equal(first_importances, second_importances))

    def test_fit_with_sample_weights(self) -> type(None):
        """
        Test `fit` with shuffling and second stage estimator that must
        be trained only on some of the objects.
        Also test that random seed works if it is set inside
        `splitter`, not for a whole instance.

        :return:
            None
        """
        X, y = get_dataset_for_regression()
        rgr = StackingRegressor(
            base_estimators_types=[KNeighborsRegressor, KNeighborsRegressor],
            base_estimators_params=[{'n_neighbors': 1}, {'n_neighbors': 2}],
            splitter=KFold(shuffle=True, random_state=361)
        )
        sample_weight = np.array(
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=float
        )
        rgr.fit(X, y, meta_fit_kwargs={'sample_weight': sample_weight})
        true_meta_X = np.array(
            [[15, 15],
             [15, 15],
             [11, 15],
             [15, 19],
             [19, 21],
             [13, 13.5],
             [13, 10.5],
             [22, 17.5],
             [16, 15],
             [21, 16],
             [15, 19],
             [21, 25],
             [23, 27],
             [29, 26],
             [8, 11]]
        )
        self.assertTrue(np.array_equal(rgr.meta_X_, true_meta_X))
        true_coefs_of_meta_estimator = np.array([0.11326539, 0.90735827])
        self.assertTrue(
            np.allclose(
                rgr.meta_estimator_.coef_,
                true_coefs_of_meta_estimator
            )
        )
        true_intercept_of_meta_estimator = 0.755864160903
        self.assertAlmostEqual(
            rgr.meta_estimator_.intercept_, true_intercept_of_meta_estimator
        )

    def test_fit_without_saving(self) -> type(None):
        """
        Test that `fit` runs if `keep_meta_X` is `False`.

        :return:
            None
        """
        X, y = get_dataset_for_regression()
        rgr = StackingRegressor(keep_meta_X=False)
        rgr.fit(X, y)
        self.assertFalse(hasattr(rgr, 'meta_X_'))
        self.assertTrue(hasattr(rgr, 'meta_estimator_'))

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

        # Fit `rgr` manually.
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

    def test_drop_training_meta_features(self) -> type(None):
        """
        Test that sample used for training second stage estimator
        can be deleted.

        :return:
            None
        """
        X, y = get_dataset_for_regression()
        rgr = StackingRegressor(keep_meta_X=True)
        rgr.fit(X, y)
        rgr.drop_training_meta_features()
        self.assertTrue(rgr.meta_X_ is None)


def get_dataset_for_classification() -> Tuple[np.ndarray, np.ndarray]:
    """
    Get dataset with categorical target variable and two
    numerical features.

    :return:
        features array and target array
    """
    dataset = np.array(
        [[1, 2, 0],
         [2, 3, 0],
         [3, 4, 0],
         [4, 5, 0],
         [5, 6, 0],
         [6, 1, 1],
         [7, 2, 1],
         [8, 3, 1],
         [9, 4, 1],
         [1, 5, 0],
         [2, 6, 0],
         [3, 7, 0],
         [4, 8, 0],
         [5, 9, 0],
         [6, 2, 1]],
        dtype=float
    )
    X = dataset[:, :-1]
    y = dataset[:, -1]
    return X, y


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

    def test_fit(self) -> type(None):
        """
        Test `fit` method.

        :return:
            None
        """
        X, y = get_dataset_for_classification()
        clf = StackingClassifier(
            base_estimators_types=[LogisticRegression, KNeighborsClassifier],
            base_estimators_params=[dict(), {'n_neighbors': 1}],
            random_state=361
        )
        clf.fit(X, y)
        # Second column has probabilities of 0 class.
        true_meta_X_ = np.array(
            [[0.69944477, 1.0],
             [0.71150077, 1.0],
             [0.72326450, 1.0],
             [0.73472740, 1.0],
             [0.74588232, 1.0],
             [0.11467189, 0.0],
             [0.1687213, 0.0],
             [0.24130277, 0.0],
             [0.33261446, 0.0],
             [0.98984866, 1.0],
             [0.99417896, 1.0],
             [0.99572679, 1.0],
             [0.99686435, 1.0],
             [0.99769978, 1.0],
             [0.1056699, 0.0]]
        )
        self.assertTrue(np.allclose(clf.meta_X_, true_meta_X_))
        true_coefs_of_base_lr = np.array([[0.80295565, -1.11280117]])
        self.assertTrue(
            np.allclose(
                clf.base_estimators_[0].coef_,
                true_coefs_of_base_lr
            )
        )
        true_coefs_of_meta_estimator = np.array([[-0.87613043, -1.5124705]])
        self.assertTrue(
            np.allclose(
                clf.meta_estimator_.coef_,
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
        X, y = get_dataset_for_classification()
        clf = StackingClassifier()
        clf.fit(X, y)

    def test_fit_with_multiple_classes(self) -> type(None):
        """
        Test `fit` in multi-class classification problem.

        :return:
            None
        """
        dataset = np.array(
            [[3, 4, 2],
             [2, 1, 0],
             [5, 7, 1],
             [2, 4, 0],
             [9, 1, 1],
             [3, 3, 0],
             [4, 2, 2],
             [1, 1, 1],
             [3, 2, 2]]
        )
        X = dataset[:, :-1]
        y = dataset[:, -1]
        clf = StackingClassifier(
            base_estimators_types=[KNeighborsClassifier, KNeighborsClassifier],
            base_estimators_params=[{'n_neighbors': 1}, {'n_neighbors': 2}],
            random_state=361
        )
        clf.fit(X, y)
        true_meta_X = np.array(
            [[1.0, 0.0, 1.0, 0.0],
             [0.0, 1.0, 0.0, 0.5],
             [1.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0],
             [1.0, 0.0, 0.5, 0.0],
             [1.0, 0.0, 1.0, 0.0],
             [1.0, 0.0, 1.0, 0.0]]
        )
        self.assertTrue(np.allclose(clf.meta_X_, true_meta_X))

    def test_fit_without_predict_proba(self) -> type(None):
        """
        Test that `fit` method works when some of base estimators
        have only `predict` method, but have not `predict_proba`
        method.

        :return:
            None
        """
        X, y = get_dataset_for_classification()
        clf = StackingClassifier(
            base_estimators_types=[SVC, KNeighborsClassifier],
            base_estimators_params=[dict(), {'n_neighbors': 1}]
        )
        clf.fit(X, y)
        # Again, second column has probabilities of 0 class.
        true_meta_X_ = np.array(
            [[0, 1],
             [0, 1],
             [0, 1],
             [0, 1],
             [0, 1],
             [0, 0],
             [0, 0],
             [0, 0],
             [0, 0],
             [0, 1],
             [0, 1],
             [0, 1],
             [0, 1],
             [0, 1],
             [1, 0]],
            dtype=float
        )
        self.assertTrue(np.array_equal(clf.meta_X_, true_meta_X_))

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
        clf = StackingClassifier(
            base_estimators_types=[LogisticRegression, KNeighborsClassifier],
            base_estimators_params=[dict(), {'n_neighbors': 1}],
            random_state=361
        )

        # Fit `clf` manually.
        X, y = get_dataset_for_classification()
        lr = LogisticRegression(random_state=361).fit(X, y)
        kn = KNeighborsRegressor(n_neighbors=1).fit(X, y)
        clf.base_estimators_ = [lr, kn]
        meta_estimator = LogisticRegression(random_state=361)
        meta_estimator.coef_ = np.array([[-0.87613043, -1.5124705]])
        meta_estimator.intercept_ = 0.535153999535
        meta_estimator.classes_ = np.array([0, 1])
        clf.meta_estimator_ = meta_estimator
        clf.classes_ = np.array([0, 1])

        result = clf.predict(X_test)
        true_answer = np.array([0, 1, 0, 1])
        self.assertTrue(np.allclose(result, true_answer))

    def test_predict_proba(self) -> type(None):
        """
        Test `predict_proba` method.

        :return:
            None
        """
        X_test = np.array(
            [[1, 6],
             [-2, -4],
             [7, 10],
             [3, 1]],
            dtype=float
        )
        clf = StackingClassifier(
            base_estimators_types=[LogisticRegression, KNeighborsClassifier],
            base_estimators_params=[dict(), {'n_neighbors': 1}],
            random_state=361
        )

        # Fit `clf` manually.
        X, y = get_dataset_for_classification()
        lr = LogisticRegression(random_state=361).fit(X, y)
        kn = KNeighborsRegressor(n_neighbors=1).fit(X, y)
        clf.base_estimators_ = [lr, kn]
        meta_estimator = LogisticRegression(random_state=361)
        meta_estimator.coef_ = np.array([[-0.87613043, -1.5124705]])
        meta_estimator.intercept_ = 0.535153999535
        meta_estimator.classes_ = np.array([0, 1])
        clf.meta_estimator_ = meta_estimator
        clf.classes_ = np.array([0, 1])

        result = clf.predict_proba(X_test)
        true_answer = np.array(
            [[0.58395508, 0.41604492],
             [0.38338127, 0.61661873],
             [0.58374647, 0.41625353],
             [0.42309387, 0.57690613]]
        )
        self.assertTrue(np.allclose(result, true_answer))

    def test_fit_predict_proba_with_false_in_keep_meta_X(self) -> type(None):
        """
        Test that `fit_predict_proba` does not modifies attributes
        that must not be altered by it.

        :return:
            None
        """
        X, y = get_dataset_for_classification()
        clf = StackingClassifier(keep_meta_X=False)
        _ = clf.fit_predict_proba(X, y)
        self.assertFalse(clf.keep_meta_X)
        self.assertTrue(clf.meta_X_ is None)


def main():
    test_loader = unittest.TestLoader()
    suites_list = []
    testers = [
        TestBaseStacking(),
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
