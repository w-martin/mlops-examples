from unittest import TestCase

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from mlops_examples.domain.numpy_forest.numpy_forest import NumpyForest
from mlops_examples.infrastructure.dataset_repository import DatasetRepository


class TestNumpyForest(TestCase):

    def setUp(self) -> None:
        # self.sut = RandomForestClassifier()
        self.sut = NumpyForest()

    def test_should_train_forest(self):
        # arrange
        X = np.array([
            [0, 1, 2],
            [5, 6, 7],
            [10, 11, 12],
            [15, 16, 17]
        ])
        y = np.array([
            0, 0, 1, 1
        ])
        X_test = np.array([
            [0, 1, 2],
            [25, 26, 27]
        ])
        y_test = np.array([
            0, 1
        ])
        # act
        self.sut.fit(X, y)
        result = self.sut.predict(X_test)
        # assert
        np.testing.assert_almost_equal(y_test, result)

    def test_should_train_forest_pon_iris(self):
        # arrange
        data = DatasetRepository().get()
        data["y"] = (data["y"] > 0).astype(int)
        train, test = train_test_split(data, train_size=0.5)
        X, y = train.drop(columns=["y"]).values, train["y"].values
        X_test, y_test = test.drop(columns=["y"]).values, test["y"].values
        # act
        self.sut = NumpyForest(max_depth=2).fit(X, y)
        result = self.sut.predict(X_test)
        sklearn_clf = RandomForestClassifier(max_depth=2, bootstrap=False)
        sklearn_clf.fit(X, y)
        sklearn_result = sklearn_clf.predict(X_test)
        # assert
        np.testing.assert_almost_equal(y_test, sklearn_result)
        np.testing.assert_almost_equal(y_test, result)
