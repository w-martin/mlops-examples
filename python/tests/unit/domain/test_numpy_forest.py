from unittest import TestCase

import numpy as np

from mlops_examples.domain.numpy_forest.numpy_forest import NumpyForest


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
