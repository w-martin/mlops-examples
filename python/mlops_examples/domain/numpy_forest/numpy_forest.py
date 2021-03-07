from typing import Union, Optional

import numpy as np

from mlops_examples.domain.numpy_forest.decision_node import DecisionNode


class NumpyForest(object):

    def __init__(self,
                 n_trees: int = 100,
                 sample_proportion: float = 0.75,
                 min_size: int = 2,
                 max_depth: Optional[int] = None
                 ):
        self._n_trees = n_trees
        self._sample_proportion = sample_proportion
        self._rng = np.random.default_rng()
        self._min_size = min_size
        self._max_depth = max_depth
        self._classes = None
        self._trees = None

    def fit(self, X, y):
        sample_size = np.ceil(X.shape[0] * self._sample_proportion).astype(int)
        training_set = np.append(X, y.reshape([-1, 1]), axis=1)
        self._classes = np.unique(y)
        self._trees = self._train_forest(training_set, sample_size)
        return self

    def _train_forest(self, training_set, sample_size):
        trees = [
            self._build_tree(training_set, sample_size)
            for _ in range(self._n_trees)
        ]
        return trees

    def _train_node(self, dataset, depth: int) -> Union[DecisionNode, int]:
        y = dataset[:, -1].astype(int)

        if len(dataset) < self._min_size \
                or (self._max_depth is not None and depth == self._max_depth) \
                or np.unique(y).shape[0] == 1:
            return np.argmax(np.bincount(y))

        n_features = np.ceil(np.sqrt(dataset.shape[1])).astype(int)
        result = DecisionNode()
        smallest_gini = np.finfo(np.float16).max
        features = self._rng.choice(dataset.shape[1] - 1, size=n_features, replace=False)
        for feature in features:
            for row in dataset:
                value = row[feature]
                left = dataset[:, feature] < value
                gini = self._gini_coefficient(y, left, ~left)
                if result is None or gini < smallest_gini:
                    smallest_gini = gini
                    result.value = value
                    result.feature = feature

        left_class_size = np.sum(left)
        if left_class_size == 0 or left_class_size == len(dataset):
            return dataset[0, -1]
        result.left = self._train_node(dataset[left], depth + 1)
        result.right = self._train_node(dataset[~left], depth + 1)
        return result

    def _gini_coefficient(self, y, l, r):
        classes = np.unique(y)
        scores = [ \
            np.sum(
                np.square(
                    np.mean(
                        np.equal(
                            classes[np.newaxis, :],
                            group[:, np.newaxis]), axis=0)))
            for group in (y[l], y[r])
        ]
        gini = np.sum((1 - np.array(scores)) * np.array([np.sum(l), np.sum(r)]) / y.shape[0])
        return gini

    def _build_tree(self, training_set, sample_size):
        sample = training_set[self._rng.choice(len(training_set), size=sample_size, replace=False)]
        root = self._train_node(sample, 0)
        return root

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def _decide(self, row, tree: Union[DecisionNode, int]) -> int:
        if isinstance(tree, DecisionNode):
            if row[tree.feature] < tree.value:
                return self._decide(row, tree.left)
            else:
                return self._decide(row, tree.right)
        else:
            return tree

    def _compute_proba(self, row):
        votes = [self._decide(row, tree) for tree in self._trees]
        return np.equal(self._classes.reshape(1, -1), np.array(votes).reshape(-1, 1)).sum(axis=0) / len(votes)

    def predict_proba(self, X):
        predictions = [self._compute_proba(row) for row in X]
        return predictions
