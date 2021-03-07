from typing import Union

import numpy as np

from mlops_examples.domain.numpy_forest.decision_node import DecisionNode


class NumpyForest(object):

    def __init__(self,
                 n_trees: int = 2,
                 sample_proportion: float = 0.75,
                 min_size: int = 1
                 ):
        self._n_trees = n_trees
        self._sample_proportion = sample_proportion
        self._rng = np.random.default_rng()
        self._min_size = min_size
        self._classes = None
        self._trees = None

    def fit(self, X, y):
        sample_size = np.ceil(X.shape[0] * self._sample_proportion).astype(int)
        training_set = np.append(X, y.reshape([-1, 1]), axis=1)
        self._classes = np.unique(y)
        self._trees = self._train_forest(training_set, sample_size)

    def _train_forest(self, training_set, sample_size):
        trees = [
            self._build_tree(training_set, sample_size)
            for _ in range(self._n_trees)
        ]
        return trees

    def _train_node(self, dataset) -> Union[DecisionNode, int]:
        if len(dataset) < self._min_size:
            return dataset[0, -1]

        n_features = np.ceil(np.sqrt(dataset.shape[1])).astype(int)
        class_values = dataset[:, -1]
        result = DecisionNode()
        smallest_gini = 1.1
        features = self._rng.choice(dataset.shape[1] - 1, size=n_features, replace=True)
        for feature in features:
            for row in dataset:
                value = row[feature]
                left = dataset[:, feature] < value
                gini = self.gini_coefficient(left, class_values)
                if result is None or gini < smallest_gini:
                    smallest_gini = gini
                    result.value = value
                    result.feature = feature

        left_class_size = np.sum(left)
        if left_class_size == 0 or left_class_size == len(dataset):
            return dataset[0, -1]
        result.left = self._train_node(dataset[left])
        result.right = self._train_node(dataset[~left])
        return result

    def gini_coefficient(self, left, classes):
        n = np.sum(left)
        if n == 0:
            return 1.
        y = np.sort(classes[left])
        if np.sum(y) == 0:
            return 1.
        i = np.arange(n) + 1
        G = (2 * np.sum(i * y) / (n * np.sum(y))) - ((n + 1) / n)
        return G

    def _build_tree(self, training_set, sample_size):
        sample = training_set[self._rng.choice(len(training_set), size=sample_size, replace=False)]
        root = self._train_node(sample)
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
        return np.equal(self._classes.reshape(1, -1), np.array(votes).reshape(-1, 1)).sum(axis=0)/len(votes)

    def predict_proba(self, X):
        predictions = [self._compute_proba(row) for row in X]
        return predictions
