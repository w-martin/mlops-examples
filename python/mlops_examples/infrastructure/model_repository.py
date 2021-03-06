import abc
from typing import Optional

import numpy as np
from sklearn.pipeline import Pipeline

from mlops_examples.domain.model import Model


class ModelRepository(abc.ABC):

    @abc.abstractmethod
    def put(self, pipeline: Pipeline, model_name: str, data_subset: Optional[np.ndarray] = None):
        pass

    @abc.abstractmethod
    def get(self, model_name: str) -> Model:
        pass
