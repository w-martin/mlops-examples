from typing import List

import numpy as np
from sklearn.pipeline import Pipeline

from mlops_examples.domain.model import Model


class SklearnModel(Model):

    def __init__(self):
        self.__pipeline: Pipeline = None

    def using(self, pipeline: Pipeline) -> Model:
        self.__pipeline = pipeline
        return self

    def predict(self, data: List[List[float]]) -> List[str]:
        return self.__pipeline.predict(np.array(data))

    def predict_proba(self, data: List[List[float]]) -> List[List[float]]:
        return self.__pipeline.predict_proba(np.array(data)).tolist()
