import numpy as np
from typing import List, Any

from mlops_examples.domain.model import Model


class PmmlModel(Model):

    def __init__(self):
        self.__model = None

    def predict(self, row: List[List[float]]) -> List[str]:
        result_proba = self.__model.predict(row)
        result = np.array(result_proba).argmax(axis=1)
        return result

    def predict_proba(self, row: List[List[float]]) -> List[List[float]]:
        result = self.__model.predict(row)
        result = list(map(list, result))
        return result

    def using(self, loaded_model: Any) -> 'Model':
        self.__model = loaded_model
        return self
