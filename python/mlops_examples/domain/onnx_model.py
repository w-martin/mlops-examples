from typing import List, Any

import numpy as np
import pandas as pd

from mlops_examples.domain.model import Model


class OnnxModel(Model):

    def __init__(self):
        self.__session = None

    def predict(self, row: List[List[float]]) -> List[str]:
        input_name = self.__session.get_inputs()[0].name
        result = self.__session.run(None, {input_name: np.array(row)})
        return result[0].tolist()

    def predict_proba(self, row: List[List[float]]) -> List[float]:
        input_name = self.__session.get_inputs()[0].name
        result = self.__session.run(None, {input_name: np.array(row)})
        return pd.DataFrame(result[1]).values.tolist()

    def using(self, loaded_model: Any) -> 'Model':
        self.__session = loaded_model
        return self
