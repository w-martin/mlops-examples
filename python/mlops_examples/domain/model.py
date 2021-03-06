import abc
from typing import Any, List


class Model(abc.ABC):

    @abc.abstractmethod
    def predict(self, row: List[List[float]]) -> List[str]:
        pass

    @abc.abstractmethod
    def predict_proba(self, row: List[List[float]]) -> List[List[float]]:
        pass

    @abc.abstractmethod
    def using(self, loaded_model: Any) -> 'Model':
        pass
