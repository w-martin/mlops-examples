from abc import ABC, abstractmethod

import numpy as np


class DataGenerator(ABC):

    @abstractmethod
    def get(self, n_rows: int) -> np.ndarray:
        pass