import numpy as np
import pandas as pd

from mlops_examples.domain.data_generation.data_generator import DataGenerator


class MandelbrotGenerator(DataGenerator):
    R_MIN = -2
    R_MAX = 0.47
    I_MIN = -1.12
    I_MAX = 1.12

    def __init__(self):
        self.__max: int = 25
        self._threshold: int = 4

    def with_max(self, m: int) -> 'MandelbrotGenerator':
        self.__max = m
        return self

    def get(self, n_rows: int) -> pd.DataFrame:
        data = np.empty(n_rows, dtype=np.complex64)
        root_n = np.ceil(np.sqrt(n_rows)).astype(int)
        data[:].real = np.repeat(np.linspace(self.R_MIN, self.R_MAX, root_n), root_n)[:n_rows]
        data[:].imag = np.tile(np.linspace(self.I_MIN, self.I_MAX, root_n), root_n)[:n_rows]
        result = self.__compute_mandelbrot(data)
        df = pd.DataFrame(
            columns=["r", "i", "target"],
            data=np.stack((data.real, data.imag, result), axis=1)
        )
        return df

    def __compute_mandelbrot(self, c: np.ndarray) -> np.ndarray:
        result = np.zeros_like(c, dtype=int)
        z = np.copy(c)
        i = 1
        while i < self.__max:
            unset = result == 0
            if unset.sum() < 1:
                break
            z = np.square(z) + c
            mask = unset & (z.real * z.imag > self._threshold)
            result[mask] = i
            i += 1
        unset = result == 0
        result[unset] = self.__max
        return result
