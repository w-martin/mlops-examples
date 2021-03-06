import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DatasetRepository:

    def __init__(self):
        self.__data: pd.DataFrame = None
        self.__label_encoder: LabelEncoder = None

    def __load(self):
        path = os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
            "data", "iris.data"
        )
        data = pd.read_csv(path, header=None, index_col=None)
        columns = data.columns.tolist()
        columns[-1] = "y"
        data.columns = columns

        label_encoder = LabelEncoder()
        data.loc[:, "y"] = label_encoder.fit_transform(data["y"])
        self.__label_encoder = label_encoder
        self.__data = data

    def get(self):
        if self.__data is None:
            self.__load()
        return self.__data

    @property
    def classes(self):
        if self.__label_encoder is None:
            self.__load()
        return self.__label_encoder.classes_

    @property
    def label_encoder(self) -> LabelEncoder:
        if self.__label_encoder is None:
            self.__load()
        return self.__label_encoder
