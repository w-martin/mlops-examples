import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DatasetRepository:

    def __init__(self):
        self.__data = None
        self.__classes = None


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
        self.__classes = label_encoder.classes_
        self.__data = data

    def get(self):
        if self.__data is None:
            self.__load()
        return self.__data

    @property
    def classes(self):
        if self.__classes is None:
            self.__load()
        return self.__classes
