import os

import pypmml
from sklearn.pipeline import Pipeline
from sklearn2pmml import make_pmml_pipeline, sklearn2pmml

from mlops_examples.domain.model import Model
from mlops_examples.domain.pmml_model import PmmlModel
from mlops_examples.infrastructure.model_repository import ModelRepository


class PmmlRepository(ModelRepository):

    def put(self, pipeline: Pipeline, model_name: str, _=None):
        pmml_pipeline = make_pmml_pipeline(pipeline)

        path = self.__get_path(model_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        sklearn2pmml(pmml_pipeline, path)
        print(f"Saved to {path}")

    def __get_path(self, model_name):
        path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
                "models", model_name
            )
        )
        return path

    def get(self, model_name: str) -> Model:
        path = self.__get_path(model_name)
        model = pypmml.Model.fromFile(path)
        return PmmlModel().using(model)
