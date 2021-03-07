import os
import pickle
from typing import Optional

import numpy as np
from sklearn.pipeline import Pipeline

from mlops_examples.domain.model_wrappers.model_wrapper import ModelWrapper
from mlops_examples.domain.model_wrappers.sklearn_model_wrapper import SklearnModelWrapper
from mlops_examples.infrastructure.models.model_repository import ModelRepository


class PickleRepository(ModelRepository):

    def put(self, pipeline: Pipeline, model_name: str, data_subset: Optional[np.ndarray] = None):
        path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
                "", model_name
            )
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(pipeline, f)
        print(f"Saved to {path}")

    def get(self, model_name: str) -> ModelWrapper:
        path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
                "", model_name
            )
        )
        with open(path, 'rb') as f:
            pipeline = pickle.load(f)
        print(f"Loaded {path}")
        return SklearnModelWrapper().using(pipeline)
