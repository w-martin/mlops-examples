import os

from onnxruntime import InferenceSession
from skl2onnx import to_onnx

from mlops_examples.domain.model_wrappers.model_wrapper import ModelWrapper
from mlops_examples.domain.model_wrappers.onnx_model_wrapper import OnnxModelWrapper
from mlops_examples.infrastructure.models.model_repository import ModelRepository


class OnnxRepository(ModelRepository):

    def put(self, pipeline, model_name, training_sample=None):
        onnx_pipeline = to_onnx(pipeline, training_sample)
        serialised_model = onnx_pipeline.SerializeToString()

        InferenceSession(serialised_model)
        path = self.__get_path(model_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(serialised_model)
        print(f"Saved to {path}")

    def __get_path(self, model_name):
        path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
                "", model_name
            )
        )
        return path

    def get(self, model_name: str) -> ModelWrapper:
        path = self.__get_path(model_name)
        sess = InferenceSession(path)
        return OnnxModelWrapper().using(sess)
