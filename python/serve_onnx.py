from fastapi import FastAPI

from mlops_examples.domain.input_dto import InputDto
from mlops_examples.infrastructure.dataset_repository import DatasetRepository
from mlops_examples.infrastructure.models.onnx_repository import OnnxRepository

app = FastAPI()
model = OnnxRepository().get("sklearn_converted_model.onnx")
label_encoder = DatasetRepository().label_encoder

MODEL = None


@app.post("/predict")
async def predict(row: InputDto):
    transformed = label_encoder.inverse_transform(model.predict(row.data))
    return {"result": transformed.tolist()}


@app.post("/predict_proba")
async def predict_proba(row: InputDto):
    return {"result": model.predict_proba(row.data)}
