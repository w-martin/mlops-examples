from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from mlops_examples.domain.transformers.onnx_transformer import OnnxTransformer
from mlops_examples.infrastructure.dataset_repository import DatasetRepository
from mlops_examples.infrastructure.model_repository import ModelRepository


def main():
    dataset_repository = DatasetRepository()
    training_set = dataset_repository.get()
    train, test = train_test_split(training_set, train_size=0.1)
    pipeline = Pipeline([
        ("forest", RandomForestClassifier())
    ])
    X = train.drop(columns=["y"]).values
    y = train["y"]
    pipeline.fit(X, y)

    X_test = test.drop(columns=["y"])
    y_test = test["y"]
    predicted = pipeline.predict(X_test)

    f1 = f1_score(y_test, predicted, average="macro")

    print(f"F1: {f1:.3f}")

    onnx_model = OnnxTransformer().transform(pipeline, X)
    ModelRepository().save(onnx_model, "sklearn_converted_model.onnx")


if __name__ == "__main__":
    main()
