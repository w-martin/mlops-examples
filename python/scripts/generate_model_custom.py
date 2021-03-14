from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from mlops_examples.domain.numpy_forest.numpy_forest import NumpyForest
from mlops_examples.infrastructure.models.onnx_repository import OnnxRepository
from mlops_examples.infrastructure.models.pickle_repository import PickleRepository
from mlops_examples.infrastructure.models.pmml_repository import PmmlRepository
from mlops_examples.infrastructure.dataset_repository import DatasetRepository


def main():
    dataset_repository = DatasetRepository()
    training_set = dataset_repository.get()
    train, test = train_test_split(training_set, train_size=0.1)
    pipeline = Pipeline([
        ("forest", NumpyForest())
    ])
    X = train.drop(columns=["y"]).values
    y = train["y"]
    pipeline.fit(X, y)

    X_test = test.drop(columns=["y"])
    y_test = test["y"]
    predicted = pipeline.predict(X_test)

    f1 = f1_score(y_test, predicted, average="macro")

    print(f"F1: {f1:.3f}")

    OnnxRepository().put(pipeline, "sklearn_converted_model.onnx", X)
    PmmlRepository().put(pipeline, "sklearn_converted_model.pmml")
    PickleRepository().put(pipeline, "sklearn_converted_model.pkl")


if __name__ == "__main__":
    main()
