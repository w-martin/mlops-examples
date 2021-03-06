import os


class ModelRepository:
    def save(self, serialised_model, filename):
        path = os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
            "models", filename
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(serialised_model)
        print(f"Saved to {path}")
