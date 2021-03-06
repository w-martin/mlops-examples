from skl2onnx import to_onnx


class OnnxTransformer:

    def transform(self, pipeline, training_sample):
        onnx_pipeline = to_onnx(pipeline, training_sample)
        return onnx_pipeline.SerializeToString()
