# Phase 5 Thesis: Model Format Conversion Experiments (Abandoned)

## Strategic Intent
Phase 5 was a brief detour driven by the question: "Can we convert our Keras models to lighter, faster formats for deployment?" Two conversion paths were explored: H5 → ONNX (via `tf2onnx`) and TFLite → TensorFlow (reverse conversion). Both scripts are under 30 lines combined and represent incomplete exploratory work.

## Scope & Boundaries
`6.py` (10 lines) loads a Keras `.h5` model, saves it as a TensorFlow `SavedModel` directory, then provides a commented-out CLI command for ONNX conversion. `7.py` (26 lines) attempts to wrap a TFLite interpreter's `invoke()` method inside a `@tf.function` decorator to create a concrete function — an architecturally invalid approach, since TFLite interpreters are C++ runtime objects that cannot be traced by TensorFlow's graph compiler.

## Failure Analysis
Both scripts were abandoned without producing usable output. `6.py`'s ONNX path was never pursued because ONNX runtime offered no compelling advantage over TensorFlow for this project's requirements (CPU-only inference, Python backend). `7.py`'s reverse TFLite-to-TF conversion is logically impossible as written — `interpreter.invoke()` returns `None` and has side effects through tensor buffers, which cannot be captured by `tf.function` tracing. The correct forward conversion path (Keras → TFLite) was not achieved until Phase 9, when `convert_audio_model.py` properly used `tf.lite.TFLiteConverter.from_keras_model()` with `SELECT_TF_OPS` to handle BiLSTM dynamic loops.
