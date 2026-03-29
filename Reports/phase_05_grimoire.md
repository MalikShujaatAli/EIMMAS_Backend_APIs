# Phase 5 Grimoire: Complete Technical Archaeology

---

## Header 1: Complete File Inventory

| File | Size (bytes) | Role |
|---|---|---|
| `FYP old/6.py` | 247 | H5 → SavedModel → ONNX conversion attempt |
| `FYP old/7.py` | 782 | TFLite → TF concrete function attempt |

---

## Header 2: Line-by-Line Logic Migration

### File: `6.py` (Complete Source)

```python

import tensorflow as tf

# Load your saved model
model = tf.keras.models.load_model("my_model.h5")
model.save("saved_model")

# Convert SavedModel to ONNX via CLI
# python -m tf2onnx.convert --saved-model saved_model --output model.onnx
```

#### Analysis
- **Line 4**: Loads a generic `my_model.h5` — the filename is a placeholder, not referencing any specific project model. This suggests the script was written while following a tutorial.
- **Line 5**: `model.save("saved_model")` saves the model in TensorFlow's `SavedModel` directory format (a directory containing `saved_model.pb` and `variables/`). This is a prerequisite for ONNX conversion.
- **Lines 8-9**: The actual conversion command is **commented out**. It was never executed from this script. The `tf2onnx` package would need to be installed separately.
- **No error handling, no path checking, no output verification.**
- **Phase 9 resolution**: ONNX was never pursued. The successful lightweight format conversion uses TFLite (`convert_audio_model.py`), not ONNX.

### File: `7.py` (Complete Source)

```python
import tensorflow as tf

# Load the TFLite model
tflite_model_file = "emotion_model.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Convert TFLite model to TensorFlow concrete function
def representative_dataset():
    for _ in range(100):
        # Provide dummy input of correct shape, float32 [1,48,48,1]
        yield [tf.random.uniform([1,48,48,1], dtype=tf.float32)]

# Create concrete function
@tf.function
def model_func(x):
    return interpreter.invoke()

# Save as SavedModel (this step is tricky; if the model is simple, you can use tflite2onnx directly)


```

#### Block 1: TFLite Interpreter Loading (Lines 3-10)
- **`emotion_model.tflite`**: A TFLite version of the Phase 1 face CNN (48×48 input). The existence of this file means a successful Keras→TFLite conversion happened at some point (possibly manually, possibly via a notebook), but the converter script was never preserved.
- **`interpreter.allocate_tensors()`**: Allocates memory for the TFLite interpreter's input/output buffers. Standard TFLite initialization.

#### Block 2: Representative Dataset (Lines 13-16)
- **Purpose**: `representative_dataset()` is a generator that provides calibration data for TFLite quantization. However, this function is **never called** anywhere in the script. It appears to have been copied from a quantization tutorial but was not integrated into any conversion pipeline.
- **Input shape `[1,48,48,1]`**: Confirms the TFLite model was a Phase 1 CNN (48×48 grayscale).

#### Block 3: The Impossible `@tf.function` (Lines 19-21)
- **`@tf.function def model_func(x): return interpreter.invoke()`**: This is architecturally invalid.
  - `tf.function` traces Python functions to build a TensorFlow computation graph. It requires that all operations inside the function are TensorFlow ops.
  - `interpreter.invoke()` is a **Python method** on a C++ TFLite interpreter object. It is NOT a TensorFlow op. It cannot be traced by `tf.function`.
  - Furthermore, `interpreter.invoke()` returns `None` — its results are accessed through `interpreter.get_tensor(output_details[0]['index'])`, which is also not a TF op.
  - The input `x` is never used — `interpreter.invoke()` operates on pre-set input tensors, not function arguments.
- **Lines 24-26**: Blank lines. The script ends with no output, no save, no further action. This is clear evidence of an abandoned experiment.

---

## Header 3: Micro-Decision Log

| Decision | Phase 5 Attempt | Phase 9 Resolution |
|---|---|---|
| Target format: ONNX | `tf2onnx.convert` (commented out) | Abandoned; TFLite chosen instead |
| Target format: TFLite→TF reverse | Impossible `@tf.function` wrapping | Forward conversion: `tf.lite.TFLiteConverter.from_keras_model()` |
| Quantization | `representative_dataset()` defined but unused | `tf.lite.Optimize.DEFAULT` + `float16` in `convert_audio_model.py` |
| BiLSTM compatibility | Not addressed | `SELECT_TF_OPS` + `_experimental_lower_tensor_list_ops = False` |

---

## Header 4: Inter-Phase Diff Analysis

| Phase 5 Artifact | Descendant |
|---|---|
| `6.py` | No descendant. ONNX path fully abandoned. |
| `7.py` `representative_dataset()` | Conceptual descendant: `convert_audio_model.py`'s quantization settings |
| `7.py` `@tf.function` wrapping | Correct usage appears in Phase 9: `@tf.function(reduce_retracing=True)` wrapping of `emotion_model(tensor, training=False)` (NOT the interpreter) |
| `emotion_model.tflite` (48×48 face) | Replaced by `audio_model.tflite` (audio BiLSTM) — face model stays in `.keras` format and uses `@tf.function` instead of TFLite |

**Key insight**: Phase 5 attempted the wrong conversions in the wrong directions. The eventual Phase 9 solution was:
- **Audio model**: Keras `.keras` → TFLite `.tflite` (via `convert_audio_model.py`) for CPU speed
- **Vision model**: Keras `.keras` → `@tf.function` compiled graph (no format conversion; compilation happens at runtime)
- **Text model**: Keras `.keras` → `@tf.function` compiled graph (same as vision)
