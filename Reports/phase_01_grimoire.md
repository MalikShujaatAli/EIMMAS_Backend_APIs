# Phase 1 Grimoire: Complete Technical Archaeology

---

## Header 1: Complete File Inventory

| File | Size (bytes) | Role |
|---|---|---|
| `FYP old/phase01_vision_cnn_trainer.py` | 1,490 | CNN face emotion model trainer |
| `FYP old/phase01_diagnostic_gpu_check.py` | 145 | GPU availability diagnostic |
| `FYP old/phase01_vision_cnn_notebook.ipynb` | 222,040 | Training notebook (Jupyter) |
| `FYP old/cnn model/face_emotion_model.h5` | 676,008 | Trained CNN weights output |
| `FYP old/phase01_vision_classification_report.txt` | 609 | Evaluation metrics |
| `FYP old/phase01_vision_accuracy_plot.png` | 36,263 | Training accuracy curve |
| `FYP old/phase01_vision_confusion_matrix.png` | 46,880 | Confusion matrix heatmap |
| `FYP old/cnn model/loss_plot.png` | 37,165 | Training loss curve |
| `FYP old/phase01_vision_classification_report_root.txt` | 609 | Root-level copy of evaluation metrics |

---

## Header 2: Line-by-Line Logic Migration

### File: `phase01_vision_cnn_trainer.py` (Complete Source)

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths (change these to your dataset location)
train_dir = "archive_5/train"
val_dir   = "archive_5/test"

# Preprocess with augmentation
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=64,
    class_mode='categorical'
)

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=64,
    class_mode='categorical'
)

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotions
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator
)

# Save trained model
model.save("emotion_model.h5")
print("✅ Model saved as emotion_model.h5")
```

#### Block 1: Imports (Lines 1-6)
- **What this solves**: Establishing the TensorFlow/Keras computational environment.
- **What it replaces**: Nothing — this is the genesis script.
- **Logical flaws**: `pandas` is imported but never used. It was likely imported out of habit from data analysis workflows. This dead import persists in no future phases.

#### Block 2: Data Paths (Lines 8-9)
- **What this solves**: Pointing to the local FER2013 dataset.
- **Artifact**: `archive_5/` is the FER2013 original dataset (before FERPlus correction).
- **Critical decision**: Using FER2013 instead of FERPlus. The FER2013 labels are crowd-sourced and contain significant noise — many images have debatable "correct" labels. This directly contributed to the 57% ceiling. The switch to FERPlus (Microsoft-corrected labels) did not happen until the Kaggle retraining in `training_models_notebook_dump.txt`.

#### Block 3: Data Generator (Lines 12-29)
- **What this solves**: Loading images from directory structure and normalizing pixel values to [0,1].
- **Logical flaw — No augmentation**: Despite the variable being named `datagen`, only `rescale=1./255` is applied. No rotation, zoom, flip, or shift augmentation is configured. The Phase 9 Kaggle notebook adds `rotation_range=10`, `zoom_range=0.1`, `width_shift_range=0.1`, `height_shift_range=0.1`, `horizontal_flip=True`.
- **Logical flaw — No class weighting**: The FER2013 dataset is heavily imbalanced (Happy: 8,989 images vs Disgust: 547 images). Without `class_weight` in `model.fit()`, the optimizer learns to ignore minority classes. This is the direct cause of Disgust's 3% recall.
- **Logical flaw — 48×48 resolution**: FER2013 images are natively 48×48. This resolution is extremely low for capturing subtle facial muscle movements. The Phase 9 model uses 112×112.
- **Logical flaw — No CLAHE**: Images are used as-is. Dark or poorly-lit faces have crushed histograms, losing contrast in critical facial features. CLAHE was first introduced in Phase 7 (`phase07_vision_api_standalone.txt`).

#### Block 4: CNN Architecture (Lines 32-43)
- **What this solves**: A minimal 3-layer CNN for spatial feature extraction.
- **Architecture**: `Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool → Flatten → Dense(128) → Dropout(0.5) → Dense(7, softmax)`.
- **Logical flaw — Too shallow**: 3 convolutional layers cannot capture the hierarchical complexity of facial expressions (edges → eyebrow curves → full facial geometry). The Phase 9 model uses 4 convolutional blocks with BatchNormalization after each, and each block has progressively larger filter counts (64 → 128 → 256 → 512).
- **Logical flaw — No BatchNormalization**: Without BatchNorm, internal covariate shift slows training and makes the model sensitive to learning rate. Every convolutional block in the Phase 9 model includes `BatchNormalization()`.
- **Logical flaw — Single Dropout layer**: Only one `Dropout(0.5)` before the output. The Phase 9 model applies `Dropout(0.25)` after every convolutional block AND `Dropout(0.5)` before the final Dense layer.

#### Block 5: Training (Lines 45-52)
- **What this solves**: Running the training loop.
- **Configuration**: `epochs=30`, `batch_size=64` (from the generator), `optimizer='adam'` with default learning rate (0.001).
- **Logical flaw — No callbacks**: No `ModelCheckpoint` (best model is not saved), no `EarlyStopping` (training runs all 30 epochs even if overfitting), no `ReduceLROnPlateau` (learning rate never adapts). The Phase 9 notebooks use all three callbacks.
- **Logical flaw — No class_weight**: `model.fit()` is called without the `class_weight` parameter. The model optimizes for overall accuracy, which it achieves by learning to always predict "Happy" (the majority class).

#### Block 6: Save (Lines 55-56)
- **Output artifact**: `emotion_model.h5` — a Keras HDF5 model file.
- **Format decision**: HDF5 (`.h5`) was the standard Keras format. Later phases switch to `.keras` (the native Keras 3 format) for better compatibility.

---

### File: `phase01_diagnostic_gpu_check.py` (Complete Source)

```python
import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))
```

#### Analysis
- **What this solves**: Checking whether TensorFlow can see any GPU hardware.
- **Context**: This was run before `phase01_vision_cnn_trainer.py` to verify the development environment. The output determines whether `os.environ["CUDA_VISIBLE_DEVICES"] = "-1"` (disable GPU) should be set in later scripts.
- **Descendants**: No direct code descendant. The GPU check concept survives implicitly in the Phase 9 services, which all set `CUDA_VISIBLE_DEVICES = "-1"` explicitly because they deploy on CPU-only servers.

---

### File: `phase01_vision_classification_report_root.txt` (Complete Content)

```
              precision    recall  f1-score   support

       angry       0.45      0.51      0.48       958
     disgust       0.50      0.03      0.05       111
        fear       0.41      0.18      0.25      1024
       happy       0.78      0.82      0.80      1774
     neutral       0.48      0.64      0.55      1233
         sad       0.45      0.47      0.46      1247
    surprise       0.72      0.72      0.72       831

    accuracy                           0.57      7178
   macro avg       0.54      0.48      0.47      7178
weighted avg       0.56      0.57      0.56      7178
```

#### Analysis
- **Angry (45% precision, 51% recall)**: The model frequently confused angry faces with sad or neutral faces — all three involve furrowed brows and downturned mouths.
- **Disgust (50% precision, 3% recall)**: With only 111 test samples and no class weighting, the model effectively learned to never predict "disgust". The 50% precision means the rare times it did predict disgust, it was right half the time — but it almost never made that prediction.
- **Fear (41% precision, 18% recall)**: Fear expressions (wide eyes, open mouth) were frequently confused with surprise. At 48×48 resolution, the subtle differences in eyebrow position and mouth shape are lost.
- **Happy (78% precision, 82% recall)**: The best-performing class because smiling is visually distinctive even at low resolution, and Happy had the most training samples.
- **Neutral (48% precision, 64% recall)**: High recall but low precision means the model predicted "neutral" too often — it was the "safe default" when uncertain.

#### Comparison to Phase 9 Model
The Phase 9 retrained model (FERPlus, 112×112, 4-block CNN, CLAHE, augmentation) achieves:
- Angry: 76% precision, 79% recall (vs 45%/51%)
- Happy: 92% precision, 89% recall (vs 78%/82%)
- Neutral: 82% precision, 86% recall (vs 48%/64%)
- Overall: **81.03% accuracy** (vs 57%)

---

## Header 3: Micro-Decision Log

| Decision | Old Value | New Value | Rationale |
|---|---|---|---|
| Dataset | FER2013 (`archive_5/`) | FERPlus (Microsoft-corrected) | FER2013 labels are noisy; FERPlus uses crowd-sourced majority voting across 15 annotators |
| Input resolution | 48×48 | 112×112 | Higher resolution preserves subtle facial muscle geometry |
| CNN depth | 3 Conv2D layers (32→64→128) | 4 Conv2D blocks (64→128→256→512) | Deeper network captures hierarchical features |
| BatchNormalization | Absent | After every Conv2D layer | Stabilizes training, enables higher learning rates |
| Data augmentation | None (`rescale` only) | rotation, zoom, shift, horizontal flip | Prevents overfitting to training pose/position |
| Class weighting | Absent | `compute_class_weight('balanced')` + manual neutral reduction (0.77) | Forces optimizer to respect minority classes |
| CLAHE preprocessing | Absent | `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))` | Normalizes lighting across images |
| Interpolation | Default (bilinear) | `cv2.INTER_CUBIC` | Higher quality resize preserves edge detail |
| Callbacks | None | `ModelCheckpoint` + `EarlyStopping(patience=10)` + `ReduceLROnPlateau(patience=5)` | Saves best model, prevents overfitting, adapts learning rate |
| Model format | `.h5` (HDF5) | `.keras` (native Keras 3) | Better serialization, supports custom objects natively |
| Output labels | Title-case: `['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']` | Lowercase alphabetical: `['angry','disgust','fear','happy','neutral','sad','surprise']` | Consistency with audio/text models, API JSON conventions |

---

## Header 4: Inter-Phase Diff Analysis

Phase 1 is the genesis — there is no previous phase to diff against. The following files born in Phase 1 have descendants:

| Phase 1 Artifact | Immediate Descendant | Ultimate Descendant (Phase 9) |
|---|---|---|
| `phase01_vision_cnn_trainer.py` (model trainer) | `phase01_vision_cnn_notebook.ipynb` (same architecture in notebook form) | `training_models_notebook_dump.txt` lines 937-1665 (complete FERPlus Kaggle retraining) |
| `emotion_model.h5` / `face_emotion_model.h5` | `emotion_api/face_emotion_model.h5` (loaded by Phase 6 monolith) | `fer_best_model.keras` (loaded by Phase 9 `phase08_vision_api_preprod.py`) |
| `phase01_diagnostic_gpu_check.py` (GPU check) | No direct descendant | Spiritual descendant: `os.environ["CUDA_VISIBLE_DEVICES"] = "-1"` in all Phase 9 services |
| `phase01_vision_classification_report_root.txt` (57% accuracy) | No direct descendant | Replaced by Phase 9 FERPlus report (81.03% accuracy) |
