# Model Training Documentation: Complete Kaggle Notebook Archaeology

---

## Overview
This document preserves the complete model training history from `FYP old MODELS apis.txt` (128KB, 2697 lines) and `Model notebooks.txt`. These files contain raw Kaggle notebook dumps — including code cells, output cells, error messages, and interactive results — for all three production models.

---

## 1. FERPlus CNN (Vision Model) — `fer_best_model.keras`

### Dataset: FERPlus
- **Source**: FERPlus (Microsoft's re-annotated FER2013)
- **Training split**: 58,379 images (angry:8000, disgust:8000, fear:8000, happy:8000, neutral:10,379, sad:8000, surprise:8000)
- **Validation split**: 7,341 images
- **Test split**: 3,543 images (imbalanced: Disgust only 21 samples, Fear only 98 samples)
- **Known issue**: Folder name `'suprise'` (typo for `'surprise'`) persists throughout training — the model learns the mapping correctly regardless

### Preprocessing Pipeline
```python
# CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Per-image processing:
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 1. Read as grayscale
img = clahe.apply(img)                              # 2. CLAHE lighting correction
img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_CUBIC)  # 3. Resize
img_normalized = img.astype('float32') / 255.0      # 4. Normalize to [0,1]
img_expanded = np.expand_dims(img_normalized, axis=-1)  # 5. Add channel dim
```

### Architecture: 4-Block CNN (14.5M params)
```
Block 1: Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
Block 2: Conv2D(128, 5×5) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)  ← 5×5 kernel for broader features
Block 3: Conv2D(256, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
Block 4: Conv2D(512, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
Flatten → Dense(512) → BatchNorm → ReLU → Dropout(0.50)
Output: Dense(7, softmax)
```
- **Total params**: 14,535,943 (55.45 MB)
- **Trainable params**: 14,532,999
- **Note**: Block 2 uses a 5×5 kernel (all others use 3×3) — this was intentional to capture broader facial features like eyebrow arches and lip shapes.

### Training Configuration
```python
optimizer = Adam(learning_rate=0.001)
loss = 'sparse_categorical_crossentropy'
batch_size = 64
max_epochs = 50

# Data Augmentation (training set only):
rotation_range = 10        # Head tilts
zoom_range = 0.1           # Camera distance variation
width_shift_range = 0.1    # Horizontal panning
height_shift_range = 0.1   # Vertical panning
horizontal_flip = True     # Face flipping

# Callbacks:
ModelCheckpoint('fer_best_model.keras', monitor='val_accuracy', save_best_only=True)
EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# Class Weights (Neutral downweighted):
class_weights_dict = {0:1.0, 1:1.0, 2:1.0, 3:1.0, 4:0.77, 5:1.0, 6:1.0}
```

### Training History (From Notebook Output)
| Epoch | Train Acc | Val Acc | Val Loss | LR | Event |
|---|---|---|---|---|---|
| 1 | 34.6% | 57.6% | 1.1666 | 0.001 | First checkpoint |
| 4 | 69.4% | 70.7% | 0.8086 | 0.001 | Major jump |
| 7 | 76.1% | 74.3% | 0.7602 | 0.001 | |
| 10 | 80.1% | 74.6% | 0.7628 | 0.001 | |
| 14 | 83.1% | 72.2% | 0.8675 | 0.001 | ReduceLR triggered → 0.0002 |
| 15 | 84.8% | 77.0% | 0.7541 | 0.0002 | **Best val_acc before LR drop** |
| 18 | 86.8% | 77.1% | 0.7764 | 0.0002 | Final best checkpoint |
| 19 | 87.3% | 76.3% | 0.8557 | 4e-5 | **Early stopping triggered** |

**Best model restored from epoch 9 weights** (per EarlyStopping's `restore_best_weights`)

### Test Evaluation: **81.03% Accuracy**
```
              precision  recall  f1-score  support
     Angry      0.76     0.79     0.77      322
    Disgust     0.46     0.57     0.51       21
       Fear     0.53     0.49     0.51       98
      Happy     0.92     0.89     0.90      929
    Neutral     0.82     0.86     0.84     1274
        Sad     0.67     0.60     0.63      449
   Surprise     0.81     0.82     0.81      450
   accuracy                       0.81     3543
```
- **Strengths**: Happy (0.92 precision, 0.89 recall), Neutral (0.82/0.86), Surprise (0.81/0.82)
- **Weaknesses**: Disgust (0.46/0.57 — but only 21 test samples), Fear (0.53/0.49)
- **Comparison to Phase 1**: 57% → **81%** accuracy. Disgust recall: 3% → **57%**. Fear recall: 18% → **49%**.

### Gradio Demo
The notebook includes a Gradio web demo deployed on Kaggle:
```python
interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(sources=["upload", "webcam"]),
    outputs=gr.Label(num_top_classes=3),
    title="Real-Time 7-Emotion Detector",
)
interface.launch(debug=True)
# Running on public URL: https://dca61cfec8fced296c.gradio.live
```

---

## 2. Audio BiLSTM — `audio_best_model.keras` → `audio_model.tflite`

### Key Facts (from `Model notebooks.txt`)
- **Dataset**: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Architecture**: Bidirectional LSTM with attention mechanism
- **Final accuracy**: **94.10%** on test set
- **Emotion classes**: 7 (angry, disgust, fearful, happy, neutral, sad, surprised) — calm merged with neutral at training time
- **Features**: 40 MFCCs, padded to 174 time steps
- **Training**: K-fold cross-validation (5 folds — evidenced by `lstm_fold1_best.h5` through `lstm_fold5_best.h5`)

### TFLite Conversion
```python
# convert_audio_model.py
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter._experimental_lower_tensor_list_ops = False  # Required for BiLSTM
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS  # Required for dynamic LSTM loops
]
```

---

## 3. Text BiLSTM+Attention — `text_best_model.keras`

### Key Facts (from `Model notebooks.txt`)
- **Dataset**: Merged Kaggle emotion text datasets (with `'suprise'` → `'surprise'` typo fix)
- **Architecture**: Embedding(30000, 128) → BiLSTM(128) → Custom AttentionLayer → Dense(64) → Dense(6, softmax)
- **Final accuracy**: **94.04%** on test set
- **Emotion classes**: 6 (joy, sad, anger, fear, love, surprise)
- **Preprocessing**: Keras Tokenizer with `MAX_VOCAB_SIZE = 30000`, `MAX_LEN = 100`, `padding='post'`
- **Training**: 15 epochs with early stopping, class weights for imbalanced classes

---

## 4. Legacy Training Artifacts

| Artifact | Phase | Notes |
|---|---|---|
| `speech.ipynb` (93KB) | Phase 2 | Audio model training notebook |
| `speech1.ipynb` (47KB) | Phase 2 | Second audio training iteration |
| `textemo.ipynb` (255KB) | Phase 4 | Text model training notebook |
| `text.ipynb` (35KB) | Phase 4 | Lighter text exploration |
| `11.ipynb` (189KB) | Unknown | Large notebook, likely training experiments |
| `test.ipynb` | Unknown | Test/experimentation notebook |
| `lstm_fold1-5_best.h5` | Phase 2 | K-fold validation checkpoints |
| `lstm_histories.pkl` | Phase 2 | Pickled training curves across 5 folds |
| `lstm_confusion_matrix.png` | Phase 2 | Visual evaluation of LSTM model |
| `lstm_training_curves.png` | Phase 2 | Training/validation convergence plots |
| `accuracy_plot.png` | Phase 1 | CNN training accuracy curves |
| `loss_plot.png` | Phase 1 | CNN training loss curves |
| `confusion_matrix.png` | Phase 1 | Phase 1 CNN confusion matrix |
| `classification_report.txt` | Phase 1 | 57% accuracy report |
