# Model Training Documentation: Complete Notebook Archaeology & Comparative Analysis

---

## Overview
This document preserves the **complete model training history** across every iteration — from the earliest Phase 1 FER2013 CNN, through each model format migration, to the final production Kaggle-trained models. Training data was extracted from:
- **`FYP old MODELS apis.txt`** (128KB, 2697 lines) — Complete Kaggle notebook dump for the FERPlus CNN
- **`Model notebooks.txt`** — Kaggle training code for Audio BiLSTM and Text BiLSTM+Attention
- **`cnn model/1.ipynb`** (222KB, 2 code cells) — Phase 1 CNN training notebook with full training logs and evaluation output
- **`11.ipynb`** (189KB, 1 code cell) — Earlier Phase 1 CNN iteration with `1.py`-equivalent architecture
- **`speech.ipynb`** (93KB, 7 code cells) — Live microphone emotion detection, LSTM model loading/testing, TF2.12 format migration, and LSTM trainer
- **`speech1.ipynb`** (47KB, 2 code cells) — Second audio LSTM trainer iteration and `voice_model_tf212_FIXED.h5` rebuilder
- **`textemo.ipynb`** (255KB, 4 code cells) — Complete text training pipeline: first attempt with stemming+context filtering, second clean training run, and universal inference code
- **`text.ipynb`** (35KB, 1 code cell) — Text inference tester with AttentionLayer + paragraph-level voting

---

## 1. Vision Model Evolution: FER2013 → FERPlus CNN

### 1.1 Iteration A: `11.ipynb` — The `1.py` Equivalent (Phase 1)

**Source**: `FYP old/11.ipynb` (189KB)
**Dataset**: FER2013 original (`archive_5/`) — 28,709 training / 7,178 test images
**Architecture**: Minimal 3-layer CNN (identical to `1.py`)
```
Conv2D(32, 3×3, relu) → MaxPool(2×2)
Conv2D(64, 3×3, relu) → MaxPool(2×2)
Conv2D(128, 3×3, relu) → MaxPool(2×2)
Flatten → Dense(128, relu) → Dropout(0.5)
Output: Dense(7, softmax)
```
- **Total params**: 619,015 (2.36 MB)
- **Input**: 48×48 grayscale
- **Preprocessing**: `rescale=1./255` only — no augmentation, no CLAHE
- **No BatchNormalization**, no class weights, no callbacks
- **Training**: 30 epochs, optimizer `adam`, loss `categorical_crossentropy`

**Training Convergence (from notebook output)**:
| Epoch | Train Acc | Val Acc | Val Loss |
|---|---|---|---|
| 1 | 24.5% | 24.8% | 1.7661 |
| 5 | 36.0% | 45.0% | 1.4418 |
| 10 | 43.2% | 49.1% | 1.3400 |
| 20 | 49.8% | 55.1% | 1.1809 |
| 28 | 51.6% | 57.6% | 1.1227 |
| 30 | 52.0% | 57.2% | 1.1291 |

**Test Evaluation**: **57% Accuracy**
```
              precision  recall  f1-score  support
       angry       0.45      0.51      0.48      958
     disgust       0.50      0.03      0.05      111
        fear       0.41      0.18      0.25     1024
       happy       0.78      0.82      0.80     1774
     neutral       0.48      0.64      0.55     1233
         sad       0.45      0.47      0.46     1247
    surprise       0.72      0.72      0.72      831
    accuracy                           0.57     7178
```
**Output Files**: `emotion_model.h5`, `accuracy_plot.png`, `loss_plot.png`, `confusion_matrix.png`, `classification_report.txt`

**Fatal Flaws**: Disgust 3% recall (model learned to never predict it), Fear 18% recall (confused with Surprise at 48×48), no augmentation caused overfitting, no class weights on heavily imbalanced data.

---

### 1.2 Iteration B: `cnn model/1.ipynb` — Improved Phase 1 CNN

**Source**: `FYP old/cnn model/1.ipynb` (222KB)
**Dataset**: FER2013 original (`archive_5/`) — same 28,709/7,178 split
**Architecture**: More sophisticated CNN with BatchNormalization and SeparableConv2D
```
Conv2D(32, 3×3, padding='same') → BatchNorm → Activation('relu')
Conv2D(32, 3×3, padding='same') → BatchNorm → Activation('relu') → MaxPool(2×2)
SeparableConv2D(64, 3×3, padding='same') → BatchNorm → Activation('relu')
SeparableConv2D(64, 3×3, padding='same') → BatchNorm → Activation('relu') → MaxPool(2×2)
SeparableConv2D(128, 3×3, padding='same') → BatchNorm → Activation('relu')
SeparableConv2D(128, 3×3, padding='same') → BatchNorm → Activation('relu') → MaxPool(2×2)
GlobalAveragePooling2D
Dense(7, softmax)
```
- **Total params**: 45,959 (179.53 KB) — **13× smaller** than `11.ipynb`'s model
- **Input**: 48×48 grayscale
- **Key improvement**: Added `compute_class_weight('balanced')` — computed weights: `{disgust: 9.41, fear: 1.00, happy: 0.57, neutral: 0.83, ...}`
- **Callbacks**: `EarlyStopping(patience=10)`, `ReduceLROnPlateau(patience=5, factor=0.5)`
- **Training**: 60 epochs max, `batch_size=64`

**Training Convergence (from notebook output)**:
| Epoch | Train Acc | Val Acc | Val Loss | LR |
|---|---|---|---|---|
| 1 | 18.1% | 5.0% | 1.9732 | 0.001 |
| 6 | 33.9% | 40.7% | 1.5778 | 0.001 |
| 15 | 42.2% | 50.1% | 1.3513 | 0.001 |
| 25 | 46.2% | 50.9% | 1.3400 | 0.0002 |
| 38 | 49.4% | 53.1% | 1.2594 | 8e-6 |
| 41 | 49.4% | 53.1% | 1.2631 | 8e-6 |

**Test Evaluation**: **53% Accuracy**
```
              precision  recall  f1-score  support
       angry       0.38      0.52      0.44      958
     disgust       0.15      0.54      0.24      111
        fear       0.42      0.18      0.25     1024
       happy       0.80      0.75      0.77     1774
     neutral       0.48      0.60      0.53     1233
         sad       0.50      0.27      0.35     1247
    surprise       0.57      0.79      0.66      831
    accuracy                           0.53     7178
```
**Output Files**: `face_emotion_model.h5`, `accuracy_plot.png`, `loss_plot.png`, `confusion_matrix.png`, `classification_report.txt`

**Key Insight**: Lower overall accuracy (53% vs 57%) BUT Disgust recall jumped from **3% → 54%** thanks to class weighting. The model was too small (45K params) — SeparableConv2D + GlobalAvgPool was too aggressive at reducing parameters. Still no CLAHE, still 48×48.

---

### 1.3 Iteration C: FERPlus Kaggle Retraining — Production Model

**Source**: `FYP old MODELS apis.txt` (128KB Kaggle dump)
**Dataset**: **FERPlus** (Microsoft-corrected labels) — 58,379 training / 7,341 validation / 3,543 test
**Architecture**: 4-Block CNN (14.5M params)
```
Block 1: Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
Block 2: Conv2D(128, 5×5) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
Block 3: Conv2D(256, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
Block 4: Conv2D(512, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
Flatten → Dense(512) → BatchNorm → ReLU → Dropout(0.50)
Output: Dense(7, softmax)
```
- **Total params**: 14,535,943 (55.45 MB)
- **Input**: **112×112** grayscale with **CLAHE** preprocessing + **INTER_CUBIC** resize
- **Augmentation**: `rotation_range=10, zoom_range=0.1, width/height_shift=0.1, horizontal_flip=True`
- **Callbacks**: `ModelCheckpoint`, `EarlyStopping(patience=10)`, `ReduceLROnPlateau(patience=5, factor=0.2)`
- **Class Weights**: `{neutral: 0.77, all others: 1.0}`

**Training History**:
| Epoch | Train Acc | Val Acc | Val Loss | LR | Event |
|---|---|---|---|---|---|
| 1 | 34.6% | 57.6% | 1.1666 | 0.001 | First checkpoint |
| 4 | 69.4% | 70.7% | 0.8086 | 0.001 | Major jump |
| 15 | 84.8% | 77.0% | 0.7541 | 0.0002 | Best val_acc |
| 18 | 86.8% | 77.1% | 0.7764 | 0.0002 | Final best checkpoint |
| 19 | 87.3% | 76.3% | 0.8557 | 4e-5 | Early stopping |

**Test Evaluation**: **81.03% Accuracy**
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
**Output File**: `fer_best_model.keras` — **ACTIVE production model**

**Gradio Demo**: Deployed on Kaggle at `https://dca61cfec8fced296c.gradio.live`

---

### Vision Model Comparative Summary

| Attribute | `11.ipynb` (Phase 1a) | `cnn model/1.ipynb` (Phase 1b) | FERPlus Kaggle (Production) |
|---|---|---|---|
| **Phase** | Phase 1 | Phase 1 | Post-Phase 7 (Kaggle) |
| **Dataset** | FER2013 (`archive_5`) | FER2013 (`archive_5`) | **FERPlus** (corrected labels) |
| **Input Size** | 48×48 | 48×48 | **112×112** |
| **CLAHE** | ❌ | ❌ | ✅ `clipLimit=2.0` |
| **Architecture** | 3-Conv Sequential | 6-Conv + SeparableConv2D | **4-Block CNN** |
| **Parameters** | 619K (2.36 MB) | 45K (180 KB) | **14.5M (55 MB)** |
| **BatchNorm** | ❌ | ✅ | ✅ |
| **Class Weights** | ❌ | ✅ (`compute_class_weight`) | ✅ (manual) |
| **Augmentation** | ❌ | ❌ | ✅ (rotation, zoom, shift, flip) |
| **Callbacks** | ❌ | ✅ (EarlyStopping, ReduceLR) | ✅ (all three) |
| **Epochs** | 30 (ran all) | 60 (early stop ~41) | 50 (early stop ~19) |
| **Best Val Acc** | 57.6% | 53.1% | **77.1%** |
| **Test Accuracy** | **57%** | **53%** | **81.03%** |
| **Disgust Recall** | 3% | **54%** | **57%** |
| **Fear Recall** | 18% | 18% | **49%** |
| **Happy Precision** | 78% | 80% | **92%** |
| **Model File** | `emotion_model.h5` | `face_emotion_model.h5` | `fer_best_model.keras` |
| **Status** | Superseded | Superseded | **ACTIVE** |

**Key Takeaway**: The 57% → 81% accuracy jump came from **four simultaneous improvements**: (1) FERPlus labels, (2) 112×112 + CLAHE, (3) 14.5M param architecture with 5×5 kernels, (4) data augmentation. No single change was sufficient alone — `cnn model/1.ipynb` proved that class weights alone (with a tiny model) actually *reduced* overall accuracy while improving minority class recall.

---

## 2. Audio Model Evolution: Simple LSTM → BiLSTM → TFLite

### 2.1 `speech.ipynb` — Phase 2 Audio Training & TF2.12 Migration

**Source**: `FYP old/speech.ipynb` (93KB, 7 code cells)

**Cell 0 (Live Microphone Detection)**:
- Loads `speech_emotion_model_7_lstm_clean.h5` — the "clean" variant of the Phase 2 model
- Uses `sounddevice` for live microphone recording + `tkinter` file dialog for WAV selection
- Labels: `['angry','calm','disgust','fearful','happy','neutral','sad','surprised']` — **8 classes** (calm NOT merged)
- Features: `librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)` padded to 174 time steps
- Notebook output shows test with RAVDESS file `Actor_12/03-01-02-02-02-01-12.wav` → `surprised 99.99%`

**Cell 2 (Debug Cell)**:
- Attempted to load RAVDESS audio with `librosa.load(test_file, res_type='kaiser_fast')`
- **Failed with `ModuleNotFoundError: No module named 'resampy'`** — dependency missing on dev machine

**Cell 3 (Second Live Detection Variant)**:
- Changed to `MODEL_PATH = "final_lstm_model.h5"` (the main Phase 2 model, not the clean variant)
- Same 8-class label list, same MFCC extraction pipeline

**Cell 4 (LSTM Trainer — The Core Training Code)**:
- **Dataset**: RAVDESS (`archive_6/`), emotion_map merges calm(02)→neutral
- **Architecture**: `Sequential LSTM(128) → Dropout(0.3) → Dense(64, relu) → Dropout(0.3) → Dense(7, softmax)`
- **Features**: 40 MFCCs + delta + delta2 stacking → shape (174, 120)
- **Class weights**: `compute_class_weight('balanced')`
- **Callbacks**: `EarlyStopping(patience=10)`, `ReduceLROnPlateau(patience=5)`
- **Training**: 40 epochs, `batch_size=32`

**Cell 5 (TF2.12 Format Migration)**:
- Loaded `emotion_api/final_lstm_model.h5` (the old format model)
- **Architecture revealed**: `Bidirectional(LSTM(128, return_sequences=True)) → Dropout(0.3) → Bidirectional(LSTM(64)) → Dropout(0.3) → Dense(64, relu) → Dropout(0.3) → Dense(8, softmax)` — **428,106 params (1.63 MB)**
- **Input shape**: `(174, 120)` — confirms 40 MFCCs × 3 (base + delta + delta2) = 120 features
- **Dense(8, softmax)** — **8 classes** (calm NOT merged in the model, merged at inference time)
- Rebuilt as `voice_emotion_model_tf212` with manually copied weights
- Saved as `emotion_api/final_lstm_model_tf212.h5`

**Cell 6 (Verification)**:
- Loaded `emotion_api/voice_model_tf212_FIXED.h5` — the final fixed model
- Verified architecture matches: 428,104 params, same BiLSTM stack

### 2.2 `speech1.ipynb` — Phase 2 Second Training Iteration

**Source**: `FYP old/speech1.ipynb` (47KB, 2 code cells)

**Cell 0 (LSTM Trainer)**:
- Near-identical to `speech.ipynb` Cell 4 but as a `Sequential` model:
- `LSTM(128) → Dropout(0.3) → Dense(64, relu) → Dropout(0.3) → Dense(7, softmax)`
- Same RAVDESS dataset, same calm→neutral merge, same MFCC extraction
- **Note**: This notebook ran on a **different Python environment** (`anaconda3/envs/ev_1`, Python 3.6) and **crashed** with `ImportError: DLL load failed` (numpy/mkl incompatibility)

**Cell 1 (TF2.12 Rebuilder)**:
- Loaded `emotion_api/final_lstm_model_tf212.h5`, rebuilt as `voice_model_tf212_fixed`
- Output Dense: `Dense(8, softmax)` — confirms 8-class model architecture
- Saved to `emotion_api/voice_model_tf212_FIXED.h5`

### 2.3 Production Model — `audio_best_model.keras` → `audio_model.tflite`

**Source**: `Model notebooks.txt` (Kaggle training)
- **Dataset**: RAVDESS
- **Architecture**: BiLSTM with custom Attention mechanism
- **Final accuracy**: **94.10%** on test set
- **Emotion classes**: 7 (calm merged with neutral at training time)
- **Features**: 40 MFCCs, padded to 174 time steps
- **Training**: K-fold cross-validation (5 folds — `lstm_fold1_best.h5` through `lstm_fold5_best.h5`)

**TFLite Conversion** (`convert_audio_model.py`):
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter._experimental_lower_tensor_list_ops = False  # Required for BiLSTM
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS  # Required for dynamic LSTM loops
]
```

### Audio Model Comparative Summary

| Attribute | `speech.ipynb` Cell 4 (Phase 2) | `speech1.ipynb` (Phase 2) | Production (Kaggle) |
|---|---|---|---|
| **Architecture** | Sequential LSTM(128) | Sequential LSTM(128) | **BiLSTM + Attention** |
| **Bidirectional** | ❌ (unidirectional) | ❌ (unidirectional) | ✅ |
| **Attention** | ❌ | ❌ | ✅ |
| **Params** | ~100K | ~100K | ~428K |
| **Input Features** | 40 MFCC + delta + delta2 (120) | 40 MFCC + delta + delta2 (120) | 40 MFCC only |
| **Output Classes** | 7 (calm merged) | 7 (calm merged) | 7 (calm merged) |
| **K-Fold** | ❌ | ❌ | ✅ (5-fold) |
| **Test Accuracy** | Unknown (not captured) | Crashed | **94.10%** |
| **Model File** | `speech_emotion_model_7.h5` | `final_lstm_model.h5` | `audio_model.tflite` |
| **Format** | HDF5 | HDF5 | **TFLite float16** |
| **Status** | Superseded | Superseded | **ACTIVE** |

**Key Takeaway**: The major accuracy jump came from (1) switching to BiLSTM + Attention, (2) dropping MFCC deltas (the BiLSTM captures temporal dynamics natively), and (3) K-fold cross-validation for more robust training. The model format migrated from HDF5 → Keras 3 → TFLite for deployment latency.

---

## 3. Text Model Evolution: CNN → BiLSTM → BiLSTM+Attention

### 3.1 `textemo.ipynb` — Phase 4 Complete Text Training Pipeline

**Source**: `FYP old/textemo.ipynb` (255KB, 4 code cells)

**Cell 0 (Inference with Stemming + Context Check)**:
- First `AttentionLayer` using `tf.tensordot` (TF 2.x compatible)
- `sentence_has_emotion_context()` — keyword-based context filter
- `handle_negations()` — regex-based negation engine
- `PorterStemmer` imported and initialized but **never used in the prediction path**
- Uses `nltk.download('punkt')` at runtime

**Cell 1 (First Training Attempt — Oversampled)**:
- **Dataset**: 414,337 samples after sexual/explicit filtering (1,771 removed to reduce LOVE bias)
- **Classes**: `['anger', 'fear', 'joy', 'love', 'sad', 'suprise']` — 6 classes (note `suprise` typo)
- **Balancing**: Oversampled ALL classes to 140,301 each → **841,806 total** balanced samples
- Training/test split: 715,535 / 126,271
- **Architecture**: Functional API BiLSTM + AttentionLayer
  - `Input(60) → Embedding(30000, 128) → Bidirectional(LSTM(128, return_sequences=True)) → AttentionLayer → Dropout → Dense(128) → Dropout → Dense(6, softmax)`
  - **Total params**: 4,137,095 (15.78 MB)
- **Training**: 20 epochs, early stopped mid-epoch 1 at `accuracy: 0.7435, loss: 0.8364` (first run partial)
- **TF-IDF analysis computed** before training for feature exploration

**Cell 2 (Clean Training Run — No Filtering)**:
- **Title**: "Clean stable training pipeline — No negation handling, No calm→neutral merge, No sexual filtering, No keyword/context filtering"
- **Dataset**: 422,746 raw samples, **no** oversampling, **no** sexual filtering
- Training/test split: 359,334 / 63,412
- **Same architecture**: 4,137,095 params
- **MAX_LEN = 60**, Vocabulary size: 30,000
- **Training**: 20 epochs, partial output captured at epoch 1: `accuracy: 0.6905, loss: 0.7822`
- This cell was run **3 times** (evidenced by triplicated output blocks in the notebook)

**Cell 3 (Universal Inference Code)**:
- Dynamic model discovery: `glob.glob("trained_models/**/*.h5", recursive=True)`
- Found model at `trained_models/aa/emotion_model_20251209_235802.h5` (timestamped!)
- **Load failed**: `"object 'model_config' doesn't exist"` — HDF5 format incompatibility with Keras 3
- This failure is what drove the migration to `.keras` format in production

### 3.2 `text.ipynb` — Phase 4 Inference Tester

**Source**: `FYP old/text.ipynb` (35KB, 1 code cell)
- Pure inference script, no training
- Custom `AttentionLayer` using `K.dot(x, self.W)` (Keras backend ops)
- Paragraph-level voting: `sentence → predict → vote_counter + prob_accumulator → paragraph_emotion`
- Loaded model successfully with Keras `input_length` deprecation warnings

### 3.3 Production Model — `text_best_model.keras`

**Source**: `Model notebooks.txt` (Kaggle training)
- **Dataset**: Merged Kaggle emotion text datasets (with `'suprise'` → `'surprise'` typo fix)
- **Architecture**: `Embedding(30000, 128) → BiLSTM(128) → Custom AttentionLayer → Dense(64) → Dense(6, softmax)`
- **Final accuracy**: **94.04%** on test set
- **Emotion classes**: 6 (joy, sad, anger, fear, love, surprise)
- **Preprocessing**: Keras Tokenizer, `MAX_VOCAB_SIZE = 30000`, `MAX_LEN = 100`, `padding='post'`
- **Training**: 15 epochs with early stopping, class weights for imbalanced classes

### Text Model Comparative Summary

| Attribute | `textemo.ipynb` Cell 1 (Phase 4a) | `textemo.ipynb` Cell 2 (Phase 4b) | Production (Kaggle) |
|---|---|---|---|
| **Dataset Size** | 841,806 (oversampled) | 422,746 (raw) | ~400K (merged Kaggle) |
| **Sexual Filtering** | ✅ (1,771 removed) | ❌ | Unknown |
| **Oversampling** | ✅ (140,301/class) | ❌ | ❌ (class weights instead) |
| **MAX_LEN** | 60 | 60 | **100** |
| **Vocabulary** | 30,000 | 30,000 | 30,000 |
| **Architecture** | BiLSTM(128) + Attention | BiLSTM(128) + Attention | BiLSTM(128) + Attention |
| **Params** | 4.14M | 4.14M | ~4.14M |
| **Epochs** | 20 (partial) | 20 (partial) | **15** (early stop) |
| **Negation Handling** | ✅ (in inference) | ❌ | ❌ |
| **Context Filter** | ✅ (keyword-based) | ❌ | ❌ (dual-threshold instead) |
| **Model Format** | `.h5` (HDF5) | `.h5` (HDF5) | **`.keras`** (Keras 3) |
| **Test Accuracy** | Unknown (partial run) | Unknown (partial run) | **94.04%** |
| **Status** | Superseded | Superseded | **ACTIVE** |

**Key Takeaway**: The text model architecture (BiLSTM + Attention, 30K vocab, 128 embedding) remained remarkably stable across all iterations. The accuracy improvement came primarily from (1) increasing `MAX_LEN` from 60 → 100 (capturing more context), (2) replacing keyword-based filtering with a principled dual-threshold confidence gate, (3) cleaner dataset preparation on Kaggle, and (4) switching from HDF5 to native `.keras` format which resolved the `model_config` loading failures.

---

## 4. Audio Model Format Migration Chain

The audio model went through a complex format migration documented across `speech.ipynb` and `speech1.ipynb`:

```
speech_emotion_model_7.h5 (Phase 2 original, Sequential LSTM)
     ↓
final_lstm_model.h5 (BiLSTM rebuild, 428K params, input shape 174×120)
     ↓
final_lstm_model_tf212.h5 (speech.ipynb Cell 5: manual weight copy to TF2.12-compatible model)
     ↓
voice_model_tf212_FIXED.h5 (speech1.ipynb Cell 1: final clean rebuild with proper layer naming)
     ↓
audio_best_model.keras (Kaggle retrained BiLSTM+Attention, 94.10%)
     ↓
audio_model.tflite (convert_audio_model.py: float16 quantized TFLite for production)
```

Each migration was triggered by TensorFlow/Keras version incompatibilities when loading old HDF5 models on newer TF versions. The final Kaggle retraining broke this chain entirely by training a fresh model with modern Keras 3.

---

## 5. Legacy Training Artifacts Inventory

| Artifact | Phase | Notebook Source | Notes |
|---|---|---|---|
| `speech.ipynb` (93KB) | Phase 2 | — | 7 cells: live detection + LSTM trainer + TF2.12 migration |
| `speech1.ipynb` (47KB) | Phase 2 | — | 2 cells: LSTM trainer (crashed on Python 3.6) + TF2.12 rebuilder |
| `textemo.ipynb` (255KB) | Phase 4 | — | 4 cells: stemming inference + 2 training runs + universal inference |
| `text.ipynb` (35KB) | Phase 4 | — | 1 cell: pure inference tester with AttentionLayer |
| `cnn model/1.ipynb` (222KB) | Phase 1 | — | 2 cells: improved CNN trainer (53% acc) + face detector UI |
| `11.ipynb` (189KB) | Phase 1 | — | 1 cell: original `1.py`-equivalent CNN (57% acc) |
| `test.ipynb` (0 bytes) | — | — | Empty file (deleted or never written) |
| `lstm_fold1-5_best.h5` | Phase 2 | Kaggle | K-fold validation checkpoints for BiLSTM+Attention |
| `lstm_histories.pkl` | Phase 2 | Kaggle | Pickled training curves across 5 folds |
| `lstm_confusion_matrix.png` | Phase 2 | Kaggle | Visual evaluation of LSTM model |
| `lstm_training_curves.png` | Phase 2 | Kaggle | Training/validation convergence plots |
| `accuracy_plot.png` | Phase 1 | `cnn model/1.ipynb` | CNN training accuracy curves |
| `loss_plot.png` | Phase 1 | `cnn model/1.ipynb` | CNN training loss curves |
| `confusion_matrix.png` | Phase 1 | `cnn model/1.ipynb` | Phase 1 CNN confusion matrix |
| `classification_report.txt` | Phase 1 | `11.ipynb` | 57% accuracy report |

---

## 6. Cross-Modality Production Model Summary

| Modality | Model File | Architecture | Params | Test Accuracy | Emotion Classes | Status |
|---|---|---|---|---|---|---|
| **Vision** | `fer_best_model.keras` | 4-Block CNN (14.5M) | 14.5M | **81.03%** | 7 (angry, disgust, fear, happy, neutral, sad, surprise) | **ACTIVE** |
| **Audio** | `audio_model.tflite` | BiLSTM + Attention (float16) | ~428K | **94.10%** | 7 (angry, disgust, fearful, happy, neutral, sad, surprised) | **ACTIVE** |
| **Text** | `text_best_model.keras` | BiLSTM + Attention (4.14M) | 4.14M | **94.04%** | 6 (anger, fear, joy, love, sad, surprise) | **ACTIVE** |
