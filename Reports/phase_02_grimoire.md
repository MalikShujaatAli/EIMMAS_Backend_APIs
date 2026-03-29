# Phase 2 Grimoire: Complete Technical Archaeology

---

## Header 1: Complete File Inventory

| File | Size (bytes) | Role |
|---|---|---|
| `FYP old/v4.py` | 2,863 | LSTM audio emotion model trainer |

---

## Header 2: Line-by-Line Logic Migration

### File: `v4.py` (Complete Source)

```python
import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# 🎵 Emotion labels (RAVDESS → merge neutral + calm)
emotion_map = {
    '01': 'neutral',   # merge neutral
    '02': 'neutral',   # merge calm → neutral
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Path to dataset
DATA_PATH = "archive_6"   # <-- change this to your dataset folder

def extract_features(file_path, max_pad_len=174):
    """
    Extract MFCC features from an audio file and pad/truncate to fixed length
    """
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print("Error:", file_path, e)
        return None

# Load dataset
X, y = [], []
for root, _, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            # Extract emotion from filename (e.g., 03-01-05-01-02-01-12.wav → 'happy')
            emotion = emotion_map[file.split("-")[2]]
            feature = extract_features(file_path)
            if feature is not None:
                X.append(feature)
                y.append(emotion)

X = np.array(X)
y = pd.get_dummies(y).values   # One-hot encode labels

print("✅ Data loaded:", X.shape, y.shape)  # should show (#samples, 40, 174) and (#samples, 7)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for LSTM [samples, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[1])

# 🎤 Build LSTM Model
model = Sequential([
    LSTM(128, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y.shape[1], activation='softmax')  # 7 outputs
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40, batch_size=32)

# Save trained model
model.save("speech_emotion_model_7.h5")
print("🎉 Model saved as speech_emotion_model_7.h5")
```

#### Block 1: Imports (Lines 1-8)
- **What this solves**: Loading audio processing (`librosa`), numerical computation (`numpy`), and deep learning (`tensorflow`) libraries.
- **Logical flaw**: `pandas` is imported solely for `pd.get_dummies()` on line 56. In the Phase 8 Kaggle notebook, this is replaced by integer labels + `sparse_categorical_crossentropy`, eliminating the pandas dependency entirely.

#### Block 2: Emotion Mapping (Lines 10-20)
- **What this solves**: Translating RAVDESS filename emotion codes (2-digit strings) to human-readable labels.
- **Critical decision — Calm/Neutral merge**: RAVDESS code `02` (calm) is mapped to `'neutral'`, merging two distinct emotional states. This was a deliberate choice to align the audio model's output classes with the face model's 7-class structure (which has no "calm" class). This merge survives through all phases into Phase 8.
- **Label ordering consequence**: Because labels are one-hot encoded via `pd.get_dummies()`, the column ordering is **alphabetical**: `['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']`. But since calm was merged into neutral, the actual one-hot has 7 columns. The exact ordering depends on which unique labels appear. This ambiguity was a source of bugs — `v5.py` uses `EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust']` (a completely different order), causing potential index mismatches when loading this model.
- **Phase 8 resolution**: The Kaggle retraining uses an explicit `EMOTION_TO_INT` dictionary (`{'angry':0, 'disgust':1, 'fearful':2, 'happy':3, 'neutral':4, 'sad':5, 'surprised':6}`) and `sparse_categorical_crossentropy`, eliminating all ordering ambiguity.

#### Block 3: Feature Extraction (Lines 25-40)
- **What this solves**: Converting raw audio waveforms into fixed-size MFCC tensors.
- **`librosa.load(file_path, res_type='kaiser_fast')`**: Loads audio and resamples to librosa's default 22050 Hz. The `res_type='kaiser_fast'` uses a fast but lower-quality resampling algorithm. In the Phase 8 Kaggle notebook, this is replaced by `librosa.load(file_path, sr=SAMPLE_RATE)` with `SAMPLE_RATE = 16000` (explicit 16kHz, which is optimal for speech) and the `res_type` parameter is dropped entirely (it caused crashes on newer Kaggle environments).
- **No silence trimming**: Raw audio including leading/trailing silence is passed directly to MFCC extraction. This means 1-2 seconds of dead air at recording boundaries are encoded as zero-valued frequency bands, biasing the model toward predicting "neutral" for any input with quiet segments. Phase 8's `2nd attempt Audio.txt` adds `librosa.effects.trim(audio, top_db=30)`.
- **No normalization**: Raw MFCC values are used as-is. MFCC magnitudes vary with recording volume — a loud recording produces higher absolute MFCC values than a quiet one, regardless of emotional content. This causes the "False Angry" bias. Phase 8's `main_audio.py` adds Z-score normalization: `mfccs = (mfccs - mean) / std`.
- **MFCC-only features**: Only 40 MFCC bands are extracted. The Phase 6 monolith (`emotion_api/main.py`) later experiments with stacking MFCC + delta + delta2 into 120 features per frame, but the Phase 8 production model reverts to MFCC-only (40 features) because the BiLSTM architecture captures temporal dynamics internally, making explicit delta features redundant.
- **Return shape**: `mfccs` has shape `(40, 174)` — 40 frequency bands × 174 time frames. This is transposed before feeding to the LSTM (see Block 5).

#### Block 4: Dataset Loading Loop (Lines 43-54)
- **What this solves**: Walking the RAVDESS directory, parsing filenames, building feature/label arrays.
- **Filename parsing**: `file.split("-")[2]` extracts the 3rd segment of the RAVDESS filename convention. For `03-01-05-01-02-01-12.wav`, this yields `'05'` → mapped to `'angry'`. This parsing logic is identical across all phases.
- **No augmentation**: Each file produces exactly one feature vector. The Phase 8 Kaggle notebook's `extract_and_augment()` function produces three vectors per file: clean, noise-injected, and pitch-shifted, tripling the effective dataset size from ~2,880 to ~8,640 samples.

#### Block 5: Encoding and Reshaping (Lines 56-65)
- **`pd.get_dummies(y).values`**: One-hot encodes string labels. The column order is determined by pandas' alphabetical sorting of unique values. This creates a fragile coupling between the training label order and the inference label order that must be manually maintained.
- **Reshape `(samples, 174, 40)`**: The raw MFCC shape is `(40, 174)` — 40 bands × 174 frames. LSTMs expect `(timesteps, features)`, so the axes are swapped: each "timestep" is one frame (of 174), and each frame has 40 MFCC coefficients. This reshape convention is preserved through all phases.

#### Block 6: LSTM Architecture (Lines 68-74)
- **Architecture**: `LSTM(128) → Dropout(0.3) → Dense(64, relu) → Dropout(0.3) → Dense(7, softmax)`.
- **Single LSTM layer**: Only one LSTM with 128 units and `return_sequences=False`. This captures temporal patterns but cannot model hierarchical temporal abstractions.
- **Phase 8 upgrade**: The Kaggle retraining uses `Bidirectional(LSTM(128, return_sequences=True)) → BatchNorm → Dropout → Bidirectional(LSTM(64, return_sequences=False)) → BatchNorm → Dropout → Dense(64, relu) → BatchNorm → Dropout → Dense(7, softmax)`. Key improvements: bidirectional (reads audio forward AND backward), two stacked LSTM layers, BatchNormalization after each layer.
- **No class weighting in `model.fit()`**: Identical to Phase 1's face model, the training loop does not use `class_weight`. The merged neutral class has double the samples of other emotions, creating bias.

#### Block 7: Training and Saving (Lines 76-83)
- **`epochs=40, batch_size=32`**: Fixed training duration with no early stopping. In the Phase 8 Kaggle notebook, training runs for up to 100 epochs with `EarlyStopping(patience=12)` and `ReduceLROnPlateau(patience=5, factor=0.5)`.
- **No `ModelCheckpoint`**: The final model is saved, not the best model. If the model overfits during later epochs, the saved model is worse than the best intermediate version.
- **Output file**: `speech_emotion_model_7.h5` — referenced in Phase 3 (`v5.py`, `v6.py`) and Phase 6 as `speech_emotion_model_7(2).h5`.

---

## Header 3: Micro-Decision Log

| Decision | Phase 2 Value | Phase 8 Value | Rationale for Change |
|---|---|---|---|
| Sample rate | 22050 Hz (librosa default) | 16000 Hz (explicit) | 16kHz is optimal for speech; higher rates waste computation on inaudible frequencies |
| Resampling | `res_type='kaiser_fast'` | Removed | Caused crashes on newer environments; default resampler is sufficient |
| Silence trimming | Absent | `librosa.effects.trim(audio, top_db=30)` | Removes dead air that biases toward "neutral" |
| Feature normalization | Absent | Z-score: `(mfccs - mean) / std` | Eliminates "False Angry" volume bias |
| Augmentation | None | White noise + pitch shift + time stretch | Triples dataset size; prevents overfitting to studio conditions |
| Label encoding | `pd.get_dummies()` (one-hot, alphabetical) | `EMOTION_TO_INT` dict + `sparse_categorical_crossentropy` | Explicit, deterministic label ordering |
| Architecture | Single LSTM(128) | 2× Bidirectional LSTM + BatchNorm | Deeper temporal modeling, bidirectional context |
| Class weighting | Absent | `compute_class_weight('balanced')` | Corrects neutral-class overrepresentation |
| Callbacks | None | ModelCheckpoint + EarlyStopping + ReduceLROnPlateau | Saves best model, prevents overfitting, adapts LR |
| Training epochs | 40 (fixed) | 100 (with early stopping at ~21) | Trains until convergence, not until a fixed number |
| Model format | `.h5` | `.keras` (then `.tflite` for inference) | Better serialization; TFLite for CPU speed |

---

## Header 4: Inter-Phase Diff Analysis

| Phase 2 Artifact | Immediate Descendant (Phase 3) | Ultimate Descendant (Phase 8) |
|---|---|---|
| `v4.py` (trainer script) | Not directly reused; model file loaded by `v5.py`/`v6.py` | `Model notebooks.txt` lines 2057-2400 (Kaggle BiLSTM retraining) |
| `speech_emotion_model_7.h5` (model file) | Loaded by `v5.py`, `v6.py` | Replaced by `audio_best_model.keras` → converted to `audio_model.tflite` |
| `extract_features()` function | Reused nearly identically in `v5.py`/`v6.py` | Replaced by `get_features_fast()` in `main_audio.py` (adds trimming, Z-score normalization, in-memory processing via soundfile) |
| `emotion_map` dictionary | Reused in `emotion_api/main.py` as `voice_labels_raw` | Replaced by `INT_TO_EMOTION` dict with integer keys |
| `pd.get_dummies()` encoding | Not used in any descendant | Replaced by integer labels everywhere |

**Orphaned logic**: The `res_type='kaiser_fast'` parameter appears only in `v4.py` and was explicitly removed in the Kaggle retraining notebook with the comment "THE FIX: Removed res_type='kaiser_fast' which causes crashes on modern Kaggle environments."
