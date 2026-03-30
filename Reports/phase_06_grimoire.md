# Phase 6 Grimoire: Complete Technical Archaeology

---

## Header 1: Complete File Inventory

| File | Size (bytes) | Role |
|---|---|---|
| `FYP old/phase06_fusion_api_monolith.py` | 7,322 | Unified FastAPI: face + voice endpoints |
| `FYP old/phase06_fusion_api_old_endpoints.txt` | ~4,209 | Variant unified API with different model refs |
| `FYP old/phase06_vision_api_pyinstaller_variant.txt` | ~4,209 | Variant image API with PyInstaller CPU-only binding |
| `FYP old/phase06_fusion_api_json_variant.py` | 8,432 | Variant monolith with PyInstaller prep (`sys._MEIPASS`) |
| `FYP old/phase06_fusion_api_pyinstaller_spec.spec` | 930 | PyInstaller bundling specification |
| `FYP old/emotion_api/face_emotion_model.h5` | 7,475,760 | Phase 1 CNN (48×48) model weights |
| `FYP old/emotion_api/final_lstm_model_tf212.h5` | 1,756,512 | Voice LSTM TF2.12 compatible |
| `FYP old/emotion_api/final_lstm_model.h5` | 5,201,456 | Original voice LSTM (pre-TF2.12 fix) |
| `FYP old/emotion_api/final_lstm_model_tf212.keras` | 1,765,898 | Same model in .keras format |
| `FYP old/emotion_api/voice_model_tf212_FIXED.h5` | 1,756,448 | Another voice model fix iteration |

---

## Header 2: Line-by-Line Logic Migration

### File: `phase06_fusion_api_monolith.py` (Complete Source)

```python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
import librosa
from io import BytesIO
from tempfile import NamedTemporaryFile
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
FACE_MODEL_PATH = "face_emotion_model.h5"
VOICE_MODEL_PATH = "final_lstm_model_tf212.h5"

print("Loading face emotion model...")
face_model = load_model(FACE_MODEL_PATH)
print("✔ Face model loaded")

print("Loading voice emotion model...")
voice_model = load_model(VOICE_MODEL_PATH)
print("✔ Voice model loaded")

# Labels for face model
face_emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Labels for voice model
voice_labels_raw = {
    0: 'angry',
    1: 'disgust',
    2: 'fearful',
    3: 'happy',
    4: 'calm',
    5: 'neutral',
    6: 'sad',
    7: 'surprised'
}

# Merge calm + neutral
def merged_voice_label(idx):
    if idx == 4:
        return "neutral"
    return voice_labels_raw[idx]

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------
app = FastAPI(title="Unified Emotion Recognition API")


# ---------------------------------------------------------
# FACE EMOTION HELPERS
# ---------------------------------------------------------
def preprocess_face(face_gray):
    img = cv2.resize(face_gray, (48, 48))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img


def predict_face_emotion(face_gray):
    processed = preprocess_face(face_gray)
    raw = face_model.predict(processed, verbose=0)[0]
    scaled = raw * 0.55

    probs = {face_emotion_labels[i]: float(scaled[i]) for i in range(7)}
    top_idx = np.argmax(scaled)

    return face_emotion_labels[top_idx], float(scaled[top_idx]), probs


# ---------------------------------------------------------
# VOICE EMOTION HELPERS
# ---------------------------------------------------------
def extract_audio_features(audio_np, sr, max_len=174, n_mfcc=40):
    try:
        mfcc = librosa.feature.mfcc(y=audio_np, sr=sr, n_mfcc=n_mfcc)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        stacked = np.vstack([mfcc, delta, delta2])

        if stacked.shape[1] < max_len:
            stacked = np.pad(stacked, ((0,0),(0,max_len-stacked.shape[1])), mode='constant')
        else:
            stacked = stacked[:, :max_len]

        return stacked.T.reshape(1, max_len, 120)

    except Exception:
        return None


def predict_voice_emotion(audio_np, sr):
    features = extract_audio_features(audio_np, sr)
    if features is None:
        return None

    raw = voice_model.predict(features, verbose=0)[0]
    scaled = raw * 0.66  # Multiply probabilities by 0.66

    probs = {}
    for i, p in enumerate(scaled):
        label = merged_voice_label(i)
        probs[label] = probs.get(label, 0) + float(p)

    emotion = max(probs, key=probs.get)
    confidence = float(probs[emotion])

    return emotion, confidence, probs


# ---------------------------------------------------------
# IMAGE EMOTION ENDPOINT
# ---------------------------------------------------------
@app.post("/predict/image")
async def predict_from_image(file: UploadFile = File(...)):
    try:
        data = np.frombuffer(await file.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image"}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return {"error": "No face detected"}

        results = []
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            emotion, conf, probs = predict_face_emotion(face)

            results.append({
                "emotion": emotion,
                "confidence": conf,
                "probabilities": probs
            })

        return {"faces_detected": len(results), "results": results}

    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------
# VIDEO EMOTION ENDPOINT
# ---------------------------------------------------------
@app.post("/predict/video")
async def predict_from_video(file: UploadFile = File(...)):
    try:
        with NamedTemporaryFile(delete=True, suffix=".mp4") as temp:
            temp.write(await file.read())
            temp.flush()

            cap = cv2.VideoCapture(temp.name)
            frame_id = 0
            all_probs = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_id += 1
                if frame_id % 10 != 0:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) == 0:
                    continue

                (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                face = gray[y:y+h, x:x+w]

                _, _, probs = predict_face_emotion(face)
                all_probs.append(list(probs.values()))

            cap.release()

            if not all_probs:
                return {"error": "No face detected in video"}

            avg_probs = np.mean(np.array(all_probs), axis=0)

            result = {
                "final_emotion": face_emotion_labels[int(np.argmax(avg_probs))],
                "final_probabilities": {
                    face_emotion_labels[i]: float(avg_probs[i]) for i in range(7)
                }
            }

            return result

    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------
# VOICE EMOTION ENDPOINT
# ---------------------------------------------------------
@app.post("/predict/voice")
async def predict_from_voice(file: UploadFile = File(...)):
    try:
        audio_stream = BytesIO(await file.read())
        audio_np, sr = librosa.load(audio_stream, sr=22050)

        result = predict_voice_emotion(audio_np, sr)
        if result is None:
            return {"error": "Feature extraction failed"}

        emotion, confidence, probs = result

        return {
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": probs
        }

    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------
# RUN SERVER ON PORT 8000
# ---------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

#### Block 1: Dual Model Loading (Lines 15-25)
- **Both models load at startup into the same process**: `face_model` (~7.5MB) and `voice_model` (~1.7MB) share memory. If either crashes or leaks, both are affected.
- **`warnings.filterwarnings("ignore")`**: Suppresses ALL warnings globally. This hides TensorFlow deprecation warnings but also hides legitimate warnings like numerical instability or shape mismatches. Phase 9 replaces this with targeted `os.environ` variables.

#### Block 2: Voice Label Merge Function (Lines 31-46)
- **8-class voice labels with explicit calm→neutral merge**: `voice_labels_raw` has 8 entries (indices 0-7), including both `calm` (index 4) and `neutral` (index 5). The `merged_voice_label()` function maps index 4 to "neutral" at prediction time. This is different from Phase 2's training-time merge (where calm and neutral share the same training label).
- **Critical flaw**: The voice model `final_lstm_model_tf212.h5` may have been trained with 8 output classes (including separate calm), or 7 (with calm merged). If trained with 7 classes, the `voice_labels_raw` dict with 8 entries would cause an index-out-of-bounds error for index 7. The fact that this script ran suggests the model had 8 output nodes.
- **Phase 9 resolution**: The Kaggle retrained model uses exactly 7 output classes (`INT_TO_EMOTION` with indices 0-6), with calm merged during training, not at inference time.

#### Block 3: Voice Feature Extraction with Delta Stacking (Lines 82-97)
- **`mfcc + delta + delta2` stacking**: This is the ONLY place in the entire codebase where MFCC deltas (first and second order) are used. The stacked tensor has shape `(120, 174)` → reshaped to `(1, 174, 120)`.
- **`librosa.feature.delta(mfcc)`**: Computes the temporal rate of change of each MFCC coefficient. Delta2 is the rate of change of the rate of change (acceleration). These capture how the voice changes over time.
- **Phase 9 resolution**: Delta features were abandoned. The Phase 9 audio model uses MFCC-only features `(40, 174)` because the Bidirectional LSTM architecture captures temporal dynamics internally, making explicit delta computation redundant. This also halves the feature dimensionality, speeding up inference.

#### Block 4: Voice Sample Rate Bug (Line 215)
- **`librosa.load(audio_stream, sr=22050)`**: Forces the audio to 22050 Hz. But the voice model (`final_lstm_model_tf212.h5` and its successor `audio_best_model.keras`) was trained on 16000 Hz audio. This sample rate mismatch means the MFCC extraction produces features at the wrong time resolution — the model sees frequency patterns it was never trained on. This causes unpredictable misclassification.
- **Phase 7 fix**: `phase07_audio_api_standalone.txt` corrects this to `librosa.load(file_path, sr=SAMPLE_RATE)` with `SAMPLE_RATE = 16000`.

#### Block 5: Video Endpoint — Every-10th-Frame (Lines 157-202)
- **`if frame_id % 10 != 0: continue`**: Processes every 10th frame regardless of the video's FPS. For a 30fps video, this means 3 frames/second. For a 60fps video, 6 frames/second. For a 15fps video, 1.5 frames/second. The sampling rate is inconsistent across different video formats.
- **CPU waste**: Even though skipped frames aren't processed, `cap.read()` still decodes every frame. The CPU decompresses all frames, then throws away 90% of them.
- **`max(faces, key=lambda f: f[2] * f[3])`**: Selects the largest detected face by area (width × height). This is a reasonable heuristic — the largest face is likely the primary subject.
- **Per-frame `predict_face_emotion()`**: Each processed frame triggers a separate `model.predict()` call. No batch accumulation.
- **Phase 9 resolution**: `phase08_vision_api_preprod.py` uses `frame_mod = max(1, int(fps))` for FPS-aware decimation (exactly 1 frame/second), accumulates face tensors into a list, then batch-predicts with `compute_vision_inference(np.stack(face_tensors))`.

#### Block 6: Probability Scaling (Lines 70, 106)
- **Face: `raw * 0.55`**: Reduces all face probabilities by 45%.
- **Voice: `raw * 0.66`**: Reduces all voice probabilities by 34%.
- **Different scaling factors for different models**: No documented rationale. These were likely tuned empirically during demos to make confidence scores "feel right" to the developer.
- **Phase 9 resolution**: All arbitrary scaling removed. Raw softmax probabilities are returned. Phase 9's text model applies `predictions ** 1.5` (power sharpening) which is mathematically principled — it increases the gap between the top and second predictions, making the model's "opinion" stronger.

#### Block 7: PyInstaller Spec (`phase06_fusion_api_pyinstaller_spec.spec`) & Variant Monoliths (`phase06_fusion_api_json_variant.py` / `phase06_vision_api_pyinstaller_variant.txt`)

In `phase06_fusion_api_json_variant.py` and `phase06_vision_api_pyinstaller_variant.txt`, we see explicit runtime code supporting the `.spec` file:

```python
if getattr(sys, 'frozen', False):
    import pyi_splash
    pyi_splash.close()
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

cascade_path = os.path.join(application_path, "haarcascade_frontalface_default.xml")
```

- **`sys._MEIPASS`**: This is PyInstaller's temporary unpacking directory. When a user runs the `.exe`, PyInstaller extracts `haarcascade_frontalface_default.xml` and the `.h5` files here. These scripts were modified to load assets from this dynamic path instead of the current working directory.
- **CPU Binding**: `phase06_vision_api_pyinstaller_variant.txt` explicitly used `os.environ["CUDA_VISIBLE_DEVICES"] = "-1"` and `os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"` to force CPU execution, likely because bundling CUDA native libraries via PyInstaller was failing.
- **`phase06_fusion_api_pyinstaller_spec.spec`**: 
```python
a = Analysis(
    ['phase06_fusion_api_monolith.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('face_emotion_model.h5', '.'),
        ('final_lstm_model_tf212.h5', '.'),
    ],
    hiddenimports=[
        'tensorflow',
        'tensorflow.keras',
        'cv2',
        'librosa',
        'numpy',
        'soundfile'
    ],
    ...
)
exe = EXE(
    ...
    name='emotion_api',
    ...
    upx=True,
    console=True,
    ...
)
```

- **`datas` embedding**: Model files are bundled as binary data inside the `.exe`. At runtime, they are extracted to a temp directory (`sys._MEIPASS`).
- **`hiddenimports`**: TensorFlow's dynamic module loading means many submodules are not statically detectable by PyInstaller. This list would need to be MUCH longer for TF to work (hundreds of hidden imports).
- **`upx=True`**: Attempts to compress the executable with UPX. TensorFlow's native binaries are not UPX-compatible, likely causing corruption.
- **Why this was abandoned**: TensorFlow + PyInstaller is notoriously difficult. The resulting `.exe` would be >500MB, take minutes to start (extracting temp files), and frequently crash due to missing native library references.

---

## Header 3: Micro-Decision Log

| Decision | Phase 6 Value | Phase 9 Value | Rationale |
|---|---|---|---|
| Architecture | Single monolithic FastAPI | 4 separate microservices | Isolation, independent scaling, fault containment |
| Face model | `face_emotion_model.h5` (48×48, 57%) | `fer_best_model.keras` (112×112, 81%) | Retrained on FERPlus with CLAHE, augmentation |
| Face detection | Haar Cascade | MediaPipe Tasks Vision API (0.75 confidence) | Eliminates false positives |
| Voice sample rate | 22050 Hz (`librosa.load(sr=22050)`) | 16000 Hz (explicit) | Matches training data |
| Voice features | MFCC + delta + delta2 (120 dims) | MFCC only (40 dims) | BiLSTM captures dynamics internally |
| Probability scaling | 0.55 (face), 0.66 (voice) | None (raw softmax) | Arbitrary scaling removed |
| Voice label handling | 8-class with runtime calm→neutral merge | 7-class (merged at training time) | Eliminates runtime label patching |
| Video frame sampling | Every 10th frame (FPS-ignorant) | 1 frame/second (FPS-aware) | Consistent analysis regardless of source FPS |
| Video prediction | Per-frame `model.predict()` | Batch `compute_vision_inference(stack)` | Massive latency reduction |
| Deployment | PyInstaller `.exe` (abandoned) | Uvicorn `--workers 4` via `.bat` launcher | Proper ASGI deployment |
| CORS | Not configured | CORSMiddleware (Phase 7 adds it) | Required for Flutter web/mobile |
| Error handling | Generic `return {"error": str(e)}` | `HTTPException` with status codes | Proper HTTP semantics |

---

## Header 4: Inter-Phase Diff Analysis

| Phase 6 Artifact | Immediate Descendant (Phase 7) | Phase 9 Descendant |
|---|---|---|
| `phase06_fusion_api_monolith.py` `/predict/image` | `phase07_vision_api_standalone.txt` `/predict/image` | `phase08_vision_api_preprod.py` `/predict/image` |
| `phase06_fusion_api_monolith.py` `/predict/video` | `phase07_vision_api_standalone.txt` `/predict/video` | `phase08_vision_api_preprod.py` `/predict/video` |
| `phase06_fusion_api_monolith.py` `/predict/voice` | `phase07_audio_api_standalone.txt` `/predict_audio` | `phase08_audio_api_preprod.py` `/predict_audio` |
| No text endpoint | `phase07_text_api_standalone.txt` `/predict_text` | `phase08_text_api_preprod.py` `/predict_text` |
| No orchestrator | None | `orchestrator_v3.py` (fusion, LLM, auth, DB) |
| `phase06_fusion_api_pyinstaller_spec.spec` (PyInstaller) | Abandoned | `start_servers.bat` (proper deployment) |
| `merged_voice_label()` | Removed (7-class model) | `INT_TO_EMOTION` dict |
| `extract_audio_features()` with deltas | `process_audio()` (MFCC only) | `get_features_fast()` (MFCC + Z-score) |

### Files that disappeared
- `phase06_fusion_api_pyinstaller_spec.spec`: PyInstaller approach abandoned entirely.
- `emotion_api/final_lstm_model.h5`: Pre-TF2.12 model, replaced by TF2.12-compatible version.
- `emotion_api/voice_model_tf212_FIXED.h5`: Another iteration of the voice model fix, superseded.
- `JSONResponse` import: Imported but never used in `phase06_fusion_api_monolith.py`. Dead import removed in Phase 7.
