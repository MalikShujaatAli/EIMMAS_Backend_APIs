# Phase 7 Grimoire: Complete Technical Archaeology

---

## Header 1: Complete File Inventory

| File | Size (bytes) | Role |
|---|---|---|
| `FYP old/phase07_audio_api_standalone.txt` | 7,678 | Standalone Audio API |
| `FYP old/phase07_vision_api_standalone.txt` | 8,737 | Standalone Image+Video API |
| `FYP old/phase07_text_api_standalone.txt` | 6,038 | Standalone Text API |

---

## Header 2: Line-by-Line Logic Migration

### File: `phase07_audio_api_standalone.txt` — Key Diff from Phase 6

The full source (202 lines) is preserved in this project. Below are the critical blocks that changed from Phase 6.

#### New: CORS Middleware (Lines 41-48)
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
- **What this solves**: Flutter mobile apps and web browsers require CORS headers. Without this middleware, every Flutter HTTP request would be rejected by the browser's same-origin policy.
- **`allow_origins=["*"]`**: Allows ALL origins. Acceptable for development, dangerous in production (allows any website to call the API). Phase 9 retains this for development convenience but notes it should be restricted in production.

#### New: Auto-Rename Zip Fix (Lines 71-77)
```python
if os.path.exists(ZIPPED_MODEL_PATH) and not os.path.exists(MODEL_PATH):
    try:
        os.rename(ZIPPED_MODEL_PATH, MODEL_PATH)
```
- **What this solves**: When model files were shared via WhatsApp or Google Drive, they were sometimes auto-compressed with a `.zip` extension appended (e.g., `audio_best_model.keras.zip`). This auto-rename silently fixes the filename at startup.
- **Phase 9**: This workaround was removed because model distribution moved to proper file sharing without auto-compression.

#### New: Silence Trimming in API Context (Lines 99-103)
```python
audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
if len(audio_trimmed) == 0:
     return None
```
- **First appearance in API code**: `librosa.effects.trim` with `top_db=30` dB threshold. Silence below 30 dB is stripped. This was already used in the Kaggle training notebook but had not appeared in any previous API code.
- **Edge case handling**: If the entire file is silent (`len(audio_trimmed) == 0`), returns `None` instead of crashing on empty array MFCC extraction.

#### New: File Size Guard (Lines 148-150)
```python
file_size_mb = len(content) / (1024 * 1024)
if file_size_mb > MAX_FILE_SIZE_MB:
    raise HTTPException(status_code=413, detail=...)
```
- **`MAX_FILE_SIZE_MB = 10`**: Rejects files over 10MB. Phase 9's audio API increases this to 50MB. Phase 9's orchestrator uses 250MB for video files.
- **HTTP 413**: Proper "Payload Too Large" status code, not a generic 400.

#### New: `finally` Cleanup Block (Lines 191-195)
```python
finally:
    if temp_audio_path and os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
```
- **What this solves**: Guarantees temporary file deletion even if processing crashes. Without `finally`, a crash during `model.predict()` would leave orphaned `.wav` files on disk.
- **Phase 9 resolution**: Temp files eliminated entirely via FFmpeg RAM pipes and `soundfile.read()` from byte buffers.

#### Persistent Flaw: Synchronous `model.predict` (Line 164)
```python
prediction = model.predict(tensor, verbose=0)[0]
```
- Still runs on the async event loop thread. Still blocks all concurrent requests during inference.

---

### File: `phase07_vision_api_standalone.txt` — Key Diff from Phase 6

#### New: MediaPipe Face Detection (Lines 65-66)
```python
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
```
- **Replaces**: `face_cascade = cv2.CascadeClassifier(haarcascade...)` from Phase 6.
- **`model_selection=1`**: Full-range model (works at various distances). `model_selection=0` would be short-range only.
- **`min_detection_confidence=0.6`**: Still too low. Phase 9 increases to 0.75 after discovering that 0.6 admitted some non-human objects.
- **API difference**: This uses `mp.solutions.face_detection` (MediaPipe's legacy Python solution API). Phase 9 uses `mediapipe.tasks.vision.FaceDetector` (the newer Tasks API) with a dedicated `.tflite` model file.

#### New: CLAHE Lighting Correction (Lines 68-69)
```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
```
- **First appearance**: Contrast Limited Adaptive Histogram Equalization. Divides the image into 8×8 tiles and normalizes the histogram within each tile independently, with a clip limit of 2.0 to prevent noise amplification.
- **Applied in `preprocess_face()`**: `face_clahe = clahe.apply(face_gray)` before resizing and normalization.
- **Matches training pipeline**: The Kaggle FERPlus training notebook applies identical CLAHE parameters during data preprocessing, ensuring consistency between training and inference.

#### New: 112×112 Resolution (Line 87)
```python
face_resized = cv2.resize(face_clahe, (112, 112), interpolation=cv2.INTER_CUBIC)
```
- **Replaces**: Phase 6's `cv2.resize(gray_face, (48, 48))`.
- **`cv2.INTER_CUBIC`**: Bicubic interpolation preserves sharper edges than the default bilinear interpolation. Matching the training pipeline.

#### New: `fer_best_model.keras` (Line 39)
```python
FACE_MODEL_PATH = "fer_best_model.keras"
```
- **Replaces**: Phase 6's `face_emotion_model.h5` (48×48, 57% accuracy).
- **New model**: FERPlus-trained, 112×112, 4-block CNN with BatchNormalization, 81.03% accuracy.

#### Persistent Flaw: Fixed Frame Decimation (Lines 200-201)
```python
frame_id += 1
if frame_id % 10 != 0:
    continue
```
- Still samples every 10th frame regardless of source FPS. A 60fps video gets 6 samples/second; a 24fps video gets 2.4 samples/second.
- Phase 9 fix: `frame_mod = max(1, int(fps))` ensures exactly 1 frame/second.

#### Persistent Flaw: Per-Frame Prediction (Line 212)
```python
_, _, probs = predict_face_emotion(face_roi)
```
- Still calls `model.predict()` for each individual frame. No tensor accumulation, no batching.

---

### File: `phase07_text_api_standalone.txt` — Key Diff from Phase 4

#### New: `AttentionLayer` with Keras Serialization (Lines 34-54)
```python
@tf.keras.utils.register_keras_serializable()
class AttentionLayer(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)
```
- **`@tf.keras.utils.register_keras_serializable()`**: Enables the layer to be saved/loaded with `model.save()` / `load_model()` without manual `custom_objects` registration. This was missing in `phase04_text_attention_tester.py`.
- **Weight shape change**: From `(dim,)` in `phase04_text_attention_tester.py` to `(input_shape[-1], 1)` — a column vector. This changes the attention from a dot product with a 1D vector to a matrix multiplication yielding scalar scores.
- **`initializer="normal"`**: Changed from `glorot_uniform` in `phase04_text_attention_tester.py`. Normal initialization has higher variance, leading to more diverse initial attention scores.
- **`K.tanh(K.dot(x, self.W) + self.b)`**: Bahdanau-style additive attention with tanh activation. Replaces `phase04_text_attention_tester.py`'s `tf.tensordot` approach.
- **Phase 9 evolution**: `K.tanh` → `keras.ops.tanh`, `K.dot` → `ops.matmul`, `K.softmax` → `ops.softmax`, `K.sum` → `ops.sum`. The Keras 2 backend ops (`K.*`) were deprecated in Keras 3.

#### New: Batch Inference (Lines 99-103)
```python
sequences = tokenizer.texts_to_sequences(sentences)
padded_batch = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
predictions = model.predict(padded_batch, verbose=0)
```
- **First appearance of batch prediction for text**: All sentences are tokenized together, padded into a single matrix, and predicted in one `model.predict()` call. This eliminates the per-sentence prediction loop from `phase04_text_negation_engine.py`.
- **Still uses `model.predict()`**: Not `@tf.function` compiled. Not offloaded with `asyncio.to_thread`.

#### Persistent Flaw: NLTK Download at Startup (Line 18)
```python
nltk.download('punkt', quiet=True)
```
- Fires on every server boot. If NLTK's CDN is down or the server has no internet, this either blocks startup or silently fails (leaving `sent_tokenize` broken).
- Phase 9 fix: `setup_nltk.py` downloads during build/setup, not at runtime.

---

## Header 3: Micro-Decision Log

| Decision | Phase 6 | Phase 7 | Phase 9 |
|---|---|---|---|
| Face detector | Haar Cascade | MediaPipe `mp.solutions` (0.6) | MediaPipe Tasks API (0.75) |
| Face resolution | 48×48 | 112×112 + CLAHE | 112×112 + CLAHE |
| Face model | `face_emotion_model.h5` (57%) | `fer_best_model.keras` (81%) | `fer_best_model.keras` (81%) |
| Audio sample rate | 22050 Hz | 16000 Hz (fixed!) | 16000 Hz |
| Audio silence trim | None | `librosa.effects.trim(top_db=30)` | `librosa.effects.trim(top_db=30)` |
| Audio normalization | None | None (still missing!) | Z-score `(mfccs-mean)/std` |
| Audio file size limit | None | 10 MB | 50 MB |
| Audio disk I/O | `NamedTemporaryFile` | `NamedTemporaryFile` + `finally` cleanup | Zero disk (soundfile in-memory) |
| Text inference | Per-sentence loop | Batch `model.predict(padded_batch)` | Batch `@tf.function` + `asyncio.to_thread` |
| Text negation | `rewrite_sentence()` (Phase 4) | Removed | Removed |
| Text context filter | `is_context_clear()` (Phase 4) | Removed | Removed (dual-threshold instead) |
| NLTK download | N/A | `nltk.download()` at startup | `setup_nltk.py` (offline) |
| Attention ops | N/A (`phase04_text_bilstm_trainer.py` had no attention) | `K.tanh`/`K.dot`/`K.softmax` | `ops.tanh`/`ops.matmul`/`ops.softmax` |
| CORS | Not configured | CORSMiddleware (audio only) | CORSMiddleware (all services) |
| Video decimation | Every 10th frame | Every 10th frame (unchanged!) | FPS-aware (1 frame/sec) |
| Async offloading | None | None (still blocking!) | `asyncio.to_thread()` everywhere |

---

## Header 4: Inter-Phase Diff Analysis

### What Phase 7 fixed from Phase 6
1. Haar Cascades → MediaPipe (face detection accuracy)
2. 48×48 CNN → 112×112 FERPlus CNN (model accuracy: 57% → 81%)
3. CLAHE lighting correction introduced
4. Audio sample rate: 22050 → 16000 Hz
5. Audio silence trimming introduced
6. File size guards introduced
7. CORS middleware introduced
8. Text negation engine removed (trust the model)
9. Text batch inference introduced
10. Proper `finally` cleanup for temp files

### What Phase 7 did NOT fix (left to Phase 9)
1. **`asyncio.to_thread()`**: All `model.predict()` calls still block the event loop.
2. **`@tf.function` compilation**: No graph compilation, still eager execution.
3. **Video FPS-aware decimation**: Still `frame_id % 10`.
4. **Video batch prediction**: Still per-frame `model.predict()`.
5. **Audio Z-score normalization**: MFCCs still unnormalized. "False Angry" bias persists.
6. **Audio TFLite**: Still using full Keras model, not TFLite.
7. **Audio disk I/O**: Still writes temp `.wav` files to disk.
8. **NLTK runtime download**: Still fires at startup.
9. **No orchestrator**: Three separate APIs with no coordination layer.
10. **No LLM integration**: No Groq, no therapeutic conversation.
11. **No authentication**: No JWT, no user isolation.
12. **No database**: No chat history persistence.
13. **No crisis/abuse detection**: No pre-flight safety gate.
14. **No contradiction engine**: No affective masking detection.
