# Phase 9 Grimoire: Complete Technical Archaeology

---

## Header 1: Complete File Inventory

| File | Location | Role |
|---|---|---|
| `main_audio.py` | `services/audio_api/` | Production Audio Emotion API |
| `convert_audio_model.py` | `services/audio_api/` | Keras → TFLite converter |
| `main_video.py` | `services/image_video_api/` | Production Vision Emotion API |
| `main_text.py` | `services/text_api/` | Production Text Emotion API |
| `orchestrator_v3.py` | `services/fusion_api/` | Master Orchestrator: fusion, LLM, auth, DB |
| `database.py` | `services/fusion_api/` | Async SQLAlchemy ORM models |
| `start_servers.bat` | `launch_scripts/` | Multi-service Windows launcher |
| `setup_nltk.py` | `launch_scripts/` | Offline NLTK data installer |

---

## Header 2: Line-by-Line Logic Migration

Due to the scale of Phase 9 files (235+ lines each, orchestrator at 651 lines), this grimoire documents the **specific code blocks that represent evolutionary leaps from all previous phases**, with cross-references to the exact Phase/file they replace.

---

### Service 1: `main_audio.py` — 8 Evolutionary Leaps

#### Leap 1: TFLite Primary Inference (Replaces Phase 2-7 `model.predict`)
```python
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def _predict_tflite(features):
    interpreter.set_tensor(input_details[0]['index'], features.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]
```
- **What this replaces**: Every `model.predict(tensor, verbose=0)[0]` call from Phases 3-7.
- **Why TFLite**: The audio BiLSTM had 2.8+ second inference latency on CPU via full Keras. TFLite's flatbuffer execution engine with float16 quantization reduced this to sub-second.
- **Fallback**: If the `.tflite` file is missing, the code falls back to loading the full `.keras` model. This dual-path design prevents deployment failures.

#### Leap 2: `soundfile.read()` In-Memory (Replaces Phase 7 `NamedTemporaryFile`)
```python
import soundfile as sf

audio_data, sr = sf.read(io.BytesIO(file_bytes))
```
- **What this replaces**: `NamedTemporaryFile(delete=False, suffix=".wav")` → `librosa.load(temp_path)` → `os.remove(temp_path)` from Phase 7.
- **Zero disk I/O**: Audio bytes go directly from HTTP upload → BytesIO buffer → soundfile → numpy array. No file touch the SSD at any point.
- **Sample rate**: `soundfile.read()` returns audio at its native sample rate. The audio is then validated/resampled to 16kHz if needed.

#### Leap 3: Z-Score MFCC Normalization (Replaces Phase 2-7 raw MFCCs)
```python
mfccs = librosa.feature.mfcc(y=audio_trimmed, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
mean = np.mean(mfccs)
std = np.std(mfccs) + 1e-9  # epsilon prevents division by zero
mfccs = (mfccs - mean) / std
```
- **What this replaces**: Raw MFCC values used in `v4.py`, `v5.py`, `v6.py`, `emotion_api/main.py`, `2nd attempt Audio.txt`.
- **Why**: Raw MFCC magnitudes vary with recording volume. A loud voice produces higher absolute values regardless of emotion. Z-score normalization centers the data at mean=0, std=1, forcing the model to focus on relative tonal patterns instead of absolute volume. This eliminated the "False Angry" bias.
- **`+ 1e-9`**: Epsilon guard against division by zero for completely silent audio (where std=0).

#### Leap 4: `asyncio.to_thread()` (Replaces Phase 7 synchronous blocking)
```python
prediction = await asyncio.to_thread(_predict_tflite, features)
```
- **What this replaces**: Direct `model.predict()` / `interpreter.invoke()` calls on the async event loop thread.
- **Why**: FastAPI's event loop must remain unblocked to accept concurrent connections. `asyncio.to_thread()` moves the CPU-intensive TFLite inference to a background thread from the default thread pool executor, freeing the event loop to handle other requests simultaneously.

#### Leap 5: File Size Guard (Replaces Phase 7's 10MB limit)
```python
MAX_FILE_SIZE_MB = 50
```
- **Increased from 10MB to 50MB**: Mobile recordings can be larger than initially expected, especially for longer therapy sessions.

#### Leap 6: Centralized Structured Logging
```python
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'audio_api.log'),
    ...
)
```
- **What this replaces**: Phase 7's `logging.basicConfig(filename="audio_api.log")` which wrote logs to the script's working directory.
- **Centralized**: All four services log to a shared `logs/` directory at the project root, enabling unified monitoring.

---

### Service 2: `main_video.py` — 6 Evolutionary Leaps

#### Leap 1: MediaPipe Tasks Vision API (Replaces Phase 7 `mp.solutions`)
```python
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

options = vision.FaceDetectorOptions(
    base_options=python.BaseOptions(model_asset_path=MP_MODEL_PATH),
    min_detection_confidence=0.75,
    min_suppression_threshold=0.3
)
detector = vision.FaceDetector.create_from_options(options)
```
- **What this replaces**: `mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.6)` from Phase 7.
- **Confidence increase 0.6 → 0.75**: The Model notebooks document this tuning: 0.85 was too strict (rejected humans in dark rooms), 0.50 was too weak (admitted animals), 0.75 was "the sweet spot."
- **Tasks API vs Solutions API**: The Tasks API uses a dedicated `.tflite` model file (`blaze_face_short_range.tflite`) and provides structured detection results. The Solutions API used the mediapipe framework internally.

#### Leap 2: `@tf.function` Compilation with Warmup
```python
@tf.function(reduce_retracing=True)
def compute_vision_inference(tensor_input):
    return emotion_model(tensor_input, training=False)

# Warmup on boot
_ = compute_vision_inference(np.zeros((1, 112, 112, 1), dtype='float32'))
```
- **What this replaces**: Every `face_model.predict(processed, verbose=0)` from Phases 3-7.
- **`reduce_retracing=True`**: Prevents TensorFlow from recompiling the graph when input shapes vary slightly. The graph is compiled once and reused.
- **Warmup**: The first `@tf.function` call triggers graph tracing and XLA compilation. By calling with a dummy zero tensor during server boot, the compilation latency is absorbed at startup rather than on the first user request.
- **`training=False`**: Disables dropout and batch normalization running statistics updates. Critical for inference — using `training=True` would produce different results each time.

#### Leap 3: FPS-Aware Frame Decimation (Replaces Phase 6-7 every-10th-frame)
```python
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_mod = max(1, int(fps))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % frame_mod == 0:
        # Process this frame
    frame_idx += 1
```
- **What this replaces**: `if frame_id % 10 != 0: continue` from Phases 6-7.
- **Why**: `frame_mod = int(fps)` means every FPS-th frame is selected, yielding exactly 1 frame per second regardless of source format. A 30fps video produces 30 frames/second → `frame_mod=30` → one frame analyzed per second. A 60fps video → `frame_mod=60` → still one frame per second. Consistent analysis density.

#### Leap 4: Batch Tensor Stacking (Replaces per-frame prediction)
```python
face_tensors = []
for ... :  # per-frame face extraction loop
    face_tensors.append(preprocessed_face)

if face_tensors:
    batch = np.stack(face_tensors, axis=0)  # (N, 112, 112, 1)
    all_preds = compute_vision_inference(batch).numpy()
```
- **What this replaces**: Per-frame `predict_face_emotion(face_roi)` calls from Phases 6-7.
- **Why**: TensorFlow's `@tf.function` processes batches with near-zero marginal cost per additional item. Processing 30 faces in one call is ~30x faster than 30 individual calls because the graph overhead (Python→C++ bridge, memory allocation, kernel launch) is incurred only once.

#### Leap 5: UUID-Based Temp Files with `finally` Cleanup
```python
temp_path = os.path.join(tempfile.gettempdir(), f"eimmas_{uuid.uuid4().hex}.mp4")
try:
    with open(temp_path, 'wb') as f:
        f.write(file_bytes)
    # ... process video ...
finally:
    if os.path.exists(temp_path):
        os.remove(temp_path)
```
- **What this replaces**: Phase 7's `NamedTemporaryFile(delete=True)` which has race conditions on Windows (the file may be deleted before OpenCV finishes reading it).
- **UUID naming**: `eimmas_{uuid4}` prevents filename collisions when multiple requests arrive simultaneously.

---

### Service 3: `main_text.py` — 5 Evolutionary Leaps

#### Leap 1: Keras 3 Ops (Replaces Phase 7 `K.*` backend ops)
```python
from keras import ops

class AttentionLayer(Layer):
    def call(self, x):
        e = ops.tanh(ops.matmul(x, self.W) + self.b)
        a = ops.softmax(e, axis=1)
        output = x * a
        return ops.sum(output, axis=1)
```
- **What this replaces**: `K.tanh(K.dot(x, self.W) + self.b)` from Phase 7's `2nd attempt Text.txt`.
- **Why**: `tensorflow.keras.backend` (`K`) was deprecated in Keras 3. The `keras.ops` module provides the same mathematical operations with backend-agnostic execution (supports JAX, PyTorch, TensorFlow).

#### Leap 2: Pre-Compiled Regex (Replaces runtime compilation)
```python
URL_PATTERN = re.compile(r'http\S+|www\.\S+')
MENTION_PATTERN = re.compile(r'@\w+')
HASHTAG_PATTERN = re.compile(r'#')
SPECIAL_CHAR_PATTERN = re.compile(r'[^a-zA-Z\s]')
WHITESPACE_PATTERN = re.compile(r'\s+')
```
- **What this replaces**: Implicit runtime regex compilation in `textemotion_tf212.py`'s `rewrite_sentence()` and `is_context_clear()`.
- **Why**: `re.compile()` at module scope compiles regex patterns once during import. Without pre-compilation, Python recompiles the regex pattern on every function call — measurable overhead when processing thousands of requests.

#### Leap 3: `@tf.function` + Batch Inference + Thread Offloading
```python
@tf.function(reduce_retracing=True)
def compute_inference(padded_input):
    return text_model(padded_input, training=False)

# In endpoint:
def _run_model():
    return compute_inference(padded_seqs).numpy()
predictions = await asyncio.to_thread(_run_model)
```
- **What this replaces**: `model.predict(padded_batch, verbose=0)` from Phase 7.
- **Triple optimization**: (1) `@tf.function` compiles to C++ graph (2) all sentences batched into single tensor (3) inference offloaded to background thread via `asyncio.to_thread`.

#### Leap 4: Prediction Confidence Sharpening
```python
predictions = predictions ** 1.5
predictions = predictions / predictions.sum(axis=1, keepdims=True)
```
- **What this replaces**: No equivalent in any previous phase.
- **Why**: Raw softmax outputs often produce "mushy" distributions (e.g., 0.25, 0.20, 0.18, 0.15, 0.12, 0.10). Raising to power 1.5 amplifies differences: (0.25^1.5=0.125, 0.10^1.5=0.032), then re-normalizing produces a sharper distribution with a clearer winner.

#### Leap 5: Dual-Threshold Confidence Filter
```python
if max_prob < 0.50 or (max_prob - second_prob) < 0.15:
    sentence_emotion = "context unclear"
```
- **What this replaces**: Phase 4's `is_context_clear()` keyword-based filter AND Phase 7's single `CONFIDENCE_THRESHOLD = 0.40`.
- **Two conditions**: (1) Top probability must exceed 50% (the model must be at least half-sure). (2) Gap between top and second must exceed 15% (the model must not be torn between two emotions). Both conditions must be met. This is more principled than keyword matching.

---

### Service 4: `orchestrator_v3.py` — 10 Evolutionary Leaps (No Previous Equivalent)

The orchestrator has NO predecessor in any previous phase. Every feature below is Phase 9-original.

#### Leap 1: Groq LLM Integration
```python
from groq import AsyncGroq
client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
response = await client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=conversation_history,
    temperature=0.75,
    max_completion_tokens=500,
)
```
- **Model**: Llama-3.3-70B via Groq's inference API. Chosen for its speed (Groq's custom LPU hardware) and quality (70B parameter model for nuanced therapeutic responses).
- **System prompt**: A multi-paragraph prompt defining "Eimma" — an empathetic psychologist who acknowledges detected emotions, uses clinical techniques (CBT, DBT, motivational interviewing), never diagnoses, never prescribes, and always responds in English.

#### Leap 2: JWT Authentication
```python
from jose import jwt, JWTError
credentials = await security(request)
payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"],
                     audience=EXPECTED_AUDIENCE, issuer=EXPECTED_ISSUER)
user_email = payload.get("email")
```
- **HS256 symmetric signing**: The JWT is validated for algorithm, audience, issuer, and expiration. The `email` claim is extracted and used to scope database queries.

#### Leap 3: FFmpeg RAM-Pipe Audio Extraction
```python
def _extract_audio_from_video(file_bytes):
    cmd = [ffmpeg_exe, "-y", "-i", "pipe:0", "-vn",
           "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
           "-f", "wav", "pipe:1"]
    result = subprocess.run(cmd, input=file_bytes,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return result.stdout
```
- **`pipe:0` / `pipe:1`**: Video bytes are streamed into FFmpeg via stdin, and the extracted WAV audio is streamed back via stdout. Zero files touch the disk. The `-ar 16000 -ac 1` flags ensure 16kHz mono PCM output matching the audio model's training format.

#### Leap 4: Weighted Multimodal Fusion
```python
WEIGHT_TEXT = 0.50
WEIGHT_VISUAL = 0.35
WEIGHT_AUDIO = 0.15
```
- **Text-dominant**: Text carries the highest weight because natural language is the most semantically rich channel for emotional expression. Visual cues are second because facial expressions are culturally universal. Audio is lowest because MFCC-based emotion detection has the highest error rate of the three modalities.

#### Leap 5: Contradiction Engine (Affective Masking)
```python
text_negative = text_emo in ["sadness", "anger", "fear", "sad", "angry", "fearful"]
visual_positive = face_emo in ["happy", "surprise"]
if text_negative and visual_positive:
    contradiction_flag = "masked_distress"
```
- **Clinical purpose**: Detects when a person's words express pain but their face shows a smile — a clinically documented phenomenon called "affective masking." The `masked_distress` flag is injected into the LLM's context, instructing Eimma to gently acknowledge the discrepancy.

#### Leap 6: Pre-Flight Crisis/Abuse Regex Gate
```python
CRISIS_PATTERNS = re.compile(r'\b(kill myself|suicide|end my life|...)\b', re.IGNORECASE)
ABUSIVE_PATTERNS = re.compile(r'\b(bsdk|bc|mc|loru|...)\b', re.IGNORECASE)
```
- **Before any ML**: These regex patterns scan the raw input text before it reaches the neural networks. If crisis language is detected, the system returns Pakistan-specific helpline information (Umang: 0317-4288665) and skips ML processing. If abusive language is detected, the system redirects to a boundary-setting response.

#### Leap 7: HTTPX Global Connection Pool
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        timeout=httpx.Timeout(60.0)
    )
    yield
    await app.state.http_client.aclose()
```
- **What this replaces**: Creating a new `httpx.AsyncClient()` per request (which creates a new TCP connection per request).
- **Persistent connections**: The pool keeps up to 20 TCP connections alive between requests to the sub-APIs (audio, video, text). Eliminates TLS handshake overhead (~150-300ms per connection establishment).

#### Leap 8: BackgroundTasks for Database Saves
```python
background_tasks.add_task(save_messages_to_db, session_id, user_email, user_text, ai_response, detected_emotion)
return JSONResponse(content=response_data)  # Returns IMMEDIATELY
```
- **Non-blocking saves**: The HTTP response is sent to the Flutter client before the database write completes. The `save_messages_to_db` function runs asynchronously in the background.

#### Leap 9: Bulk Delete (N+1 Fix)
```python
from sqlalchemy import delete
stmt = delete(ChatMessage).where(ChatMessage.session_id == session_id)
await db.execute(stmt)
```
- **What this replaces**: `for msg in messages: await db.delete(msg)` which would execute N individual DELETE queries.

#### Leap 10: `show_emotion_ui` Flag
```python
response_data["show_emotion_ui"] = True if detected_emotion else False
```
- **Flutter coordination**: Tells the mobile app whether to display the emotion visualization panel. If no emotion was detected (e.g., the user sent a greeting without multimedia), the Flutter UI can hide the emotion widgets.

---

## Header 2.5: Orchestrator Evolution (v0 → v1 → v2 → v3)

### What v0 had (314 lines):
- Groq LLM only (no Cerebras fallback)
- `llama-3.1-8b-instant` (small model, fast but less nuanced)
- Hardcoded API key: `gsk_wUyI0u...` directly in source
- `WEIGHTS = {visual: 0.45, audio: 0.35, text: 0.20}` — visual-dominant
- Simple JWT: `jwt.decode(token, SECRET_KEY, algorithms=["HS256"])` — no issuer, no audience
- 4-sentence system prompt: "validate, reflect, inquire, keep it short"
- `NamedTemporaryFile` FFmpeg with disk I/O
- `asyncio.create_subprocess_exec("ffmpeg", ...)` with `afftdn=nf=-25,loudnorm` filters
- Single `emotion_map` for normalization
- `httpx.AsyncClient()` created fresh per request — no connection pool
- **Bug**: Session title set AFTER `db.add(new_session)` but BEFORE `await db.commit()`, and the `db.add` / `await db.commit()` are incorrectly indented (always execute, not just when `if not session_id`)
- `uvicorn.run("orchestrator_v2:app")` — filename mismatch (copy-paste error)

### What v1 added (528 lines, +214):
- **Cerebras dual-engine**: `USE_PRODUCTION_MODEL` toggle — Groq 70B for presentation, Cerebras 8B for daily testing
- **Weights rebalanced**: `{visual: 0.35, audio: 0.15, text: 0.50}` — text-dominant
- **File logging**: `logging.FileHandler(LOG_FILE, encoding="utf-8")` — fixes Windows CP1252 crash on emojis
- **Comprehensive JWT**: Searches `email`, `unique_name`, and two XML schema claim paths
- **Issuer/audience validation**: `issuer=VALID_ISSUER, audience=VALID_AUDIENCE`
- **5-mode system prompt** (340 lines): Standard Counseling / Crisis Protocol / Boundary Enforcement / Scope Enforcement / Language Barrier
- **Extended `emotion_map`**: Preserves `love` as a class
- **Confidence differentiation**: Text uses `weighted_probabilities`, audio/video use `confidence/100`
- **LLM fallback responses**: Hardcoded per-emotion fallback dict when API fails
- **DELETE `/sessions/{id}`** endpoint: Includes N+1 delete loop (`for msg in messages: await db.delete(msg)`)
- **FFmpeg simplified**: Removed `afftdn=nf=-25,loudnorm` filters that were corrupting natural voice patterns

### What v2 added (617 lines, +89):
- **Crisis/abuse regex gate**: 6 crisis patterns with `\b` word boundaries and `.*?` wildcards, 2 abuse patterns with Desi slang
- **Contradiction engine**: Detects `masked_distress` when text is negative but visual/audio is positive
- **Whisper hallucination cleaner**: `re.sub(r'\[.*?\]|\(.*?\)', '', text)` strips `[Silence]`, `(Music)` artifacts
- **Ghost Gate**: Rejects requests where no text AND no face/voice were detected
- **`show_emotion_ui`**: Flutter coordination flag
- **ISO timestamp**: `.isoformat() + "Z"` for Flutter `DateTime.parse()`
- **LLM response cleaner**: Regex strips leaked `**ACTIVATED**`, `**PROTOCOL**`, `Mode B:` labels
- **`max_tokens`**: 120 → 250 (longer therapeutic responses)
- **Audio gate**: Only sends audio to ML API if Whisper confirmed actual speech
- **System prompt additions**: ACTIVE CONTEXTUAL RECALL directive, FORBIDDEN OPENERS list, explicit mode-label-leaking prevention

### What v3 (production) added over v2:
- **`httpx.AsyncClient` connection pool** via `lifespan()` context manager
- **FFmpeg RAM pipes** (`pipe:0`/`pipe:1`) — zero disk I/O
- **`BackgroundTasks`** for non-blocking DB saves
- **Bulk `delete().where()`** — fixes N+1 delete loop
- **`.env` environment variables** — API keys no longer hardcoded
- **Pre-compiled regex** at module scope with `re.compile()`
- **`llama-3.3-70b-versatile`** — upgraded from Llama-3.1-8b (v0) → 3.1-8b (v1) → 3.3-70B production

---

## Header 3: Micro-Decision Log (Phase 8 → Phase 9 Complete Diff)

| Component | Phase 7 | Phase 9 | Change Type |
|---|---|---|---|
| Audio inference engine | Keras `model.predict()` | TFLite `interpreter.invoke()` | Algorithmic pivot |
| Audio disk I/O | `NamedTemporaryFile` → `librosa.load(path)` | `soundfile.read(BytesIO(bytes))` | Abstraction removal |
| Audio normalization | None | Z-score `(x-μ)/σ` | Feature engineering addition |
| Video face detector API | `mp.solutions.face_detection` | `mediapipe.tasks.vision.FaceDetector` | Library migration |
| Video confidence threshold | 0.6 | 0.75 | Configuration drift |
| Video frame sampling | `frame_id % 10` | `frame_idx % max(1, int(fps))` | Algorithmic pivot |
| Video prediction | Per-frame `model.predict()` | Batch `compute_vision_inference(np.stack(...))` | Abstraction addition |
| Video graph compilation | None (eager) | `@tf.function(reduce_retracing=True)` + warmup | Algorithmic pivot |
| Text attention ops | `K.tanh`/`K.dot`/`K.softmax`/`K.sum` | `ops.tanh`/`ops.matmul`/`ops.softmax`/`ops.sum` | Library migration |
| Text regex | Runtime compilation | `re.compile()` at module scope | Performance optimization |
| Text confidence filter | Single threshold (0.40) | Dual threshold (0.50 + 0.15 gap) | Algorithmic pivot |
| Text sharpening | None | `predictions ** 1.5` | Feature engineering addition |
| NLTK | `nltk.download()` at runtime | `setup_nltk.py` during build | Abstraction removal |
| Concurrency | All synchronous on event loop | `asyncio.to_thread()` everywhere | Architectural pivot |
| Orchestration | None | `orchestrator_v3.py` | New component |
| LLM | None | Groq Llama-3.3-70B | New component |
| Auth | None | JWT HS256 | New component |
| Database | None | Async SQLAlchemy + aiosqlite | New component |
| Crisis detection | None | Pre-compiled regex gate | New component |
| Contradiction engine | None | Affective masking detection | New component |
| Deployment | Manual `python main.py` | `start_servers.bat` with 4 workers each | Infrastructure addition |

---

## Header 4: Inter-Phase Diff Analysis

### Complete Function Lineage (Phase 1 → Phase 9)

| Function Concept | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 6 | Phase 7 | Phase 9 |
|---|---|---|---|---|---|---|---|
| Face preprocessing | — | — | `preprocess_face()` 48×48 | — | `preprocess_face()` 48×48 | `preprocess_face()` 112×112+CLAHE | `preprocess_face()` 112×112+CLAHE |
| Face detection | — | — | `face_cascade.detectMultiScale()` | — | `face_cascade.detectMultiScale()` | `face_detector.process()` | `detector.detect()` (Tasks API) |
| Face prediction | — | — | `model.predict()` | — | `model.predict()` × 0.55 | `model.predict()` | `compute_vision_inference()` `@tf.function` |
| Audio features | — | `extract_features()` | `extract_features()` | — | `extract_audio_features()` +delta | `process_audio()` +trim | `get_features_fast()` +trim+Z-score |
| Audio prediction | — | — | `predict_emotion()` | — | `predict_voice_emotion()` × 0.66 | `model.predict()` | `interpreter.invoke()` TFLite |
| Text tokenization | — | — | — | `tokenizer.texts_to_sequences([text])` | — | `tokenizer.texts_to_sequences(sentences)` batch | `tokenizer.texts_to_sequences(sentences)` batch |
| Text prediction | — | — | — | `model.predict(pad)` per-sent | — | `model.predict(batch)` | `compute_inference(batch)` `@tf.function` |
| Text attention | — | — | — | `tf.tensordot` / `K.tanh` | — | `K.tanh`/`K.dot` | `ops.tanh`/`ops.matmul` |
| Negation handling | — | — | — | `rewrite_sentence()` | — | Removed | Removed |
| LLM response | — | — | — | — | — | — | `groq.chat.completions.create()` |
| Fusion | — | — | — | — | — | — | Weighted scoring + contradiction engine |
