# 00: Master Chronology Index

## Methodology
Chronological ordering was determined by cross-referencing the following signals:
1. **Import dependency chains**: Older scripts import simpler libraries; newer scripts import from older models or reference their outputs (`.h5`, `.pkl` files).
2. **Model file references**: Scripts referencing `face_emotion_model.h5` (48×48 input, Phase 1 CNN) predate scripts referencing `fer_best_model.keras` (112×112 input, Phase 4+ CNN).
3. **Architectural complexity**: Desktop-only scripts (`sounddevice`, `keyboard`) predate FastAPI scripts. Monolithic FastAPI scripts predate separated microservices.
4. **Label mapping evolution**: Early scripts use `LabelEncoder` with alphabetical sklearn ordering; later scripts use hardcoded `INT_TO_EMOTION` dicts that match specific training notebooks.
5. **Filename versioning**: `v4.py` → `v5.py` → `v6.py`; `textemo.py` → `textemotion.py` → `textemotion_tf212.py`.
6. **Code patterns**: `model.predict()` → `model(tensor, training=False)` → `@tf.function` → TFLite `interpreter.invoke()`.
7. **File naming conventions**: `2nd attempt *.txt` files explicitly declare themselves as second iterations of separated APIs.

---

## Phase Map

### Phase 1: CNN Model Training & GPU Diagnostics
**Strategic intent**: "Can a neural network classify facial emotions at all?"

| File | Role | Key Evidence |
|---|---|---|
| `1.py` | First CNN face emotion model trainer | Uses `archive_5/` (FER2013 original), 48×48 input, 3-layer CNN, 30 epochs, outputs `emotion_model.h5` |
| `2.py` | GPU availability diagnostic | 3-line script: `tf.config.list_physical_devices('GPU')`. Pure diagnostic utility, no ML logic. |
| `cnn model/1.ipynb` | Training notebook for first CNN | Contains training logs, generates `face_emotion_model.h5` (676KB — tiny model) |
| `cnn model/classification_report.txt` | Evaluation output of first CNN | **57% accuracy**. Disgust recall: 3%. Fear recall: 18%. |
| `cnn model/accuracy_plot.png` | Training curve visualization | Visual evidence of overfitting |
| `cnn model/confusion_matrix.png` | Confusion matrix | Shows massive misclassification on minority classes |
| `classification_report.txt` (root) | Duplicate of the CNN classification report | Identical 57% accuracy report, copied to root for quick reference |

**Phase termination reason**: 57% accuracy was unacceptable. Disgust had 3% recall. The FER2013 labels were noisy. Model architecture was too shallow.

---

### Phase 2: Audio LSTM Model Training
**Strategic intent**: "Can we build a speech emotion recognizer from RAVDESS audio?"

| File | Role | Key Evidence |
|---|---|---|
| `v4.py` | First LSTM audio model trainer | Uses `archive_6/` (RAVDESS), `emotion_map` merges calm→neutral, basic `Sequential` LSTM (128 units), no augmentation, no trimming, no normalization, 40 epochs, outputs `speech_emotion_model_7.h5` |

**Phase termination reason**: Model trained but had no serving infrastructure. Raw `.h5` file sitting on disk. No API, no preprocessing pipeline for real-time input. MFCC extraction had no silence trimming, no normalization.

---

### Phase 3: Desktop Real-Time Prototypes (Hardware-Locked)
**Strategic intent**: "Can we run emotion detection live on this laptop?"

| File | Role | Key Evidence |
|---|---|---|
| `3.py` | Live webcam face emotion detector | Loads `face_emotion_model.h5` (Phase 1 CNN), uses `cv2.CascadeClassifier` (Haar), webcam loop `cv2.VideoCapture(0)`, labels are Title-Case `['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']`, applies arbitrary scaling `raw * 0.55`, renders bounding boxes and probabilities directly onto the OpenCV window |
| `v5.py` | Continuous live speech emotion + Vosk subtitles | Loads `speech_emotion_model_7.h5` (Phase 2 LSTM), uses `sounddevice.RawInputStream`, `queue.Queue`, `vosk.Model` for live transcription, infinite `while True` loop, emotion predicted only AFTER `KeyboardInterrupt` (Ctrl+C), labels include 'calm' as separate class (`EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust']`) |
| `v6.py` | Push-to-talk version with spacebar control | Identical model loading to `v5.py`, adds `keyboard.is_pressed("space")` for record control, `heartbeat_bar()` volume visualizer, `MIN_SPEECH_VOLUME = 1500` threshold, color-coded terminal output per emotion |

**Phase termination reason**: These scripts are hardware-locked. `sounddevice` requires a physical microphone. `keyboard.is_pressed` requires a physical keyboard. `cv2.VideoCapture(0)` requires a physical webcam. Impossible to deploy to a cloud server or connect to a Flutter mobile app.

---

### Phase 4: Text Emotion Model Exploration (Multiple Iterations)
**Strategic intent**: "Can we classify emotion from text? Which model architecture works?"

| File | Role | Key Evidence |
|---|---|---|
| `textemo.py` | First BiLSTM text model trainer | Uses `archive_8/combined_emotion.csv`, `LabelEncoder`, single `Bidirectional(LSTM(128))`, `max_words=20000`, `max_len=50`, only 5 epochs, `validation_split=0.2` (no stratification), outputs `emotion_bilstm_model.h5`, `tokenizer1.pkl`, `label_encoder1.pkl` |
| `textcalemo.py` | Simple CNN-based text emotion predictor | Loads `emotion_TEXT_cnn_model.h5` + `tokenizer.pkl` + `label_encoder.pkl`, `pad_sequences(seq, maxlen=100)`, uses `le.inverse_transform`, simple `while True: input()` loop. No sentence splitting, no paragraph analysis. |
| `textemotion.py` | Text emotion tester with custom AttentionLayer | First appearance of `AttentionLayer` class (using `tf.tensordot`), uses `glob.glob("trained_models/aa/...")` to dynamically find model files, loads with `compile=False`, `PorterStemmer` imported but never used (abandoned feature), `MAX_LEN = 60` (changed later to 50, then 100) |
| `textemotion_tf212.py` | Advanced text predictor: negation engine + context filter | Adds `is_context_clear()` function (rejects sentences ≤2 words, rejects non-alpha-heavy strings, requires presence of emotion keywords), adds `NEGATION_MAP` dictionary (`"happy":"sad"`, `"angry":"calm"`, etc.), adds `rewrite_sentence()` regex engine, `MAX_LEN = 50`, sentence-level `for` loop with individual `model.predict()` calls, paragraph-level vote counter and probability accumulator |
| `emotion_api/testtext.py` | Copy of `textemotion_tf212.py` | Byte-identical to `textemotion_tf212.py`, placed inside `emotion_api/` folder for co-location with the unified API |

**Phase termination reason**: The `is_context_clear()` function was too aggressive — it rejected valid emotional sentences like "I'm sad" (only 2 words). The `NEGATION_MAP` approach was fundamentally unscalable (cannot cover the English language with a dictionary). The model itself (`emotion_model_tf212_fixed.h5`, `max_len=50`) was retrained from scratch later on Kaggle with a BiLSTM+Attention architecture, `max_len=100`, and `MAX_VOCAB_SIZE=30000`.

---

### Phase 5: Model Format Conversion Experiments (Abandoned)
**Strategic intent**: "Can we convert models to lighter formats for deployment?"

| File | Role | Key Evidence |
|---|---|---|
| `6.py` | H5 → SavedModel → ONNX conversion attempt | Loads `my_model.h5`, saves as `saved_model/`, comments out CLI conversion: `python -m tf2onnx.convert --saved-model saved_model --output model.onnx`. Script is 10 lines. No error handling. |
| `7.py` | TFLite → TensorFlow concrete function attempt | Loads `emotion_model.tflite`, creates `representative_dataset()` generator, attempts `@tf.function` wrapping of `interpreter.invoke()` — this is architecturally invalid (you cannot wrap an interpreter invocation in a tf.function). Script ends with no output, two blank lines. Abandoned. |

**Phase termination reason**: Both approaches were dead ends. ONNX conversion was not pursued further. The TFLite-to-TF reverse conversion in `7.py` is logically impossible as written. The correct forward conversion (Keras → TFLite) was not achieved until Phase 8 (`convert_audio_model.py`).

---

### Phase 6: First Unified API (The Monolith)
**Strategic intent**: "Combine face + voice into a single FastAPI server that Flutter can call."

| File | Role | Key Evidence |
|---|---|---|
| `emotion_api/main.py` | Unified FastAPI: face + voice endpoints | Loads `face_emotion_model.h5` (Phase 1 CNN, 48×48) AND `final_lstm_model_tf212.h5` (voice model), Haar Cascade face detection, labels Title-Case `['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']`, voice labels include `calm` as index 4, `merged_voice_label()` maps calm→neutral, face scaling `raw * 0.55`, voice scaling `raw * 0.66`, voice uses `librosa.load(audio_stream, sr=22050)` (not 16kHz!), voice features use MFCC+delta+delta2 stacked to shape `(1, 174, 120)`, video endpoint uses `NamedTemporaryFile(delete=True)`, every-10th-frame sampling, Haar Cascade detection per frame, `model.predict()` per frame (no batching), single Uvicorn worker on port 8000 with `reload=True` |
| `old endpoints.txt` | Variant of the unified API | Nearly identical to `emotion_api/main.py` but references different model files: `facial_emotion.json` + `facial_emotion.h5` (JSON+weights format instead of single H5), `speech_emotion_model_7(2).h5` (copy of Phase 2 model), face scaling `raw * 0.65` (different from main.py's 0.55), includes `sys._MEIPASS` check (PyInstaller compatibility) |
| `emotion_api.spec` | PyInstaller bundling specification | Bundles `face_emotion_model.h5` + `final_lstm_model_tf212.h5` into a single `.exe`, hidden imports include `tensorflow`, `cv2`, `librosa`. Evidence of an attempt to distribute the API as a standalone Windows executable. |

**Phase termination reason**: The monolith loaded all models into one process, consuming excessive RAM. Face and voice models competed for the same thread. `reload=True` in production caused memory leaks. The 48×48 CNN model had 57% accuracy. Haar Cascades produced false positives. `librosa.load` at 22050 Hz was wrong for the 16kHz-trained voice model. No text endpoint existed. No orchestration logic existed.

---

### Phase 7: Second Attempt — Separated APIs (Still Flawed)
**Strategic intent**: "Split each modality into its own FastAPI server. Fix the worst bugs from Phase 6."

| File | Role | Key Evidence |
|---|---|---|
| `2nd attempt Audio.txt` | Standalone Audio API | Port 8002, `MAX_FILE_SIZE_MB = 10`, file extension whitelist `[".wav", ".m4a", ".mp3", ".ogg"]`, `NamedTemporaryFile` for disk I/O, `librosa.load(file_path, sr=SAMPLE_RATE)` (corrected to 16kHz), `librosa.effects.trim(audio, top_db=30)` (first appearance of silence trimming in API context), `model.predict(tensor, verbose=0)` (still eager), `CONFIDENCE_THRESHOLD = 0.40`, auto-rename of `.keras.zip` files (WhatsApp/Google Drive corruption fix), `finally:` block for temp file cleanup |
| `2nd attempt Video.txt` | Standalone Image+Video API | Port 8002, first use of `mediapipe.solutions.face_detection` (replacing Haar Cascade), `min_detection_confidence=0.6`, first use of `cv2.createCLAHE(clipLimit=2.0)`, 112×112 input size with `cv2.INTER_CUBIC`, loads `fer_best_model.keras` (new retrained model), video still uses every-10th-frame (`frame_id % 10`), `model.predict()` called per-frame inside loop (no batching), `NamedTemporaryFile(delete=True)` for video disk I/O |
| `2nd attempt Text.txt` | Standalone Text API | Port 8001, `nltk.download('punkt', quiet=True)` at global scope, `AttentionLayer` using `K.tanh`/`K.dot`/`K.softmax`/`K.sum` (Keras 2 backend ops), `initializer="normal"` for attention weights, includes `compute_output_shape()` method, batch inference introduced (`model.predict(padded_batch)`), but still `model.predict()` not `@tf.function`, no `asyncio.to_thread`, no regex precompilation, no prediction sharpening |

**Phase termination reason**: The `async def` endpoints still ran synchronous `model.predict()` and `librosa.load()`, blocking the event loop. Video used fixed every-10th-frame decimation regardless of FPS. MediaPipe confidence at 0.6 was too low (still admitted non-human objects). No orchestrator existed to combine the three APIs. No LLM integration. No database. No authentication.

---

### Phase 8: Modern Production Microservices (Current)
**Strategic intent**: "Enterprise-grade, concurrent, safe, fast. Ready for FYP presentation and real users."

| File | Role | Key Evidence |
|---|---|---|
| `services/audio_api/main_audio.py` | Production Audio API | Port 8000, TFLite primary inference with Keras fallback, `soundfile.read()` for in-memory audio (no disk), Z-score normalization `(mfccs - mean) / std`, MFCC-only features (no delta/delta2), `asyncio.to_thread()` for all heavy ops, `MAX_FILE_SIZE_MB = 50`, structured logging to central `logs/` directory |
| `services/audio_api/convert_audio_model.py` | Keras → TFLite converter | `tf.lite.Optimize.DEFAULT` + `float16` quantization, `SELECT_TF_OPS` for BiLSTM compatibility, `_experimental_lower_tensor_list_ops = False` |
| `services/image_video_api/main_video.py` | Production Vision API | Port 8002, MediaPipe Tasks Vision API (`blaze_face_short_range.tflite`, auto-downloaded), `min_detection_confidence=0.75`, `@tf.function(reduce_retracing=True)` with warmup, FPS-aware decimation `frame_mod = max(1, int(fps))`, batch prediction via `np.stack` + single `compute_vision_inference(batch)` call, UUID-based temp files with `finally:` cleanup, `MAX_FILE_SIZE = 250MB` |
| `services/text_api/main_text.py` | Production Text API | Port 8001, pre-compiled regex (`re.compile`), `keras.ops.tanh`/`ops.matmul`/`ops.softmax`/`ops.sum` (Keras 3 ops), `@keras.saving.register_keras_serializable()`, `@tf.function(reduce_retracing=True)` with warmup, `asyncio.to_thread()`, prediction sharpening `predictions ** 1.5`, dual-threshold filtering (confidence < 0.50 OR gap < 0.15 → "context unclear"), NLTK verified locally (not downloaded at runtime) |
| `services/fusion_api/orchestrator_v3.py` | Master Orchestrator | Port 8003, `httpx.AsyncClient` global connection pool via `lifespan()`, FFmpeg RAM pipes (`pipe:0`/`pipe:1`, zero disk I/O), Groq Whisper transcription, Groq Llama-3.3-70B LLM with 6-mode system prompt, JWT authentication (HS256, issuer/audience validation), emotion fusion with weighted scoring (`text:0.50`, `visual:0.35`, `audio:0.15`), contradiction engine (`masked_distress`), crisis/abuse regex pre-flight gate, `BackgroundTasks` for async DB saves, bulk `delete()` for session cleanup, `show_emotion_ui` flag for Flutter |
| `services/fusion_api/database.py` | Async SQLAlchemy ORM | `aiosqlite` async engine, `ChatSession` + `ChatMessage` models, `ForeignKey` relationships, `datetime.utcnow` timestamps |
| `launch_scripts/start_servers.bat` | Multi-service launcher | Spawns 4 separate `cmd` windows, each activating its own `venv`, each running `uvicorn --workers 4` |
| `launch_scripts/setup_nltk.py` | Offline NLTK data installer | Downloads `punkt` and `punkt_tab` during build, not at runtime |

---

### Support Files (Not Phase-Specific)

| File | Role |
|---|---|
| `Model notebooks.txt` | Complete Kaggle training notebooks for all 3 production models (Text BiLSTM+Attention 94.04% accuracy, FER CNN 81.03% accuracy, Audio BiLSTM 94.10% accuracy). Contains dataset descriptions, preprocessing code, training logs, evaluation metrics, and Gradio demo code. |
| `problems to fix.txt` | Architectural flaw catalog identifying 8 critical loopholes and their proposed solutions. Served as the engineering roadmap for Phase 7→Phase 8 transition. |
| `HWMonitor.txt` | Hardware monitoring data (CPU/GPU temperatures, utilization during inference testing) |
| `.env` | Environment variables for Groq and Cerebras API keys |

---

## Chronological Timeline Summary

```
Phase 1  →  1.py, 2.py, cnn model/
Phase 2  →  v4.py
Phase 3  →  3.py, v5.py, v6.py
Phase 4  →  textemo.py, textcalemo.py, textemotion.py, textemotion_tf212.py
Phase 5  →  6.py, 7.py
Phase 6  →  emotion_api/main.py, old endpoints.txt, emotion_api.spec
Phase 7  →  2nd attempt Audio.txt, 2nd attempt Video.txt, 2nd attempt Text.txt
Phase 8  →  services/audio_api/*, services/image_video_api/*, services/text_api/*, services/fusion_api/*, launch_scripts/*
```
