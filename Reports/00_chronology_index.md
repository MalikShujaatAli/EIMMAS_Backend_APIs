# 00: Master Chronology Index

## Methodology
Chronological ordering was determined by cross-referencing the following signals:
1. **Import dependency chains**: Older scripts import simpler libraries; newer scripts import from older models or reference their outputs (`.h5`, `.pkl` files).
2. **Model file references**: Scripts referencing `face_emotion_model.h5` (48×48 input, Phase 1 CNN) predate scripts referencing `fer_best_model.keras` (112×112 input, Phase 4+ CNN).
3. **Architectural complexity**: Desktop-only scripts (`sounddevice`, `keyboard`) predate FastAPI scripts. Monolithic FastAPI scripts predate separated microservices.
4. **Label mapping evolution**: Early scripts use `LabelEncoder` with alphabetical sklearn ordering; later scripts use hardcoded `INT_TO_EMOTION` dicts that match specific training notebooks.
5. **Filename versioning**: `phase02_audio_lstm_trainer.py` → `phase03_audio_live_vosk.py` → `phase03_audio_push_to_talk.py`; `phase04_text_bilstm_trainer.py` → `phase04_text_attention_tester.py` → `phase04_text_negation_engine.py`.
6. **Code patterns**: `model.predict()` → `model(tensor, training=False)` → `@tf.function` → TFLite `interpreter.invoke()`.
7. **File naming conventions**: `2nd attempt *.txt` files explicitly declare themselves as second iterations of separated APIs.
8. **Orchestrator versioning**: `phase09_fusion_orchestrator_v0.py` → `phase09_fusion_orchestrator_v1.py` → `phase09_fusion_orchestrator_v2.py` → `orchestrator_v3.py` (production).

---

## Phase Map

### Phase 1: CNN Model Training & GPU Diagnostics
**Strategic intent**: "Can a neural network classify facial emotions at all?"

| File | Role | Key Evidence |
|---|---|---|
| `phase01_vision_cnn_trainer.py` | First CNN face emotion model trainer | Uses `archive_5/` (FER2013 original), 48×48 input, 3-layer CNN, 30 epochs, outputs `emotion_model.h5` |
| `phase01_diagnostic_gpu_check.py` | GPU availability diagnostic | 3-line script: `tf.config.list_physical_devices('GPU')`. Pure diagnostic utility, no ML logic. |
| `phase01_vision_cnn_notebook.ipynb` | Training notebook for first CNN | Contains training logs, generates `face_emotion_model.h5` (676KB — tiny model) |
| `phase01_vision_classification_report.txt` | Evaluation output of first CNN | **57% accuracy**. Disgust recall: 3%. Fear recall: 18%. |
| `phase01_vision_accuracy_plot.png` | Training curve visualization | Visual evidence of overfitting |
| `phase01_vision_confusion_matrix.png` | Confusion matrix | Shows massive misclassification on minority classes |
| `phase01_vision_classification_report_root.txt` (root) | Duplicate of the CNN classification report | Identical 57% accuracy report, copied to root for quick reference |

**Phase termination reason**: 57% accuracy was unacceptable. Disgust had 3% recall. The FER2013 labels were noisy. Model architecture was too shallow.

---

### Phase 2: Audio LSTM Model Training
**Strategic intent**: "Can we build a speech emotion recognizer from RAVDESS audio?"

| File | Role | Key Evidence |
|---|---|---|
| `phase02_audio_lstm_trainer.py` | First LSTM audio model trainer | Uses `archive_6/` (RAVDESS), `emotion_map` merges calm→neutral, basic `Sequential` LSTM (128 units), no augmentation, no trimming, no normalization, 40 epochs, outputs `speech_emotion_model_7.h5` |
| `phase02_audio_training_notebook.ipynb` | Audio model training notebook | 93KB notebook, likely contains training logs and experimentation for LSTM models |
| `phase02_audio_training_notebook_v2.ipynb` | Second audio training notebook | 47KB, another iteration of audio model training |
| `lstm_fold1_best.h5` through `lstm_fold5_best.h5` | 5-Fold cross-validation model checkpoints | Evidence of systematic k-fold validation of the LSTM model |
| `lstm_histories.pkl` | Training history for all 5 folds | Pickled training curves across folds |
| `lstm_confusion_matrix.png` | Confusion matrix for LSTM evaluation | 234KB image showing per-class performance |
| `lstm_training_curves.png` | Training/validation curves across folds | 238KB image showing convergence patterns |
| `speech_emotion_model_7.h5` | First trained LSTM model | 1.18MB, 7-class (with calm→neutral merge at training time) |
| `speech_emotion_model_7(2).h5` | Duplicate/re-download of same model | Nearly identical file size (1.18MB), likely a copy |
| `speech_emotion_model_7_lstm_clean.h5` | Cleaned variant | Same architecture, minor retraining iteration |

**Phase termination reason**: Model trained but had no serving infrastructure. Raw `.h5` file sitting on disk. No API, no preprocessing pipeline for real-time input. MFCC extraction had no silence trimming, no normalization.

---

### Phase 3: Desktop Real-Time Prototypes (Hardware-Locked)
**Strategic intent**: "Can we run emotion detection live on this laptop?"

| File | Role | Key Evidence |
|---|---|---|
| `phase03_vision_webcam_scaled.py` | Live webcam face emotion detector | Loads `face_emotion_model.h5` (Phase 1 CNN), uses `cv2.CascadeClassifier` (Haar), webcam loop `cv2.VideoCapture(0)`, labels are Title-Case `['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']`, applies arbitrary scaling `raw * 0.55`, renders bounding boxes and probabilities directly onto the OpenCV window |
| `phase03_vision_webcam_early.py` | **Earliest webcam prototype** | Loads `emotiondetector1.json` + `emotiondetector1.h5` (JSON+weights format — even older than `face_emotion_model.h5`), Haar Cascade, `cv2.VideoCapture(0)`, labels `{0:'angry'...6:'surprise'}`, per-frame `model.predict(img)`, overlays text with `cv2.putText`, catches `cv2.error` silently. **This is the earliest face detection prototype** — simpler than `phase03_vision_webcam_scaled.py`, uses a different model file, no probability display, no scaling factor. |
| `phase03_audio_live_vosk.py` | Continuous live speech emotion + Vosk subtitles | Loads `speech_emotion_model_7.h5` (Phase 2 LSTM), uses `sounddevice.RawInputStream`, `queue.Queue`, `vosk.Model` for live transcription, infinite `while True` loop, emotion predicted only AFTER `KeyboardInterrupt` (Ctrl+C), labels include 'calm' as separate class |
| `phase03_audio_push_to_talk.py` | Push-to-talk version with spacebar control | Identical model loading to `phase03_audio_live_vosk.py`, adds `keyboard.is_pressed("space")` for record control, `heartbeat_bar()` volume visualizer, `MIN_SPEECH_VOLUME = 1500` threshold, color-coded terminal output per emotion |

**Phase termination reason**: These scripts are hardware-locked. `sounddevice` requires a physical microphone. `keyboard.is_pressed` requires a physical keyboard. `cv2.VideoCapture(0)` requires a physical webcam. Impossible to deploy to a cloud server or connect to a Flutter mobile app.

---

### Phase 4: Text Emotion Model Exploration (Multiple Iterations)
**Strategic intent**: "Can we classify emotion from text? Which model architecture works?"

| File | Role | Key Evidence |
|---|---|---|
| `phase04_text_bilstm_trainer.py` | First BiLSTM text model trainer | Uses `archive_8/combined_emotion.csv`, `LabelEncoder`, single `Bidirectional(LSTM(128))`, `max_words=20000`, `max_len=50`, only 5 epochs, outputs `emotion_bilstm_model.h5`, `tokenizer1.pkl`, `label_encoder1.pkl` |
| `phase04_text_training_notebook.ipynb` | Text model training notebook | 255KB notebook containing training experiments and evaluation |
| `phase04_text_exploration_notebook.ipynb` | Another text training notebook | 35KB, lighter iteration of text model exploration |
| `phase04_text_cnn_predictor.py` | Simple CNN-based text emotion predictor | Loads `emotion_TEXT_cnn_model.h5`, `maxlen=100`, simple `while True: input()` loop. No sentence splitting. |
| `phase04_text_attention_tester.py` | Text tester with custom AttentionLayer | First `AttentionLayer` (using `tf.tensordot`), `glob.glob()` dynamic model discovery, `PorterStemmer` imported but never used, `MAX_LEN = 60` |
| `phase04_text_negation_engine.py` | Advanced text predictor: negation engine + context filter | Adds `is_context_clear()`, `NEGATION_MAP`, `rewrite_sentence()` regex engine, `MAX_LEN = 50`, per-sentence `model.predict()`, vote counter |
| `phase04_text_bilstm_attention_v1.py` | **Text predictor with AttentionLayer (K.tanh/K.dot)** | Uses `K.tanh(K.dot(x, self.W) + self.b)` attention, `initializer="normal"`, `Counter(emotions).most_common(1)` for majority voting, loads `emotion_model_20251203_192244.h5` (timestamped model), `MAX_LEN = 50`, per-sentence prediction loop. **Bridge between `phase04_text_attention_tester.py` and the API versions** — has the mature Bahdanau attention but no FastAPI. |
| `phase04_text_api_bilstm_keyword.py` | **First text emotion FastAPI endpoint** | Identical `AttentionLayer` to `phase04_text_bilstm_attention_v1.py`, adds **massive `rawemotionwords` keyword list** (~150 words including "happy", "sad", "betrayed", "spiraling"), `sentence_has_emotion()` checks if any word in the sentence matches the list before predicting, FastAPI `POST /predict_text`, `Pydantic TextInput`, vote counter + probability accumulator, port 8001, `reload=True`. **This is the missing link between Phase 4's desktop scripts and Phase 7's `phase07_text_api_standalone.txt`**. |
| `phase04_text_negation_engine_copy.py` | Copy of `phase04_text_negation_engine.py` | Byte-identical, placed in `emotion_api/` for co-location with the unified API |

**Phase termination reason**: The `is_context_clear()` / `rawemotionwords` / `NEGATION_MAP` approaches were all fundamentally unscalable. The model was retrained on Kaggle with BiLSTM+Attention, `max_len=100`, and `MAX_VOCAB_SIZE=30000`.

---

### Phase 5: Model Format Conversion Experiments (Abandoned)
**Strategic intent**: "Can we convert models to lighter formats for deployment?"

| File | Role | Key Evidence |
|---|---|---|
| `phase05_convert_onnx_attempt.py` | H5 → SavedModel → ONNX conversion attempt | 10 lines. CLI conversion commented out. Abandoned. |
| `phase05_convert_tflite_attempt.py` | TFLite → TensorFlow concrete function attempt | Architecturally invalid `@tf.function` wrapping. Abandoned. |

**Phase termination reason**: Both approaches were dead ends.

---

### Phase 6: First Unified API (The Monolith)
**Strategic intent**: "Combine face + voice into a single FastAPI server that Flutter can call."

| File | Role | Key Evidence |
|---|---|---|
| `phase06_fusion_api_monolith.py` | Unified FastAPI: face + voice endpoints | Loads Phase 1 CNN (48×48) + voice LSTM, Haar Cascade, `sr=22050` (bug), MFCC+delta stacking, face ×0.55, voice ×0.66, every-10th-frame, `reload=True` |
| `phase06_fusion_api_json_variant.py` | **Variant monolith with JSON+weights model** | Loads `facial_emotion.json` + `facial_emotion.h5` (JSON+weights format), `speech_emotion_model_7(2).h5`, face scaling ×0.65 (different from phase06_fusion_api_monolith.py's 0.55), `CUDA_VISIBLE_DEVICES="-1"` (force CPU), `sys._MEIPASS` PyInstaller check for Haar Cascade path, `model.compile()` called explicitly after loading, `workers=1`, `reload=False`. **Evidence of a pre-PyInstaller deployment attempt** — this script was designed to run as a bundled `.exe`. |
| `phase06_fusion_api_old_endpoints.txt` | Another variant of the unified API | References `facial_emotion.json` + `speech_emotion_model_7(2).h5`, face scaling ×0.65, includes `sys._MEIPASS` check |
| `phase06_vision_api_pyinstaller_variant.txt` | Variant image API | Focuses on single-modality PyInstaller binding with strict CPU flags (`TF_ENABLE_ONEDNN_OPTS="0"`). Contains `model_from_json()` loading for `emotiondetector1.json`. |
| `phase06_fusion_api_pyinstaller_spec.spec` | PyInstaller bundling specification | Attempted `.exe` bundling. Abandoned. |

**Phase termination reason**: Monolith, low accuracy, Haar Cascades, wrong sample rate, no text, no orchestration.

---

### Phase 7: Second Attempt — Separated APIs (Still Flawed)
**Strategic intent**: "Split each modality into its own FastAPI server. Fix the worst bugs."

| File | Role | Key Evidence |
|---|---|---|
| `phase07_audio_api_standalone.txt` | Standalone Audio API | Port 8002, 10MB limit, silence trim, `sr=16000`, `finally` cleanup, still `model.predict()` on event loop |
| `phase07_vision_api_standalone.txt` | Standalone Image+Video API | Port 8002, MediaPipe 0.6, CLAHE, 112×112, `fer_best_model.keras`, still every-10th-frame, per-frame prediction |
| `phase07_text_api_standalone.txt` | Standalone Text API | Port 8001, `K.tanh`/`K.dot` attention, batch `model.predict(padded_batch)`, `nltk.download()` at startup |

**Phase termination reason**: Still blocking event loop. No `asyncio.to_thread()`. No `@tf.function`. No orchestrator.

---

### Phase 8: Pre-Production APIs (The Bridge to Phase 9)
**Strategic intent**: "Deploy the individual APIs with production-grade patterns before adding the orchestrator."

| File | Role | Key Evidence |
|---|---|---|
| `phase08_audio_api_preprod.py` (FYP old/) | **Pre-production Audio API** | Port 8000, `soundfile.read()` in-memory (first zero-disk audio!), `model(tensor, training=False)` direct call (not `model.predict()`), warmup with dummy input, `INT_TO_EMOTION` dict (7-class), `CONFIDENCE_THRESHOLD = 0.40`, 10MB limit. **Missing from Phase 9**: No Z-score normalization, no TFLite, no `asyncio.to_thread()`, no `@tf.function`. |
| `phase08_vision_api_preprod.py` (FYP old/) | **Pre-production Vision API** | Port 8002, MediaPipe Tasks Vision API (`blaze_face_short_range.tflite`, auto-download from Google Storage), `min_detection_confidence=0.75`, `fer_best_model.keras` with zip auto-extract, `emotion_model(inp, training=False)` direct call, warmup. **Missing from Phase 9**: Still `frame_id % 10` (not FPS-aware), per-frame `analyze_emotion()` (no batching), no `@tf.function`, temp file uses `f"v_temp_{file.filename}"` (collision risk), no UUID. |
| `phase08_text_api_preprod.py` (FYP old/) | **Pre-production Text API** | Port 8001, `keras.ops.tanh`/`ops.matmul`/`ops.softmax`/`ops.sum` (Keras 3 ops!), `@keras.saving.register_keras_serializable()`, `predictions ** 1.5` sharpening (first appearance!), dual-threshold `max_prob < 0.50 or gap < 0.15`, `clean_text()` preprocessing, `INT_TO_EMOTION` with `'love'` class, `text_best_model.keras` + `text_tokenizer.pkl`. **Missing from Phase 9**: No `@tf.function`, no `asyncio.to_thread()`, no pre-compiled regex, `nltk.download('punkt')` still at startup, `len(text.split()) < 3` hardcoded rejection. |
| `phase08_text_api_intermediate.py` | **Intermediate text API variant** | Port 8001, `tf.math.tanh`/`tf.linalg.matmul` (TF ops, not keras.ops), `nltk.download('punkt_tab')` (different NLTK package), single threshold `CONFIDENCE_THRESHOLD = 0.40` (no gap check), batch `model.predict()`, `get_config()` method, `INT_TO_EMOTION` with `'love'` class. **This sits between `phase07_text_api_standalone.txt` (Phase 7) and `phase08_text_api_preprod.py` (Phase 8)** — it has the new model but not the dual-threshold or sharpening. |

**Phase termination reason**: These APIs were individually deployable but lacked the orchestration layer. No fusion, no LLM, no auth, no DB, no safety gates. They became the direct ancestors of the Phase 9 `services/` files.

---

### Phase 9: Orchestrator Evolution (v0 → v1 → v2 → v3)
**Strategic intent**: "Build the master orchestrator to coordinate all APIs, add LLM, auth, DB, and safety."

| File | Role | Key Evidence |
|---|---|---|
| `phase09_fusion_orchestrator_v0.py` | **First orchestrator** (314 lines) | Groq LLM only (no Cerebras), `llama-3.1-8b-instant` (small model), hardcoded API key in source, `WEIGHTS` visual-heavy `{visual:0.45, audio:0.35, text:0.20}`, simple JWT (`jwt.decode` no issuer/audience), 4-sentence system prompt, `NamedTemporaryFile` FFmpeg (disk I/O), `fuse_emotions()` with single-line `emotion_map`, `httpx.AsyncClient()` created per request (no pool), chat history via DB but session title set BEFORE input validation (bug), `uvicorn run` references `orchestrator_v2:app` (copy-paste filename mismatch). |
| `phase09_fusion_orchestrator_v1.py` | **Second orchestrator** (528 lines) | Adds Cerebras dual-engine `USE_PRODUCTION_MODEL` toggle, `WEIGHTS` rebalanced to text-heavy `{visual:0.35, audio:0.15, text:0.50}`, file logging with UTF-8 encoding fix (Windows emoji crash), comprehensive JWT claim extraction (`email`, `unique_name`, XML schema claims), `issuer`/`audience` validation, 5-mode system prompt (Standard/Crisis/Boundary/Scope/Language Barrier), extended `emotion_map` (preserves `love`), confidence extraction differentiated by modality, LLM fallback dict, DELETE `/sessions/{id}` endpoint with N+1 loop (`for msg in messages: await db.delete(msg)`), `filter out loudnorm/afftdn` comment (removed audio filters), Whisper emoji fix. |
| `phase09_fusion_orchestrator_v2.py` | **Third orchestrator** (617 lines) | Adds **pre-compiled crisis+abuse regex gate** with `\b` word boundaries and `.*?` wildcard patterns, **contradiction engine** (`masked_distress` flag), **Whisper hallucination cleaner** (`re.sub(r'\[.*?\]\|\(.*?\)', '', text)`), **Ghost Gate** (rejects empty media), `show_emotion_ui` Flutter flag, ISO timestamp formatting (`isoformat() + "Z"`), **LLM response regex cleaner** (strips leaked `**ACTIVATED**`, `**PROTOCOL**`, `Mode B:` labels), `max_tokens` increased from 120 to 250, audio gate (only send to Audio ML if Whisper confirmed speech). |
| `orchestrator_v3.py` (production) | **Current production orchestrator** (651 lines) | Located in `services/fusion_api/`. All v2 features plus: `httpx.AsyncClient` global connection pool via `lifespan()`, FFmpeg RAM pipes (`pipe:0`/`pipe:1`), `BackgroundTasks` for async DB saves, bulk `delete().where()` (N+1 fix), `.env` environment variables (no hardcoded keys), pre-compiled regex at module scope with `re.compile()`, `llama-3.3-70b-versatile` (upgraded from 8b). |

---

### Phase 9 Production Services (Current — Final Form)

| File | Location | Role |
|---|---|---|
| `phase08_audio_api_preprod.py` | `services/audio_api/` | Production Audio API: TFLite, Z-score, `asyncio.to_thread()`, `soundfile` in-memory |
| `convert_audio_model.py` | `services/audio_api/` | Keras → TFLite converter with float16 quantization |
| `phase08_vision_api_preprod.py` | `services/image_video_api/` | Production Vision API: `@tf.function` batch, FPS-aware decimation, MediaPipe Tasks 0.75 |
| `phase08_text_api_preprod.py` | `services/text_api/` | Production Text API: `@tf.function`, `asyncio.to_thread()`, pre-compiled regex, dual-threshold |
| `orchestrator_v3.py` | `services/fusion_api/` | Master Orchestrator: fusion, LLM, auth, DB, crisis/abuse gate, contradiction engine |
| `database.py` | `services/fusion_api/` | Async SQLAlchemy ORM models |
| `start_servers.bat` | `launch_scripts/` | Multi-service Windows launcher |
| `setup_nltk.py` | `launch_scripts/` | Offline NLTK data installer |

---

### Model Training Documentation

| File | Role |
|---|---|
| `training_models_notebook_dump.txt` | Complete Kaggle training notebooks for all 3 production models (Text BiLSTM+Attention 94.04%, FER CNN 81.03%, Audio BiLSTM 94.10%). Contains dataset descriptions, preprocessing code, training logs, evaluation metrics, and Gradio demo code. |
| `training_models_old_apis_dump.txt` | **Extended model training notebook dump** (128KB, 2697 lines). Contains the complete FERPlus CNN training pipeline: dataset unzipping, class counting (8000/class + 10379 neutral), class weight computation, CLAHE preprocessing, 4-block CNN architecture (14.5M params), 50-epoch training with augmentation/early stopping/LR reduction, test evaluation (81.03%), classification report, confusion matrix, and Gradio demo UI. Also contains training code for earlier model iterations. |
| `phase04_text_training_notebook.ipynb` | Text model training notebook (255KB) |
| `phase04_text_exploration_notebook.ipynb` | Lighter text model exploration notebook (35KB) |
| `phase02_audio_training_notebook.ipynb` | Audio model training notebook (93KB) |
| `phase02_audio_training_notebook_v2.ipynb` | Second audio training notebook (47KB) |
| `phase01_vision_cnn_notebook_alt.ipynb` | Phase 1 CNN training notebook (189KB) — `phase01_vision_cnn_trainer.py`-equivalent architecture: 3-layer CNN on FER2013, 30 epochs, **57% accuracy**. Contains full training logs with per-epoch metrics. Output: `emotion_model.h5` |
| `test.ipynb` | Empty file (0 bytes) — deleted or never written |

---

### Support Files & Artifacts

| File | Role |
|---|---|
| `.env` | Environment variables for Groq and Cerebras API keys |
| `2022-cs-663(lab 2).txt` | **NOT part of FYP** — C++ compiler design lab assignment. Included accidentally. |
| `recorded_audio.wav` | Test audio recording used during development |
| `phase01_vision_accuracy_plot.png` / `loss_plot.png` | Phase 1 CNN training visualizations |
| `emotion_model.h5` / `face_emotion_model.h5` | Phase 1 model artifacts |
| `emotion_model.tflite` | Phase 1 CNN converted to TFLite (used in Phase 5 experiments) |
| `emotion_TEXT_cnn_model.h5` | Phase 4 CNN text model (abandoned architecture) |
| `final_lstm_model (1).h5` | Phase 2 LSTM model (original, pre-TF2.12 fix) |
| `tokenizer.pkl` / `label_encoder.pkl` | Phase 4 text preprocessing artifacts |
| `vosk-model-small-en-us-0.15/` | Phase 3 Vosk offline speech recognition model |

---

## Chronological Timeline Summary

```
Phase 1    → phase01_vision_cnn_trainer.py, phase01_diagnostic_gpu_check.py, cnn model/
Phase 2    → phase02_audio_lstm_trainer.py, phase02_audio_training_notebook.ipynb, phase02_audio_training_notebook_v2.ipynb, lstm_fold*.h5
Phase 3    → phase03_vision_webcam_early.py, phase03_vision_webcam_scaled.py, phase03_audio_live_vosk.py, phase03_audio_push_to_talk.py
Phase 4    → phase04_text_bilstm_trainer.py, phase04_text_cnn_predictor.py, phase04_text_attention_tester.py, phase04_text_negation_engine.py, phase04_text_bilstm_attention_v1.py, phase04_text_api_bilstm_keyword.py
Phase 5    → phase05_convert_onnx_attempt.py, phase05_convert_tflite_attempt.py
Phase 6    → phase06_fusion_api_monolith.py, phase06_fusion_api_json_variant.py, phase06_fusion_api_old_endpoints.txt, phase06_vision_api_pyinstaller_variant.txt, phase06_fusion_api_pyinstaller_spec.spec
Phase 7    → phase07_audio_api_standalone.txt, phase07_vision_api_standalone.txt, phase07_text_api_standalone.txt
Phase 8  → phase08_text_api_intermediate.py, phase08_audio_api_preprod.py, phase08_vision_api_preprod.py, phase08_text_api_preprod.py (all in FYP old/)
Phase 9a   → phase09_fusion_orchestrator_v0.py
Phase 9b   → phase09_fusion_orchestrator_v1.py
Phase 9c   → phase09_fusion_orchestrator_v2.py
Phase 9    → services/audio_api/*, services/image_video_api/*, services/text_api/*, services/fusion_api/*, launch_scripts/*
```
