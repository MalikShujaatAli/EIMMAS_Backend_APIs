# 99: Cross-Reference Validation Matrix

This document traces the complete lineage of every function, class, configuration constant, and architectural pattern across all 9 phases, identifying births, mutations, deaths, and resurrections.

---

## 1. Function/Class Lineage Table

| Entity | Born | Mutated | Died | Resurrected | Final Form (Phase 9) |
|---|---|---|---|---|---|
| `preprocess_face()` | P3 (`phase03_vision_webcam_scaled.py`) | P6 (axis shorthand), P7 (CLAHE+112×112+CUBIC) | — | — | `main_video.py:preprocess_face()` |
| `predict_face_emotion()` | P3 (`phase03_vision_webcam_scaled.py`) | P6 (×0.55 scaling), P7 (scaling removed) | — | — | Inlined into batch prediction loop |
| `get_emotion_scores()` | P3 (`phase03_vision_webcam_scaled.py`) | — | P6 (renamed to `predict_face_emotion`) | — | — |
| `extract_features()` (audio) | P2 (`phase02_audio_lstm_trainer.py`) | P3 (added `.T` transpose), P6 (added delta stacking → renamed `extract_audio_features`) | P7 (renamed `process_audio`, deltas removed) | — | `main_audio.py:get_features_fast()` |
| `predict_emotion()` (audio) | P3 (`phase03_audio_live_vosk.py`) | P3/`phase03_audio_push_to_talk.py` (added sorted confidence printout) | P6 (renamed `predict_voice_emotion`) | — | TFLite `_predict_tflite()` |
| `heartbeat_bar()` | P3 (`phase03_audio_live_vosk.py`) | P3/`phase03_audio_push_to_talk.py` (identical) | P6 (no terminal UI in API) | Never | — |
| `merged_voice_label()` | P6 (`phase06_fusion_api_monolith.py`) | — | P7 (7-class model, merge at training time) | Never | — |
| `AttentionLayer` | P4 (`phase04_text_attention_tester.py`) | P4/`phase04_text_negation_engine.py` (identical copy), P7 (`K.tanh`/`K.dot`, shape change) | — | — | `main_text.py` (`ops.tanh`/`ops.matmul`, Keras 3) |
| `is_context_clear()` | P4 (`phase04_text_negation_engine.py`) | — | P4 (`phase04_text_api_bilstm_keyword.py`, replaced by `sentence_has_emotion`) | Never | — |
| `rewrite_sentence()` | P4 (`phase04_text_negation_engine.py`) | — | P7 (removed entirely) | Never | — |
| `NEGATION_MAP` | P4 (`phase04_text_negation_engine.py`) | — | P7 (removed entirely) | Never | — |
| `sentence_has_emotion()` | P4 (`phase04_text_api_bilstm_keyword.py`) | — | P7 (removed entirely) | Never | — |
| `rawemotionwords` list | P4 (`phase04_text_api_bilstm_keyword.py`) | — | P7 (removed entirely) | Never | — |
| `predict_emotion()` (text) | P4 (`phase04_text_negation_engine.py`) | P4/`phase04_text_cnn_predictor.py` (CNN version) | — | — | `compute_inference()` `@tf.function` |
| `predict()` (text simple) | P4 (`phase04_text_attention_tester.py`) | P4 (`phase04_text_bilstm_attention_v1.py`, batch support) | P4/`phase04_text_negation_engine.py` (expanded version) | — | — |
| `extract_largest_face()` | P7 (`phase07_vision_api_standalone.txt`) | — | — | — | Inlined with MediaPipe Tasks API |
| `compute_vision_inference()` | P9 (`main_video.py`) | — | — | — | Current (born in P9) |
| `compute_inference()` (text) | P9 (`main_text.py`) | — | — | — | Current (born in P9) |
| `fuse_emotions()` | P9 (`orchestrator_v3.py`) | — | — | — | Current (born in P9) |
| `_extract_audio_from_video()` | P9 (`orchestrator_v3.py`) | — | — | — | Current (born in P9) |

---

## 2. Configuration Constant Lineage

| Constant | P2 | P3 | P4 | P6 | P7 | P8 | P9 |
|---|---|---|---|---|---|---|---|
| Audio sample rate | 22050 (librosa default) | 16000 (Vosk) | — | 22050 (BUG) | 16000 (fixed) | 16000 | 16000 |
| MFCC bands (`N_MFCC`) | 40 | 40 | — | 40 | 40 | 40 | 40 |
| MFCC pad length (`MAX_PAD_LEN`) | 174 | 174 | — | 174 | 174 | 174 | 174 |
| Face input size | 48×48 | 48×48 | — | 48×48 | 112×112 | 112×112 | 112×112 |
| Face detection method | — | Haar | — | Haar | MediaPipe 0.6 | MediaPipe 0.75 | MediaPipe 0.75 |
| Face prob scaling | — | ×0.55 | — | ×0.55 (face), ×0.66 (voice) | None | None | None |
| Text `max_len` | — | — | 50/100/60 | — | 100 | 100 | 100 |
| Text `max_words` | — | — | 20,000 | — | Unknown | 30,000 | 30,000 |
| Text confidence threshold | — | — | — | — | 0.40 | 0.50 + 0.15 gap | 0.50 + 0.15 gap |
| Audio file size limit | — | — | — | None | 10 MB | 50 MB | 50 MB |
| Video file size limit | — | — | — | None | None | 250 MB | 250 MB |
| CORS | — | — | — | None | Audio only | All services | All services |

---

## 3. Architectural Pattern Lineage

| Pattern | Born | Died | Replacement |
|---|---|---|---|
| Hardware-locked desktop app | P3 (`sounddevice`, `keyboard`, `cv2.VideoCapture(0)`) | P6 (FastAPI transition) | HTTP `UploadFile` |
| Monolithic multi-model process | P6 (`phase06_fusion_api_monolith.py`, `phase06_fusion_api_json_variant.py`) | P7 (service separation) | 4 independent microservices |
| PyInstaller `.exe` bundling | P6 (`phase06_fusion_api_pyinstaller_spec.spec`, `phase06_fusion_api_json_variant.py`, `phase06_vision_api_pyinstaller_variant.txt`) | P6 (abandoned) | `start_servers.bat` multi-worker |
| Haar Cascade face detection | P3 (`phase03_vision_webcam_scaled.py`, `phase03_vision_webcam_early.py`) | P7 (MediaPipe) | MediaPipe Tasks Vision API |
| `pd.get_dummies()` one-hot labels | P2 (`phase02_audio_lstm_trainer.py`) | P2 (training only) | `INT_TO_EMOTION` dict + sparse categorical |
| MFCC delta/delta2 stacking | P6 (`phase06_fusion_api_monolith.py`) | P7 (MFCC only) | MFCC only (BiLSTM captures dynamics) |
| Manual negation dictionary | P4 (`phase04_text_negation_engine.py`) | P7 (removed) | Model handles negation natively |
| Keyword-based context filter | P4 (`phase04_text_negation_engine.py`) | P7 (removed) | Dual-threshold confidence filter |
| Vosk offline speech recognition | P3 (`phase03_audio_live_vosk.py`, `phase03_audio_push_to_talk.py`) | P6 (no transcription) | Groq Whisper API (P9 orchestrator) |
| `model.predict()` eager execution | P3 (all scripts) | P8 | `@tf.function` compiled graph |
| Per-item prediction loops | P3-P7 (video frames, text sentences) | P8 | Batch tensor stacking |
| Disk-based audio processing | P6 (`NamedTemporaryFile`) | P8 | `soundfile.read(BytesIO())` + FFmpeg RAM pipes |
| `nltk.download()` at runtime | P7 (`phase07_text_api_standalone.txt`) | P9 | `setup_nltk.py` offline pre-download |
| Arbitrary probability scaling (×0.55, ×0.66) | P3 (`phase03_vision_webcam_scaled.py`), P6 (`phase06_fusion_api_monolith.py`) | P8 | Raw softmax + power sharpening (×1.5) |

---

## 4. Orphaned Logic (Code That Disappeared Without Replacement)

| Orphaned Entity | Last Seen | Explanation |
|---|---|---|
| `PorterStemmer` import | P4 (`phase04_text_attention_tester.py`) | Imported, assigned to `stemmer`, never called. Stemming was considered but abandoned because the tokenizer was trained on unstemmed text. |
| `heartbeat_bar()` | P3 (`phase03_audio_live_vosk.py`, `phase03_audio_push_to_talk.py`) | Terminal-only volume visualization. No equivalent exists in HTTP APIs — there is no terminal to draw bars in. |
| `keyboard` library | P3 (`phase03_audio_push_to_talk.py`) | Push-to-talk control. Replaced by Flutter's native microphone capture, not by any server-side equivalent. |
| `sounddevice` library | P3 (`phase03_audio_live_vosk.py`, `phase03_audio_push_to_talk.py`) | Hardware microphone capture. Replaced by HTTP file upload, not by any server-side equivalent. |
| `vosk` library | P3 (`phase03_audio_live_vosk.py`, `phase03_audio_push_to_talk.py`) | Offline speech-to-text. Replaced by Groq Whisper in the orchestrator, but the usage patterns are completely different (offline streaming vs. cloud batch). |
| `DURATION = 5` variable | P3 (`phase03_audio_live_vosk.py`) | Defined but never referenced. Evidence of an abandoned fixed-duration recording mode. |
| ANSI color `COLOR` dict | P3 (`phase03_audio_push_to_talk.py`) | Terminal color coding per emotion. No equivalent in JSON APIs. |
| `JSONResponse` import | P6 (`phase06_fusion_api_monolith.py`) | Imported but never used. FastAPI auto-converts dicts to JSON. |
| `word_tokenize` import | P4 (`phase04_text_attention_tester.py`) | Imported from NLTK but never called. Only `sent_tokenize` was used. |
| `time` import | P3 (`phase03_audio_live_vosk.py`) | Imported but never called. Likely intended for timing measurements. |
| `representative_dataset()` | P5 (`phase05_convert_tflite_attempt.py`) | TFLite calibration generator defined but never called. |
| `model_func()` | P5 (`phase05_convert_tflite_attempt.py`) | Invalid `@tf.function` wrapping of `interpreter.invoke()`. Never executed. |
| `cv2.putText` emotion overlay | P3 (`phase03_vision_webcam_early.py`) | Direct window overlay abandoned when moving to headless APIs. |

---

## 5. Model File Lineage

| Model File | Born | Format | Accuracy | Status |
|---|---|---|---|---|
| `emotion_model.h5` | P1 (`phase01_vision_cnn_trainer.py`) | HDF5 | ~57% | Superseded |
| `face_emotion_model.h5` (48×48) | P1 (`cnn model/`) | HDF5 | 57% | Superseded |
| `face_emotion_model.h5` (7.5MB, `emotion_api/`) | P6 | HDF5 | Unknown (different model?) | Superseded |
| `speech_emotion_model_7.h5` | P2 (`phase02_audio_lstm_trainer.py`) | HDF5 | Unknown | Superseded |
| `final_lstm_model.h5` | P6 (`emotion_api/`) | HDF5 | Unknown | Superseded by TF2.12 fix |
| `final_lstm_model_tf212.h5` | P6 (`emotion_api/`) | HDF5 | Unknown | Superseded |
| `voice_model_tf212_FIXED.h5` | P6 (`emotion_api/`) | HDF5 | Unknown | Superseded |
| `emotion_bilstm_model.h5` | P4 (`phase04_text_bilstm_trainer.py`) | HDF5 | Unknown (5 epochs) | Superseded |
| `emotion_TEXT_cnn_model.h5` | P4 (external) | HDF5 | Unknown | Superseded |
| `emotion_model_tf212_fixed.h5` | P4 (`phase04_text_negation_engine.py`) | HDF5 | Unknown | Superseded |
| `fer_best_model.keras` | P7+ (Kaggle) | Keras 3 | **81.03%** | **ACTIVE** (Vision) |
| `audio_best_model.keras` | P8 (Kaggle) | Keras 3 | **94.10%** | Converted to TFLite |
| `audio_model.tflite` | P8 (`convert_audio_model.py`) | TFLite float16 | 94.10% | **ACTIVE** (Audio) |
| `text_best_model.keras` | P8 (Kaggle) | Keras 3 | **94.04%** | **ACTIVE** (Text) |

---

## 6. Port Assignment Lineage

| Service | P6 | P7 Audio | P7 Video | P7 Text | P8 | P9 |
|---|---|---|---|---|---|---|
| Unified/Audio | 8000 | 8002 | — | — | 8000 | 8000 |
| Video | — | — | 8002 | — | 8002 | 8002 |
| Text | — | — | — | 8001 | 8001 | 8001 |
| Orchestrator | — | — | — | — | — | 8003 |

Note: In Phase 7, both Audio and Video used port 8002 (they were never meant to run simultaneously — they were separate experiments). Phase 9 resolved this by giving Audio port 8000 and Video port 8002.
