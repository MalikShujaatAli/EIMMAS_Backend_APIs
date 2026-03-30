# Phase 8 Grimoire: Pre-Production APIs (The Bridge)

---

## Strategic Context
Phase 8 represents the critical bridge between the "2nd attempt" separated APIs (Phase 7) and the production `services/` architecture (Phase 9). These four scripts (`phase08_audio_api_preprod.py`, `phase08_vision_api_preprod.py`, `phase08_text_api_preprod.py`, `phase08_text_api_intermediate.py` — all in `FYP old/`) are the **immediate predecessors** to the production files. They introduced several key innovations (in-memory audio, MediaPipe Tasks API, Keras 3 ops, prediction sharpening, dual-threshold) but still lacked the concurrency optimizations (`asyncio.to_thread`, `@tf.function`) that define Phase 9.

---

## Header 1: Complete File Inventory

| File | Size | Lines | Role |
|---|---|---|---|
| `FYP old/phase08_audio_api_preprod.py` | 6,327 | 179 | Pre-production Audio API |
| `FYP old/phase08_vision_api_preprod.py` | 6,736 | 179 | Pre-production Vision API |
| `FYP old/phase08_text_api_preprod.py` | 6,412 | 190 | Pre-production Text API |
| `FYP old/phase08_text_api_intermediate.py` | 5,273 | 140 | Intermediate text API variant |

---

## Header 2: Key Evolutionary Leaps in Phase 8

### `phase08_audio_api_preprod.py` — What changed from Phase 7

| Feature | Phase 7 (`phase07_audio_api_standalone.txt`) | Phase 8 (`phase08_audio_api_preprod.py`) | Phase 9 (`services/audio_api/phase08_audio_api_preprod.py`) |
|---|---|---|---|
| Audio I/O | `NamedTemporaryFile` → `librosa.load(path)` | `soundfile.read(BytesIO(bytes))` — **zero disk** | Same as 7.5 |
| Inference method | `model.predict(tensor, verbose=0)` | `model(tensor, training=False)` — **direct TF call** | TFLite `interpreter.invoke()` |
| Warmup | None | `model(dummy_input)` on boot | Same concept, TFLite version |
| INT_TO_EMOTION | Missing (used `label_encoder.pkl`) | `{0:'angry', 1:'disgust', ..., 6:'surprised'}` | Same |
| Z-score normalization | Missing | **Missing** | `(mfccs - mean) / (std + 1e-9)` |
| `asyncio.to_thread()` | Missing | **Missing** | Present |
| TFLite | Missing | **Missing** | Primary inference engine |
| File size limit | 10 MB | 10 MB | 50 MB |

### `phase08_vision_api_preprod.py` — What changed from Phase 7

| Feature | Phase 7 (`phase07_vision_api_standalone.txt`) | Phase 8 (`phase08_vision_api_preprod.py`) | Phase 9 (`services/image_video_api/phase08_vision_api_preprod.py`) |
|---|---|---|---|
| Face detector API | `mp.solutions.face_detection` (legacy) | `mediapipe.tasks.vision.FaceDetector` (Tasks API) | Same as 7.5 |
| Model download | Manual placement | `urllib.request.urlretrieve(MP_MODEL_URL)` — **auto-download** | Same |
| Zip handling | None | `zipfile.ZipFile(ZIPPED_MODEL_PATH)` auto-extract | Same concept but for different artifact |
| Confidence | 0.6 | **0.75** | Same (0.75) |
| Warmup | None | `emotion_model(np.zeros(...))` on boot | `@tf.function` warmup |
| Frame sampling | `frame_id % 10` | `frame_id % 10` — **still not FPS-aware** | FPS-aware `frame_idx % max(1, int(fps))` |
| Batch prediction | No | **No** — still per-frame `analyze_emotion()` | Yes — `np.stack()` + batch `compute_vision_inference()` |
| Temp file naming | `NamedTemporaryFile` | `f"v_temp_{file.filename}"` — **collision risk** | `eimmas_{uuid4}.mp4` |
| `@tf.function` | No | No | Yes with `reduce_retracing=True` |

### `phase08_text_api_preprod.py` — What changed from Phase 7

| Feature | Phase 7 (`phase07_text_api_standalone.txt`) | Phase 8 (`phase08_text_api_preprod.py`) | Phase 9 (`services/text_api/phase08_text_api_preprod.py`) |
|---|---|---|---|
| Attention ops | `K.tanh`/`K.dot`/`K.softmax`/`K.sum` | `ops.tanh`/`ops.matmul`/`ops.softmax`/`ops.sum` — **Keras 3!** | Same as 7.5 |
| Serialization | `@tf.keras.utils.register_keras_serializable()` | `@keras.saving.register_keras_serializable(package="Custom", name="AttentionLayer")` — **explicit package** | Same as 7.5 |
| Confidence filter | Single `CONFIDENCE_THRESHOLD = 0.40` | **Dual-threshold**: `max_prob < 0.50 or gap < 0.15` | Same as 7.5 |
| Prediction sharpening | None | `predictions ** 1.5` + re-normalize — **first appearance** | Same as 7.5 |
| Text cleaning | None | `clean_text()` with regex (URLs, HTML, mentions, special chars) | Pre-compiled `re.compile()` versions |
| Short text rejection | None | `len(text.split()) < 3` → return neutral | Same concept |
| Inference | `model.predict(padded_batch)` | `model(padded_seqs, training=False)` — **direct call** | `@tf.function` + `asyncio.to_thread()` |
| NLTK | `nltk.download('punkt')` at boot | `nltk.download('punkt')` at boot — **still runtime** | `setup_nltk.py` offline |
| `@tf.function` | No | No | Yes |
| `asyncio.to_thread()` | No | No | Yes |
| Pre-compiled regex | No | No | Yes |

### `phase08_text_api_intermediate.py` — The Intermediate Text Variant

`phase08_text_api_intermediate.py` sits **between Phase 7's `phase07_text_api_standalone.txt` and Phase 8's `phase08_text_api_preprod.py`**:

| Feature | `phase08_text_api_intermediate.py` | `phase08_text_api_preprod.py` |
|---|---|---|
| Attention ops | `tf.math.tanh` / `tf.linalg.matmul` — **TF ops** | `ops.tanh` / `ops.matmul` — **Keras ops** |
| NLTK package | `punkt_tab` (different!) | `punkt` |
| Confidence filter | Single threshold (0.40) | Dual-threshold (0.50 + 0.15 gap) |
| Prediction sharpening | None | `** 1.5` |
| Text preprocessing | None | `clean_text()` with regex |
| `get_config()` method | Present | Present |

This confirms that `phase08_text_api_intermediate.py` was written BEFORE `phase08_text_api_preprod.py` — it has the new model (`text_best_model.keras`) but not the dual-threshold or sharpening innovations.

---

## Header 3: `phase04_text_bilstm_attention_v1.py` and `phase04_text_api_bilstm_keyword.py` — The Missing Phase 4 Links

### `phase04_text_bilstm_attention_v1.py` (129 lines)
- **Role**: Desktop text emotion predictor with the mature `AttentionLayer` (`K.tanh`/`K.dot` ops, `initializer="normal"`, shape `(input_shape[-1], 1)`).
- **Model**: `emotion_model_20251203_192244.h5` — timestamped name proves this was trained on Dec 3, 2025.
- **Why it matters**: This script proves the `AttentionLayer` evolved from `phase04_text_attention_tester.py`'s `tf.tensordot` to `K.tanh`/`K.dot` BEFORE any API was built. It uses the same attention architecture as `phase07_text_api_standalone.txt` (Phase 7), confirming the lineage.
- **Voting**: Uses `Counter(emotions).most_common(1)[0][0]` — Python's `collections.Counter` for majority voting, simpler than the manual `vote_counter` dict in `phase04_text_negation_engine.py`.

### `phase04_text_api_bilstm_keyword.py` (204 lines)
- **Role**: The **first text emotion FastAPI endpoint** — the missing bridge between Phase 4's desktop scripts and Phase 7's `phase07_text_api_standalone.txt`.
- **Key innovation — `rawemotionwords` list**: A massive 150+ word set covering sadness, fear, disgust, surprise, stress, psychological states, and social/relational cues. The `sentence_has_emotion()` function checks if ANY word in the sentence appears in this set before running the model.
- **Why this replaced `is_context_clear()`**: Instead of Phase 4's rigid rules (word count, character ratio, keyword presence), `phase04_text_api_bilstm_keyword.py` uses a much larger vocabulary of emotion-related words. If none of the 150+ words appear, the sentence is marked "context unclear" without running inference. This is more permissive than `is_context_clear()` but still fundamentally a keyword-matching approach.
- **Why this was ALSO abandoned**: The Phase 9 production `phase08_text_api_preprod.py` removes the keyword check entirely, relying purely on model confidence (dual-threshold). The model's softmax output is a better indicator of emotional content than any keyword list.
- **Port 8001**: Same as Phase 7 and Phase 9 text APIs, confirming the lineage.

---

