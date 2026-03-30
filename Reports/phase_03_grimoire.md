# Phase 3 Grimoire: Complete Technical Archaeology

---

## Header 1: Complete File Inventory

| File | Size (bytes) | Role |
|---|---|---|
| `FYP old/phase03_vision_webcam_scaled.py` | 3,004 | Live webcam face emotion detector |
| `FYP old/phase03_vision_webcam_early.py` | 1,427 | Earliest live webcam face emotion prototype |
| `FYP old/phase03_audio_live_vosk.py` | 3,341 | Continuous speech emotion + Vosk subtitles |
| `FYP old/phase03_audio_push_to_talk.py` | 4,389 | Push-to-talk speech emotion with spacebar |

---

## Header 2: Line-by-Line Logic Migration

### File: `phase03_vision_webcam_early.py` (Earliest Concept)

This 39-line script is the **absolute earliest face detection prototype**. 
- Loads `emotiondetector1.json` + `emotiondetector1.h5` instead of `face_emotion_model.h5`.
- Uses a tiny `{0:'angry'...6:'surprise'}` label dictionary.
- Resizes directly to `(48,48)` and reshapes to `(1,48,48,1)` before `/255.0` normalization.
- Uses `cv2.rectangle` and `cv2.putText` to overlay the top emotion directly on the webcam window.
- Catches and silently ignores `cv2.error`.
- **Significance**: Proves the initial concept of running a CNN on webcam frames via Haar Cascades, setting the stage for `phase03_vision_webcam_scaled.py` and the FastAPI monoliths.

---

### File: `phase03_vision_webcam_scaled.py` (Complete Source)

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
model = load_model("face_emotion_model.h5")
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# ---------------------------------------------------------
# LOAD CASCADE
# ---------------------------------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------------------------------------------------
# PREPROCESS FACE
# ---------------------------------------------------------
def preprocess_face(gray_face):
    face = cv2.resize(gray_face, (48, 48))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)
    return face


# ---------------------------------------------------------
# PREDICT EMOTIONS WITH PROBABILITIES × 0.55
# ---------------------------------------------------------
def get_emotion_scores(gray_face):
    processed_face = preprocess_face(gray_face)

    raw_pred = model.predict(processed_face, verbose=0)[0]
    scaled_pred = raw_pred * 0.55

    # Convert to dictionary
    emotion_probs = {
        emotion_labels[i]: float(scaled_pred[i])
        for i in range(len(scaled_pred))
    }

    # Pick top emotion
    top_idx = np.argmax(scaled_pred)
    top_emotion = emotion_labels[top_idx]
    top_conf = scaled_pred[top_idx] * 100

    return top_emotion, top_conf, emotion_probs


# ---------------------------------------------------------
# WEBCAM LOOP
# ---------------------------------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]

        # Get predictions
        top_emotion, top_conf, prob_dict = get_emotion_scores(face_gray)

        # Draw bounding box + top emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{top_emotion} ({top_conf:.2f}%)",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

        # Show all scaled probabilities
        start_y = y + h + 20
        for emo, prob in prob_dict.items():
            txt = f"{emo}: {prob*100:.2f}%"
            cv2.putText(frame, txt, (x, start_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)
            start_y += 25

    cv2.imshow("Emotion Detection (Webcam)", frame)

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

#### Block 1: Model Loading (Lines 1-9)
- **What this solves**: Loading the Phase 1 CNN model for real-time inference.
- **`face_emotion_model.h5`**: This is the 48×48 CNN from Phase 1 (or the `cnn model/` notebook), NOT the later 112×112 FERPlus model.
- **Label ordering**: Title-case, manually ordered as `['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']`. This ordering does NOT match sklearn's alphabetical convention (which would put Neutral before Sad). This is a hardcoded assumption about the training label order. If the training used `flow_from_directory` (which sorts alphabetically), the correct order would be `['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']`. This potential mismatch was a latent bug.
- **Phase 9 resolution**: `main_video.py` uses `INT_TO_EMOTION = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'neutral', 5:'sad', 6:'surprise'}` which is explicitly alphabetical and matches the Kaggle training notebook's `sorted(os.listdir())` ordering.

#### Block 2: Haar Cascade (Lines 14-16)
- **What this solves**: Face detection — locating the rectangular region of an image containing a face.
- **`haarcascade_frontalface_default.xml`**: An ancient (2001) machine learning algorithm based on Haar-like features and Adaboost cascade classifiers. It runs fast but has high false-positive rates.
- **Logical flaw — False positives**: Haar cascades detect patterns of light/dark regions characteristic of faces, but similar patterns appear in textured wallpaper, electrical outlets, pet faces, and book covers. Every false detection sends garbage pixel data into the emotion model, producing random predictions that undermine user trust.
- **Phase 7 replacement**: `phase07_vision_api_standalone.txt` replaces Haar with `mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)`.
- **Phase 9 replacement**: `main_video.py` uses the MediaPipe Tasks Vision API (`vision.FaceDetector`) with `min_detection_confidence=0.75`.

#### Block 3: Preprocessing (Lines 21-26)
- **`cv2.resize(gray_face, (48, 48))`**: Matches the Phase 1 CNN's input size. No CLAHE, no INTER_CUBIC interpolation (uses default bilinear).
- **`face.astype("float32") / 255.0`**: Standard [0,1] normalization. This pattern survives unchanged into Phase 9.
- **`np.expand_dims(face, axis=0)` then `axis=-1`**: Reshapes from `(48,48)` to `(1,48,48,1)` — batch dimension + channel dimension. The double expand_dims is equivalent to `np.reshape(face, (1,48,48,1))`. Phase 9 uses a single reshape call.

#### Block 4: Prediction with 0.55 Scaling (Lines 32-50)
- **`raw_pred = model.predict(processed_face, verbose=0)[0]`**: Synchronous, eager-mode prediction. `verbose=0` suppresses the progress bar.
- **`scaled_pred = raw_pred * 0.55`**: An arbitrary calibration factor that reduces all probability values by 45%. The comment says "PREDICT EMOTIONS WITH PROBABILITIES × 0.55" but provides no mathematical justification. This scaling was likely an empirical attempt to make confidence scores "feel" more realistic — raw softmax outputs often cluster near 0.9+ for the top class, which seemed "overconfident" during demos. The scaling factor varies across phases: `phase03_vision_webcam_scaled.py` uses 0.55, `phase06_fusion_api_old_endpoints.txt` uses 0.65, `phase06_fusion_api_monolith.py` uses 0.55 for face and 0.66 for voice. All scaling was removed in Phase 9 — the raw softmax output is reported directly.

#### Block 5: Webcam Loop (Lines 55-92)
- **`cv2.VideoCapture(0)`**: Opens the default camera device. The `0` is a hardware device index, not a file path.
- **`face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)`**: Runs the Haar cascade at multiple scales. `scaleFactor=1.3` means each scale is 30% larger than the previous (coarse but fast). `minNeighbors=5` requires 5 overlapping detections to confirm a face (reduces false positives but misses smaller faces).
- **Per-face loop with `model.predict` per face**: If 3 faces are detected in a frame, `model.predict` is called 3 times sequentially. No batching. At ~30fps, this means 90 predict calls per second, which is catastrophically slow.
- **Drawing on frame**: Bounding boxes (`cv2.rectangle`), top emotion label (`cv2.putText`), and full probability breakdown are drawn directly onto the OpenCV frame. This is purely visual — no data is persisted or transmitted.
- **`cv2.waitKey(1) & 0xFF == ord("q")`**: Polls for keyboard input every 1ms. 'Q' to quit. This is standard OpenCV practice for desktop applications.
- **Hardware coupling**: `cv2.VideoCapture(0)`, `cv2.imshow()`, `cv2.waitKey()` all require a physical display and camera. None of these exist on a headless server.

---

### File: `phase03_audio_live_vosk.py` (Complete Source)

```python
import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import sys
import time
import queue
from vosk import Model, KaldiRecognizer
import json

# Load trained model
model = tf.keras.models.load_model("speech_emotion_model_7.h5")

# Emotion labels (must match training order)
EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust']

# Audio settings
SAMPLE_RATE = 16000   # Vosk works best with 16kHz
DURATION = 5
N_MFCC = 40

# Load Vosk model for live speech recognition
vosk_model = Model("vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)

def extract_features(audio, sample_rate, max_pad_len=174):
    """Extract MFCCs keeping time dimension (for CNN/RNN input)"""
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)

    # Pad or truncate to fixed length (174 frames)
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]

    return mfccs.T   # shape (174, 40)


def predict_emotion(audio):
    """Predict emotion from audio"""
    features = extract_features(audio, SAMPLE_RATE)  # (174,40)
    features = np.expand_dims(features, axis=0)      # (1,174,40)
    prediction = model.predict(features, verbose=0)[0]
    return EMOTIONS[np.argmax(prediction)]


def heartbeat_bar(block):
    volume_norm = np.linalg.norm(block) * 10
    bar_length = int(min(volume_norm, 50))
    if bar_length < 15:
        color = "\033[92m"
    elif bar_length < 30:
        color = "\033[93m"
    else:
        color = "\033[91m"
    bar = "█" * bar_length
    return f"{color}[{bar:<50}]\033[0m"

print("🎤 Real-time Speech Emotion Recognition + Live Subtitles Started")
print("Press Ctrl+C to stop.\n")

q = queue.Queue()

def callback(indata, frames, time_, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize = 8000, device=None,
                       dtype="int16", channels=1, callback=callback):
    full_audio = []
    print("🎙 Listening...\n")
    try:
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text.strip():
                    print(f"\n📝 {text}")
            else:
                partial = json.loads(recognizer.PartialResult())
                if partial.get("partial", ""):
                    sys.stdout.write(
                        "\r" + heartbeat_bar(np.frombuffer(data, dtype=np.int16).astype(np.float32)) +
                        " 📝 " + partial["partial"]
                    )
                    sys.stdout.flush()
            # store audio for emotion later
            full_audio.append(np.frombuffer(data, dtype=np.int16).astype(np.float32))

    except KeyboardInterrupt:
        print("\n\n✅ Recording stopped, analyzing emotion...")
        audio = np.concatenate(full_audio)
        emotion = predict_emotion(audio)
        print(f"\nPredicted Emotion: 😃 {emotion}")
```

#### Block 1: Model and Label Loading (Lines 1-15)
- **`EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust']`**: This label list has 7 entries including "calm" as a separate class. However, `phase02_audio_lstm_trainer.py` merged calm into neutral during training, so the model's output layer has 7 nodes but "calm" and "neutral" share the same training distribution. If the model predicts index 1 ("calm" in this list), it's predicting a class that was trained as "neutral" — effectively duplicating the neutral prediction under a different name. The Phase 6 monolith addresses this with `merged_voice_label()`, but `phase03_audio_live_vosk.py` does not.
- **`DURATION = 5`**: Defined but never used. Dead variable — evidence of an abandoned fixed-duration recording mode.

#### Block 2: Feature Extraction (Lines 26-37)
- **Identical to `phase02_audio_lstm_trainer.py`'s `extract_features`** with one key difference: the returned shape is `mfccs.T` (transposed), yielding `(174, 40)` — ready for LSTM input without the separate reshape step needed in `phase02_audio_lstm_trainer.py`.
- **Still no trimming or normalization**: The "False Angry" bias persists.

#### Block 3: Vosk Speech Recognition (Lines 22-24, 65-89)
- **`vosk.Model("vosk-model-small-en-us-0.15")`**: A small, offline-capable speech recognition model. This is the English-US variant at 40MB.
- **`KaldiRecognizer(vosk_model, SAMPLE_RATE)`**: Initializes a Vosk recognizer at 16kHz.
- **`recognizer.AcceptWaveform(data)`**: Feed raw audio chunks to Vosk. Returns `True` when a complete utterance is detected.
- **Vosk was abandoned entirely**: No phase after Phase 3 uses Vosk. The Phase 9 orchestrator uses Groq's Whisper API for transcription when needed.

#### Block 4: The Unbounded Memory Array (Lines 91)
- **`full_audio.append(np.frombuffer(data, dtype=np.int16).astype(np.float32))`**: Every 8000-sample audio chunk (0.5 seconds at 16kHz) is appended to `full_audio` indefinitely. If the user records for 10 minutes, this list contains 1,200 chunks of 8,000 float32 values = ~38.4 MB. For an hour recording: ~230 MB. There is no upper bound, no circular buffer, no chunked processing. This is a guaranteed memory leak leading to eventual crash.
- **Phase 9 resolution**: Audio is received as a single `UploadFile` with a `MAX_FILE_SIZE_MB = 50` guard. No unbounded accumulation.

#### Block 5: Keyboard Interrupt Pattern (Lines 93-97)
- **Emotion is predicted ONLY after Ctrl+C**: The user must physically interrupt the program to trigger `predict_emotion()`. During recording, no emotion feedback is provided — only Vosk subtitles and the heartbeat bar. This makes the system feel unresponsive for emotion detection.
- **`np.concatenate(full_audio)`**: Concatenates all accumulated chunks into a single massive array, then runs a single MFCC extraction + prediction. For long recordings, this produces an MFCC matrix far wider than the `max_pad_len=174` truncation, meaning only the first ~5.5 seconds of audio are actually analyzed. The rest is silently discarded.

---

### File: `phase03_audio_push_to_talk.py` (Complete Source)

```python
import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import sys
import time
import queue
import json
import keyboard
from vosk import Model, KaldiRecognizer

# Load emotion model
model = tf.keras.models.load_model("speech_emotion_model_7.h5")

EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust']

SAMPLE_RATE = 16000
N_MFCC = 40
MIN_SPEECH_VOLUME = 1500

# Load Vosk model
vosk_model = Model("vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)

q = queue.Queue()


# ===== FEATURE EXTRACTION =====
def extract_features(audio, sample_rate, max_pad_len=174):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
    if mfccs.shape[1] < max_pad_len:
        mfccs = np.pad(mfccs, ((0,0),(0, max_pad_len - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs.T


# ===== EMOTION PREDICTION =====
def predict_emotion(audio):
    features = extract_features(audio, SAMPLE_RATE)
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features, verbose=0)[0]

    sorted_indices = np.argsort(prediction)[::-1]

    print("\n📊 **Confidence Scores:**")
    for idx in sorted_indices:
        print(f"   {EMOTIONS[idx]:<10} → {prediction[idx]:.3f}")

    return EMOTIONS[np.argmax(prediction)]


# ===== HEARTBEAT BAR =====
def heartbeat_bar(block):
    volume_norm = np.linalg.norm(block) * 10
    bar_length = int(min(volume_norm, 50))

    if bar_length < 15:
        color = "\033[92m"
    elif bar_length < 30:
        color = "\033[93m"
    else:
        color = "\033[91m"

    bar = "█" * bar_length
    return f"{color}[{bar:<50}]\033[0m"


def callback(indata, frames, time_, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))


print("🎤 Press & Hold SPACEBAR to Speak")
print("==============================================\n")

with sd.RawInputStream(
    samplerate=SAMPLE_RATE,
    blocksize=8000,
    dtype="int16",
    channels=1,
    callback=callback
):
    while True:

        print("\n⏳ Waiting for SPACEBAR... (Hold to record)")
        keyboard.wait("space")

        print("\n🎙 Recording started... (release SPACEBAR to stop)\n")

        full_audio = []
        final_subtitle = ""

        # RECORD WHILE SPACE IS HELD
        while keyboard.is_pressed("space"):
            data = q.get()
            block_float = np.frombuffer(data, dtype=np.int16).astype(np.float32)

            # Live subtitles
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text.strip():
                    final_subtitle += " " + text
                    print(f"\n📝 {text}")
            else:
                partial = json.loads(recognizer.PartialResult())
                if partial.get("partial", ""):
                    sys.stdout.write(
                        "\r" + heartbeat_bar(block_float) +
                        " 📝 " + partial["partial"]
                    )
                    sys.stdout.flush()

            full_audio.append(block_float)

        print("\n⏹ SPACEBAR released — Processing...")

        audio = np.concatenate(full_audio)

        # ===== NO SPEECH DETECTED =====
        if np.linalg.norm(audio) < MIN_SPEECH_VOLUME:
            print("\n🔇 No speech detected — Try again.\n")
            continue

        # ===== DISPLAY FINAL FULL SUBTITLES =====
        if final_subtitle.strip():
            print(f"\n📝 **Full Subtitle:** \"{final_subtitle.strip()}\"")

        else:
            print("\n📝 No subtitle recognized.")

        # ===== EMOTION PREDICTION =====
        emotion = predict_emotion(audio)

        COLOR = {
            "happy": "\033[92m",
            "neutral": "\033[96m",
            "calm": "\033[94m",
            "sad": "\033[94m",
            "angry": "\033[91m",
            "fearful": "\033[95m",
            "disgust": "\033[93m"
        }

        print(f"\n🎭 Predicted Emotion: {COLOR.get(emotion,'')}{emotion.upper()}\033[0m")

        print("\n�4 Ready for next input... (Hold SPACEBAR again)\n")
```

#### Improvements over `phase03_audio_live_vosk.py`
- **`keyboard.wait("space")` + `keyboard.is_pressed("space")`**: Replaces the Ctrl+C interrupt pattern with push-to-talk. Recording starts when spacebar is held and stops when released. This is a significant UX improvement — the user controls recording duration.
- **`MIN_SPEECH_VOLUME = 1500`**: A volume threshold check using `np.linalg.norm(audio)`. If the total audio energy is below 1500, the prediction is skipped ("No speech detected"). This prevents the model from hallucinating emotions from background noise. This concept evolved into Phase 7's `CONFIDENCE_THRESHOLD = 0.40` and Phase 9's dual-threshold filtering.
- **Confidence score printout**: `predict_emotion()` now prints ALL class probabilities sorted by confidence, not just the top prediction. This debugging output helped identify the "False Angry" bias during development.
- **Per-emotion terminal colors**: The `COLOR` dict maps each emotion to an ANSI color code for visual impact in the terminal.
- **Subtitle accumulation**: `final_subtitle` concatenates Vosk transcription results across the recording session, then displays the full transcript after spacebar release.

#### Persistent Flaws from `phase03_audio_live_vosk.py`
- Same unbounded `full_audio.append()` memory leak.
- Same lack of MFCC normalization.
- Same `model.predict()` eager execution.
- Same 7-class label list with "calm" as a separate entry.
- Same hardware coupling (sounddevice + keyboard).

---

## Header 3: Micro-Decision Log

| Decision | `phase03_audio_live_vosk.py` Value | `phase03_audio_push_to_talk.py` Value | Phase 9 Value | Rationale |
|---|---|---|---|---|
| Recording trigger | Ctrl+C (interrupt) | Spacebar (push-to-talk) | HTTP `UploadFile` | Progressive decoupling from hardware |
| Volume gating | None | `np.linalg.norm(audio) < 1500` | `CONFIDENCE_THRESHOLD` on model output | Moved from input-level to output-level filtering |
| Confidence display | Top emotion only | All 7 sorted | JSON probabilities dict | Structured data for programmatic consumption |
| Transcription engine | Vosk (offline) | Vosk (offline) | Groq Whisper API (cloud) | Higher accuracy, no local model weight |
| Face detection method | Haar Cascade (`phase03_vision_webcam_scaled.py`) | N/A | MediaPipe Tasks Vision API | Eliminated false positives |
| Label list | 7 entries (includes 'calm') | 7 entries (includes 'calm') | 7 entries (calm merged into neutral at training time) | Label mapping aligned with training |
| Probability scaling | N/A | Raw softmax | Raw softmax | 0.55/0.65 scaling abandoned |

---

## Header 4: Inter-Phase Diff Analysis

### Files that disappeared after Phase 3

| Phase 3 File | Where did its logic migrate? |
|---|---|
| `phase03_vision_webcam_scaled.py` webcam loop | Replaced by FastAPI `/predict/image` and `/predict/video` endpoints (Phase 6 onward) |
| `phase03_audio_live_vosk.py` continuous recording | Replaced by single-file upload `/predict_audio` endpoint (Phase 7 onward) |
| `phase03_audio_push_to_talk.py` push-to-talk | Replaced by Flutter's native microphone capture → HTTP upload → `/predict_audio` |
| Vosk speech recognition | Replaced by Groq Whisper transcription in Phase 9 orchestrator |
| `heartbeat_bar()` function | No descendant. Terminal-only visualization with no server equivalent. |
| `keyboard` library | No descendant. Hardware dependency eliminated. |
| `sounddevice` library | No descendant. Hardware dependency eliminated. |
| ANSI color `COLOR` dict | No descendant. JSON APIs have no terminal colors. |

### Concepts that survived into Phase 9

| Phase 3 Concept | Phase 9 Descendant |
|---|---|
| `extract_features()` (MFCC extraction + pad/truncate) | `get_features_fast()` in `main_audio.py` |
| `preprocess_face()` (resize + normalize + reshape) | `preprocess_face()` in `main_video.py` (112×112, CLAHE added) |
| `model.predict(features)` | `compute_vision_inference(tensor)` / `compute_inference(tensor)` (`@tf.function` compiled) |
| `MIN_SPEECH_VOLUME` threshold | `CONFIDENCE_THRESHOLD` on prediction output |
| `full_audio → np.concatenate → predict` | `file_bytes → soundfile.read → get_features_fast → predict` |
