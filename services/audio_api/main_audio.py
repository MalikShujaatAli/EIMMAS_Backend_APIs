import os
import io

# ---------------------------------------------------------
# 1. ENVIRONMENT & WARNING SUPPRESSION
# ---------------------------------------------------------
# Must be set BEFORE importing tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import time
import logging
import warnings
import asyncio
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ---------------------------------------------------------
# 2. PROFESSIONAL LOGGING
# ---------------------------------------------------------
warnings.filterwarnings('ignore')

# Automatically locate the central \logs\ directory
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logs"))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "audio_api.log")

# Professional Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 2. CONFIGURATIONS
# ---------------------------------------------------------
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_PAD_LEN = 174
CONFIDENCE_THRESHOLD = 0.40
MAX_FILE_SIZE_MB = 50
MODEL_PATH = "audio_best_model.keras"
TFLITE_PATH = "audio_model.tflite"

INT_TO_EMOTION = {
    0: 'angry', 1: 'disgust', 2: 'fearful', 
    3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprised'
}

# --- TFLITE BOOTSTRAP ---
interpreter = None
input_details = None
output_details = None
audio_model = None

def load_audio_model():
    global interpreter, input_details, output_details, audio_model
    if os.path.exists(TFLITE_PATH):
        try:
            # Removed emoji to prevent Windows console UnicodeEncodeError
            logger.info(f"Loading TFLite Model: {TFLITE_PATH} (Optimized for CPU)")
            
            # Note: BiLSTMs often require Flex Ops. 
            # If this fails, the try-block catches it and falls back to Keras.
            interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            logger.info("TFLite Model loaded successfully.")
            return
        except Exception as e:
            logger.warning(f"TFLite loading failed, falling back to Keras. Error: {e}")
            interpreter = None

    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading heavy TensorFlow model: {MODEL_PATH}")
        audio_model = tf.keras.models.load_model(MODEL_PATH)
    else:
        logger.error(f"Critical Error: No audio model found in {os.getcwd()}.")

load_audio_model()

# ---------------------------------------------------------
# 3. APP INITIALIZATION & MODEL LOAD
# ---------------------------------------------------------
app = FastAPI(title="Production Audio Emotion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Starting Production Audio Emotion API")
logger.info("Loading AI Model into memory...")

def compute_inference(tensor):
    """Infers emotion using either TFLite or Keras model automatically."""
    global interpreter, input_details, output_details, audio_model
    
    if interpreter:
        # TFLite Inference (Flash Fast on CPU)
        interpreter.set_tensor(input_details[0]['index'], tensor)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])[0]
    elif audio_model:
        # Full TF Inference (Slow on CPU)
        return audio_model(tensor, training=False).numpy()[0]
    else:
        raise ValueError("No model loaded.")

# ---------------------------------------------------------
# 4. HIGH-SPEED PREPROCESSING (IN-RAM)
# ---------------------------------------------------------
def get_features_fast(audio_bytes):
    """Processes audio in memory to ensure speed and concurrency safety."""
    try:
        with io.BytesIO(audio_bytes) as buf:
            audio, sr = sf.read(buf)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
        # LIGHTNING SPEED: Bypass librosa's heavy mathematical resampling 
        # if the Orchestrator has already forced 16kHz via FFmpeg.
        if sr != SAMPLE_RATE:
            logger.info(f"Resampling audio from {sr} to {SAMPLE_RATE}...")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        else:
            # Bypass logic to save ~500-800ms of CPU time!
            pass

        # Trim silence
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
        if len(audio_trimmed) == 0:
            return None

        # Extract MFCC
        mfccs = librosa.feature.mfcc(y=audio_trimmed, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        
        # Pad/Truncate
        if mfccs.shape[1] < MAX_PAD_LEN:
            mfccs = np.pad(mfccs, ((0,0), (0, MAX_PAD_LEN - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :MAX_PAD_LEN]
            
        # ACCURACY FIX: Z-score Normalization
        # Most RAVDESS models expect normalized MFCCs. This fixes 'False Angry' bias.
        mean = np.mean(mfccs)
        std = np.std(mfccs) + 1e-9
        mfccs = (mfccs - mean) / std
            
        # Reshape to (1, 174, 40) for the model
        return np.expand_dims(mfccs.T, axis=0).astype(np.float32)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return None

# ---------------------------------------------------------
# 5. CONCURRENT-SAFE API ENDPOINTS
# ---------------------------------------------------------

@app.get("/")
def health_check():
    return {"status": "online", "service": "Audio Emotion API", "port": 8000}

@app.post("/predict_audio")
async def predict_audio(file: UploadFile = File(...)):
    """Receives audio file, processes in RAM, and predicts emotion."""
    start_time = time.time()
    logger.info(f"Receiving audio request: {file.filename}")
    
    try:
        # Security: Size limit check BEFORE reading into memory
        if file.size and file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit.")

        # Read file into memory buffer
        content = await file.read()

        # Preprocess asynchronously to avoid freezing the event loop
        tensor = await asyncio.to_thread(get_features_fast, content)
        if tensor is None:
            raise HTTPException(status_code=400, detail="Audio file is corrupted or silent.")

        # INFERENCE (Flash Fast if TFLite is generated)
        prediction = await asyncio.to_thread(compute_inference, tensor)
        
        # Post-processing
        winning_idx = int(np.argmax(prediction))
        conf = float(prediction[winning_idx])
        
        emotion = INT_TO_EMOTION[winning_idx] if conf >= CONFIDENCE_THRESHOLD else "context unclear"
        
        latency = round(time.time() - start_time, 2)
        logger.info(f"Result: {file.filename} -> {emotion} ({latency}s)")

        return {
            "predicted_emotion": emotion,
            "confidence": round(conf * 100, 2),
            "processing_time": f"{latency}s",
            "probabilities": {INT_TO_EMOTION[i]: round(float(prediction[i]), 4) for i in range(len(INT_TO_EMOTION))}
        }

    except HTTPException as he:
        logger.warning(f"Audio request rejected: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Endpoint prediction crashed: {e}")
        return {"error": "Internal processing error.", "face_detected": False}

# ---------------------------------------------------------
# 6. START SERVER
# ---------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting Uvicorn server on PORT 8000...")
    # Bound to Port 8000
    uvicorn.run("main_audio:app", host="0.0.0.0", port=8000, reload=False)