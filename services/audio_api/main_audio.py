import os
import io
import sys
import time
import logging
import warnings
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ---------------------------------------------------------
# 1. ENVIRONMENT & PROFESSIONAL LOGGING
# ---------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF C++ warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
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
MAX_FILE_SIZE_MB = 10
MODEL_PATH = "audio_best_model.keras"

INT_TO_EMOTION = {
    0: 'angry', 1: 'disgust', 2: 'fearful', 
    3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprised'
}

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

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found.")
    
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # PERFORMANCE WARMUP: Ensures the first request isn't slow
    dummy_input = np.zeros((1, MAX_PAD_LEN, N_MFCC))
    _ = model(dummy_input, training=False)
    
    logger.info("Audio Model loaded successfully!")
except Exception as e:
    logger.error(f"CRITICAL ERROR: Could not load model. {e}")
    sys.exit(1)

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
            
        # Resample only if needed
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

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
        # Read file into memory buffer
        content = await file.read()
        
        # Security: Size limit
        if len(content) / (1024*1024) > MAX_FILE_SIZE_MB:
            raise HTTPException(status_code=413, detail="File too large.")

        # Preprocess
        tensor = get_features_fast(content)
        if tensor is None:
            raise HTTPException(status_code=400, detail="Audio file is corrupted or silent.")

        # DIRECT INFERENCE: Faster than model.predict() for single requests
        prediction_tensor = model(tensor, training=False)
        prediction = prediction_tensor.numpy()[0]
        
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

    except Exception as e:
        logger.error(f"Endpoint prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# ---------------------------------------------------------
# 6. START SERVER
# ---------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting Uvicorn server on PORT 8000...")
    # Bound to Port 8000
    uvicorn.run("main_audio:app", host="0.0.0.0", port=8000, reload=False)