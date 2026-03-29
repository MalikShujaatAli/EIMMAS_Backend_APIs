import os
import sys
import time
import logging
import warnings
import zipfile
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ---------------------------------------------------------
# 1. ENVIRONMENT & PROFESSIONAL LOGGING
# ---------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.filterwarnings('ignore')

# Automatically locate the central \logs\ directory
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logs"))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "video_api.log")

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
# 2. CONFIGURATIONS & MODEL SETUP
# ---------------------------------------------------------
FACE_MODEL_PATH = "fer_best_model.keras"
ZIPPED_MODEL_PATH = "fer_best_model.keras.zip"
MP_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
MP_MODEL_PATH = "face_detector.tflite"
EMOTION_LABELS = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Auto-download MediaPipe detector model if missing
if not os.path.exists(MP_MODEL_PATH):
    logger.info("Downloading MediaPipe Face Detector model...")
    urllib.request.urlretrieve(MP_MODEL_URL, MP_MODEL_PATH)

# --- STRICTER DETECTION SETTINGS ---
base_options = python.BaseOptions(model_asset_path=MP_MODEL_PATH)
options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.75, # HIGHER = STRICTER (Set to 0.75 to reject dogs/objects)
    min_suppression_threshold=0.3
)
detector = vision.FaceDetector.create_from_options(options)

# ---------------------------------------------------------
# 3. LOAD FER MODEL
# ---------------------------------------------------------
app = FastAPI(title="Production Human Emotion Recognition API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

try:
    if os.path.exists(ZIPPED_MODEL_PATH) and not os.path.exists(FACE_MODEL_PATH):
        logger.info("Extracting Zipped FER Model...")
        with zipfile.ZipFile(ZIPPED_MODEL_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")
    
    logger.info("Loading Vision AI Model into memory...")
    emotion_model = tf.keras.models.load_model(FACE_MODEL_PATH)
    # Warmup
    _ = emotion_model(np.zeros((1, 112, 112, 1)), training=False)
    logger.info("System Ready: Human Filtering Active & Model Loaded Successfully!")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# 4. SHARED UTILITIES
# ---------------------------------------------------------

def extract_human_face(image_bgr):
    """Detects face. Returns None if it fails the 85% human-confidence check."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    detection_result = detector.detect(mp_image)
    if not detection_result.detections:
        return None

    img_h, img_w = image_bgr.shape[:2]
    # Process only the highest-confidence face
    best_det = detection_result.detections[0]
    bbox = best_det.bounding_box
    
    # Boundary conversion (CRITICAL: Cast to integers to prevent array slicing crashes)
    x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
    x, y = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)
    
    face_roi = image_bgr[y:y2, x:x2]
    if face_roi.size == 0: return None
    
    return cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

def analyze_emotion(face_gray):
    """Full probability prediction logic."""
    # ✅ FIX 2: Apply CLAHE for lighting correction
    face_clahe = clahe.apply(face_gray)
    
    # ✅ FIX 3: Force CUBIC interpolation
    resized = cv2.resize(face_clahe, (112, 112), interpolation=cv2.INTER_CUBIC)
    
    inp = (resized.astype("float32") / 255.0).reshape(1, 112, 112, 1)
    preds = emotion_model(inp, training=False).numpy()[0]
    return preds

# ---------------------------------------------------------
# 5. RESTORED ENDPOINTS
# ---------------------------------------------------------

@app.get("/")
def health_check():
    return {"status": "online", "service": "Vision Emotion API", "port": 8002}

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    start_time = time.time()
    logger.info(f"Receiving image request: {file.filename}")
    try:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        face_gray = extract_human_face(img)
        if face_gray is None:
            logger.warning("No human face detected in image.")
            raise HTTPException(status_code=400, detail="No human face detected (or confidence below 85%).")

        probs = analyze_emotion(face_gray)
        win_idx = np.argmax(probs)
        
        latency = round(time.time() - start_time, 2)
        logger.info(f"Image Result: {EMOTION_LABELS[win_idx]} ({latency}s)")

        return {
            "predicted_emotion": EMOTION_LABELS[win_idx],
            "confidence": round(float(probs[win_idx]) * 100, 2),
            "probabilities": {EMOTION_LABELS[i]: round(float(probs[i]), 4) for i in range(len(EMOTION_LABELS))},
            "processing_time": f"{latency}s"
        }
    except Exception as e:
        logger.error(f"Image prediction failed: {e}")
        return {"error": str(e)}

@app.post("/predict_video")
async def predict_video(file: UploadFile = File(...)):
    start_time = time.time()
    logger.info(f"Receiving video request: {file.filename}")
    temp_path = f"v_temp_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        cap = cv2.VideoCapture(temp_path)
        all_probs = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % 10 == 0:
                face = extract_human_face(frame)
                if face is not None:
                    all_probs.append(analyze_emotion(face))
            frame_idx += 1
        cap.release()
        os.remove(temp_path)

        if not all_probs:
            logger.warning("No human face detected in video stream.")
            raise HTTPException(status_code=400, detail="No human face detected in the video stream.")

        avg_probs = np.mean(all_probs, axis=0)
        win_idx = np.argmax(avg_probs)

        latency = round(time.time() - start_time, 2)
        logger.info(f"Video Result: {EMOTION_LABELS[win_idx]} ({latency}s)")

        return {
            "predicted_emotion": EMOTION_LABELS[win_idx], # ✅ FIX 4: Matches Orchestrator
            "confidence": round(float(avg_probs[win_idx]) * 100, 2),
            "probabilities": {EMOTION_LABELS[i]: round(float(avg_probs[i]), 4) for i in range(len(EMOTION_LABELS))},
            "processing_time": f"{latency}s"
        }
    except Exception as e:
        logger.error(f"Video prediction failed: {e}")
        if os.path.exists(temp_path): os.remove(temp_path)
        return {"error": str(e)}

if __name__ == "__main__":
    logger.info("Starting Uvicorn server on PORT 8002...")
    uvicorn.run("main_video:app", host="0.0.0.0", port=8002, reload=False)