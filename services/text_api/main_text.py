import os
import warnings

# ===================================
# 0) ENVIRONMENT & WARNING CONTROLS
# ===================================
# Must be set BEFORE importing tensorflow or keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

warnings.filterwarnings("ignore")
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import pickle
import logging
import numpy as np
import tensorflow as tf
import keras  
from keras.layers import Layer
from keras import ops  
from keras.utils import pad_sequences
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
from nltk.tokenize import sent_tokenize

import re

# PRE-COMPILE REGEX FOR HEAVY WORKLOADS
URL_REGEX = re.compile(r'http\S+|www\S+|https\S+')
HTML_REGEX = re.compile(r'<.*?>')
TAG_REGEX = re.compile(r'\@w+|\#')
CHAR_REGEX = re.compile(r'[^a-zA-Z\s]')
SPACE_REGEX = re.compile(r'\s+')

def clean_text(text):
    text = str(text).lower()
    text = URL_REGEX.sub('', text)
    text = HTML_REGEX.sub('', text)
    text = TAG_REGEX.sub('', text)
    text = CHAR_REGEX.sub('', text)
    text = SPACE_REGEX.sub(' ', text).strip()
    return text

import sys
import os

# ===================================
# 0) LOGGING SETUP
# ===================================
# Automatically locate the central \logs\ directory
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logs"))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "text_api.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# NOTE: nltk.download('punkt') was removed from the global scope.
# It MUST be run statically during build/deployment via setup_nltk.py
# This prevents the container from crashing or hanging if NLTK servers are offline.
try:
    # Check if we have the package before booting
    nltk.data.find('tokenizers/punkt')
    logger.info("NLTK punkt data verified locally.")
except LookupError:
    logger.error("CRITICAL: NLTK punkt data missing. Please run setup_nltk.py on the server first.")

# ===================================
# 1) CONFIGURATION & LABELS
# ===================================
INT_TO_EMOTION = {0: 'joy', 1: 'sad', 2: 'anger', 3: 'fear', 4: 'love', 5: 'surprise'}
MODEL_PATH = "text_best_model.keras"
TOKENIZER_PATH = "text_tokenizer.pkl"
MAX_LEN = 100 
CONFIDENCE_THRESHOLD = 0.40

# ===================================
# 2) ATTENTION LAYER
# ===================================
@keras.saving.register_keras_serializable(package="Custom", name="AttentionLayer")
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="glorot_uniform")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = ops.tanh(ops.matmul(x, self.W) + self.b)
        a = ops.softmax(e, axis=1)
        output = x * a
        return ops.sum(output, axis=1)

    def get_config(self):
        return super(AttentionLayer, self).get_config()

# ===================================
# 3) APP INITIALIZATION
# ===================================
class TextInput(BaseModel):
    paragraph: str

app = FastAPI(title="Production Text Emotion API")

logger.info("Loading Tokenizer...")
try:
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    logger.info("Tokenizer loaded.")
except Exception as e:
    logger.error(f"CRITICAL ERROR: Tokenizer config failed: {e}")
    sys.exit(1)

logger.info("Loading Text Model...")
try:
    custom_objects = {'AttentionLayer': AttentionLayer, 'Custom>AttentionLayer': AttentionLayer}
    model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)

    # LIGHTNING SPEED FIX: Compile the Python model into a static C++ Graph!
    @tf.function(reduce_retracing=True)
    def compute_inference(tensor_input):
        return model(tensor_input, training=False)

    dummy_text = pad_sequences(tokenizer.texts_to_sequences(["warmup"]), maxlen=MAX_LEN)
    _ = compute_inference(dummy_text)

    logger.info("Model loaded and warmed up!")
except Exception as e:
    logger.error(f"CRITICAL ERROR: AI Neural Model failed to load: {e}")
    sys.exit(1)

# ===================================
# 4) ENDPOINTS
# ===================================
@app.get("/")
def health_check():
    return {"status": "online", "service": "Text Emotion API", "port": 8001}

import asyncio

@app.post("/predict_text")
async def predict_text(input_data: TextInput):
    text = input_data.paragraph

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    # Removed destructive length filter: "I'm sad" (2 words) should be analyzed!
    try:
        sentences = [clean_text(s) for s in sent_tokenize(text)]
        
        # SAFETY SHIELD: Filter out any sentences that were reduced to blank spaces (e.g., if user sent pure emojis)
        sentences = [s for s in sentences if len(s.strip()) > 1]
        if not sentences:
            logger.warning("All input text was stripped (mostly emojis/symbols). Bypassing ML.")
            return {"sentences": [], "final_emotion": "context unclear", "weighted_probabilities": {}}
            
        sequences = tokenizer.texts_to_sequences(sentences)
        padded_seqs = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding='post',
        truncating='post'
        )
 
        # LIGHTNING SPEED FIX: Offload computation to background thread to free up FastAPI!
        def _run_model():
            return compute_inference(padded_seqs).numpy()
            
        predictions = await asyncio.to_thread(_run_model)

        # Sharpen predictions (amplify confidence gaps between top and lower classes)
        predictions = predictions ** 1.5
        predictions = predictions / predictions.sum(axis=1, keepdims=True)
    
        results = []
        vote_counter = {emo: 0 for emo in INT_TO_EMOTION.values()}
        prob_accumulator = {emo: 0.0 for emo in INT_TO_EMOTION.values()}
        valid_sentence_count = 0

        for i, pred_array in enumerate(predictions):
            predicted_idx = int(np.argmax(pred_array))
            max_prob = float(pred_array[predicted_idx])

            predicted_emotion = INT_TO_EMOTION[predicted_idx]
            probs_dict = {INT_TO_EMOTION[idx]: float(prob) for idx, prob in enumerate(pred_array)}

            sorted_probs = np.sort(pred_array)
            gap = sorted_probs[-1] - sorted_probs[-2]

            if max_prob < 0.50 or gap < 0.15:
                emotion_label = "context unclear"
            else:
                emotion_label = predicted_emotion

            results.append({
                "sentence": sentences[i],
                "emotion": emotion_label,
                "probabilities": probs_dict
            })

            if emotion_label != "context unclear":
                vote_counter[predicted_emotion] += 1
                for emotion, prob in probs_dict.items():
                    prob_accumulator[emotion] += prob
                valid_sentence_count += 1

        final_emotion = max(vote_counter, key=vote_counter.get) if valid_sentence_count > 0 else "context unclear"

        avg_probs = {
            emo: round(prob / valid_sentence_count, 4)
            for emo, prob in prob_accumulator.items()
        } if valid_sentence_count > 0 else {}

        logger.info(f"✅ Prediction Complete: {final_emotion}")

        return {
            "sentences": results,
            "final_emotion": final_emotion,
            "weighted_probabilities": avg_probs
        }

    except Exception as e:
        logger.error(f"🔥 ERROR DURING PREDICTION: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_text:app", host="0.0.0.0", port=8001, reload=False)