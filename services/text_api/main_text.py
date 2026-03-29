import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

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

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ===================================
# 0) LOGGING SETUP
# ===================================
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

try:
    # ✅ FIX: correct tokenizer
    nltk.download('punkt', quiet=True)
    logger.info("✅ NLTK punkt checked.")
except Exception as e:
    logger.error(f"❌ NLTK Download failed: {e}")

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

logger.info("⏳ Loading Tokenizer...")
try:
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    logger.info("✅ Tokenizer loaded.")
except Exception as e:
    logger.error(f"❌ Tokenizer failed: {e}")

logger.info("⏳ Loading Text Model...")
try:
    custom_objects = {'AttentionLayer': AttentionLayer, 'Custom>AttentionLayer': AttentionLayer}
    model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)

    dummy_text = pad_sequences(tokenizer.texts_to_sequences(["warmup"]), maxlen=MAX_LEN)
    _ = model(dummy_text, training=False)

    logger.info("✅ Model loaded and warmed up!")
except Exception as e:
    logger.error(f"❌ Model failed: {e}")

# ===================================
# 4) ENDPOINTS
# ===================================
@app.get("/")
def health_check():
    return {"status": "online", "service": "Text Emotion API", "port": 8001}

@app.post("/predict_text")
async def predict_text(input_data: TextInput):
    text = input_data.paragraph

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    if len(text.split()) < 3:
        return {
        "sentences": [],
        "final_emotion": "neutral",
        "weighted_probabilities": {}
        }

    try:
        sentences = [clean_text(s) for s in sent_tokenize(text)]
        sequences = tokenizer.texts_to_sequences(sentences)
        padded_seqs = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding='post',
        truncating='post'
        )

        predictions = model(padded_seqs, training=False).numpy()

        # soften predictions
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