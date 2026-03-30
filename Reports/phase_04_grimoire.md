# Phase 4 Grimoire: Complete Technical Archaeology

---

## Header 1: Complete File Inventory

| File | Size (bytes) | Role |
|---|---|---|
| `FYP old/phase04_text_bilstm_trainer.py` | 2,869 | First BiLSTM text model trainer |
| `FYP old/phase04_text_cnn_predictor.py` | 968 | CNN-based text emotion predictor |
| `FYP old/phase04_text_attention_tester.py` | 2,971 | AttentionLayer text tester with glob discovery |
| `FYP old/phase04_text_negation_engine.py` | 5,324 | Advanced text predictor with negation engine |
| `FYP old/phase04_text_bilstm_attention_v1.py` | 3,845 | Text predictor with AttentionLayer (K.tanh/K.dot) |
| `FYP old/phase04_text_api_bilstm_keyword.py` | 6,211 | First text emotion FastAPI endpoint |
| `FYP old/phase04_text_negation_engine_copy.py` | 5,324 | Byte-identical copy of `phase04_text_negation_engine.py` |

---

## Header 2: Line-by-Line Logic Migration

### File: `phase04_text_bilstm_trainer.py` (Complete Source)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from sklearn.model_selection import train_test_split
import pickle

# ----- Load and inspect your CSV file -----
df = pd.read_csv("archive_8/archive/combined_emotion.csv")
df.drop_duplicates(inplace=True)

plt.figure(figsize=(10,5))
sns.countplot(x='emotion', data=df, order=df['emotion'].value_counts().index, palette='viridis')
plt.title("Emotion Distribution")
plt.xticks(rotation=45)
plt.show()

# ----- Encode labels -----
label_encoder = LabelEncoder()
df['emotion_encoded'] = label_encoder.fit_transform(df['emotion'])

# ----- Split train/test -----
X_train, X_test, y_train, y_test = train_test_split(
    df['sentence'], df['emotion_encoded'], 
    test_size=0.2, random_state=42, stratify=df['emotion_encoded']
)

max_words = 20000
max_len = 50

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

y_train = np.array(y_train)
y_test = np.array(y_test)
num_classes = len(label_encoder.classes_)

# ----- Build and train the BiLSTM model -----
model_bilstm = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    Bidirectional(LSTM(128, return_sequences=False, dropout=0.3, recurrent_dropout=0.3)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model_bilstm.compile(loss='sparse_categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])

history_bilstm = model_bilstm.fit(
    X_train_pad, y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=256
)

# ----- Evaluate and save artifacts locally -----
loss, accuracy = model_bilstm.evaluate(X_test_pad, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

model_bilstm.save("emotion_bilstm_model.h5")
with open("tokenizer1.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("label_encoder1.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
print("✅ Model, tokenizer, and label encoder saved to current directory.")
```

#### Block 1: Dataset Loading (Lines 14-22)
- **`archive_8/archive/combined_emotion.csv`**: A local CSV with `sentence` and `emotion` columns. The exact dataset composition is not documented, but based on the Phase 9 Kaggle notebook, it is a merged dataset of Kaggle emotion text datasets.
- **`df.drop_duplicates()`**: Removes exact duplicate rows. Does NOT check for near-duplicates or contradictory labels.
- **Visualization**: `sns.countplot` generates a bar chart of emotion distribution. This is a diagnostic step — the output is visual-only, not stored programmatically.
- **Missing**: No handling of the `'suprise'` typo that appears in the training data. The Phase 9 Kaggle notebook adds `df[LABEL_COLUMN].replace('suprise', 'surprise', inplace=True)`.

#### Block 2: Label Encoding (Lines 25-27)
- **`LabelEncoder()`**: Sklearn's `LabelEncoder` maps string labels to integers alphabetically. The exact class-to-integer mapping depends on the unique labels in the dataset.
- **Fragile coupling**: The label encoder is pickled to `label_encoder1.pkl`. Any script loading this file MUST use the same label encoder instance to decode predictions. If the training data changes (different unique labels or different order), the mapping silently breaks.

#### Block 3: Tokenization (Lines 35-45)
- **`max_words = 20000`**: Vocabulary size cap. Words outside the top 20,000 are replaced by the `<OOV>` (Out Of Vocabulary) token. The Phase 9 Kaggle notebook increases this to `MAX_VOCAB_SIZE = 30000`.
- **`max_len = 50`**: Maximum sequence length. Sentences longer than 50 tokens are truncated. The Phase 9 production model uses `MAX_LEN = 100`.
- **`padding='post'`**: Zero-padding is added after the text, not before. This convention is preserved through all phases.

#### Block 4: BiLSTM Architecture (Lines 52-59)
- **No Attention mechanism**: This is a plain `Bidirectional(LSTM(128))` — every LSTM hidden state has equal weight in the final representation. The custom `AttentionLayer` is not introduced until `phase04_text_attention_tester.py`.
- **`recurrent_dropout=0.3`**: Applies dropout within the LSTM recurrence. This was removed in later phases because it prevents CuDNN kernel acceleration on GPU.
- **`Embedding(input_dim=max_words, output_dim=128)`**: Learns 128-dimensional word vectors from scratch. No pretrained embeddings (GloVe, Word2Vec) are used.

#### Block 5: Training (Lines 65-70)
- **`epochs=5`**: Absurdly low. 5 epochs is insufficient for convergence on text classification tasks. The Phase 9 Kaggle notebook trains for up to 15 epochs with early stopping.
- **`batch_size=256`**: Large batch size. Efficient but may harm generalization on small datasets.
- **`validation_split=0.2`**: Uses 20% of training data for validation, but this split is random (not stratified). The `train_test_split` above already stratified the holdout test set, but the validation split within training is not stratified.

#### Block 6: Save Artifacts (Lines 77-82)
- **Three output files**: `emotion_bilstm_model.h5`, `tokenizer1.pkl`, `label_encoder1.pkl`. The `1` suffix suggests this was the first attempt. Later artifacts use timestamped filenames (e.g., `tokenizer_20251203_192244.pkl`).

---

### File: `phase04_text_cnn_predictor.py` (Complete Source)

```python
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load model, tokenizer, and label encoder ---
emotion_model_path = "emotion_TEXT_cnn_model.h5"
tokenizer_path = "tokenizer.pkl"
label_encoder_path = "label_encoder.pkl"

model = load_model(emotion_model_path)
tokenizer = pickle.load(open(tokenizer_path, 'rb'))
le = pickle.load(open(label_encoder_path, 'rb'))

def predict_emotion(text):
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=100)
    pred = model.predict(seq)
    emotion_idx = pred.argmax(axis=1)[0]
    emotion = le.inverse_transform([emotion_idx])[0]
    return emotion

# ----- Input Loop -----
print("Enter text (or type 'exit' to quit):")
while True:
    text = input("Text: ")
    if text.lower() == 'exit':
        break
    emotion = predict_emotion(text)
    print(f"Predicted emotion: {emotion}\n")
```

#### Analysis
- **Different model**: `emotion_TEXT_cnn_model.h5` — a CNN-based text model (NOT the BiLSTM from `phase04_text_bilstm_trainer.py`). This confirms that multiple model architectures were tested in parallel.
- **`maxlen=100`**: Different from `phase04_text_bilstm_trainer.py`'s `max_len=50`. This means the CNN model was trained with a different preprocessing pipeline.
- **`pickle.load(open(..., 'rb'))`**: File handle is never explicitly closed. A minor resource leak pattern.
- **No sentence splitting**: The entire input text is processed as a single sequence. No `sent_tokenize()`. This is the simplest approach but loses per-sentence granularity.
- **No confidence filtering**: The argmax prediction is returned regardless of confidence. No threshold, no "context unclear" fallback.

---

### File: `phase04_text_attention_tester.py` (Complete Source)

```python
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import os
import glob

nltk.download("punkt", quiet=True)

stemmer = PorterStemmer()

# ========================================
# 1. CUSTOM ATTENTION LAYER (TF 2.12 SAFE)
# ========================================
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.W = self.add_weight(name="att_weight", shape=(dim,), initializer="glorot_uniform")
        self.b = self.add_weight(name="att_bias", shape=(1,), initializer="zeros")
        super().build(input_shape)

    def call(self, inputs, mask=None):
        score = tf.tensordot(inputs, self.W, axes=[[2], [0]]) + self.b
        weights = tf.nn.softmax(score, axis=1)
        weights = tf.expand_dims(weights, -1)
        return tf.reduce_sum(inputs * weights, axis=1)

# ========================================
# 2. FIND MODEL FILES
# ========================================
print("Searching text model...")

MODEL_PATH = TOKENIZER_PATH = LABEL_PATH = None

if os.path.isdir("trained_models/aa"):
    h5 = sorted(glob.glob("trained_models/aa/emotion_model_*.h5"))
    tok = sorted(glob.glob("trained_models/aa/tokenizer_*.pkl"))
    enc = sorted(glob.glob("trained_models/aa/label_encoder_*.pkl"))

    if h5 and tok and enc:
        MODEL_PATH = h5[-1]
        TOKENIZER_PATH = tok[-1]
        LABEL_PATH = enc[-1]

print("MODEL:", MODEL_PATH)
print("TOKENIZER:", TOKENIZER_PATH)
print("ENCODER:", LABEL_PATH)

# ========================================
# 3. LOAD MODEL SAFELY
# ========================================
custom_objects = {
    "AttentionLayer": AttentionLayer
}

try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects=custom_objects,
        compile=False
    )
    tokenizer = pickle.load(open(TOKENIZER_PATH, "rb"))
    label_encoder = pickle.load(open(LABEL_PATH, "rb"))

    print("✅ TEXT MODEL LOADED SUCCESSFULLY!")

except Exception as e:
    print("❌ ERROR LOADING TEXT MODEL:", str(e))
    exit()

# ========================================
# 4. SIMPLE TEST
# ========================================
def predict(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=60, padding="post")
    pred = model.predict(pad)[0]
    label = label_encoder.classes_[np.argmax(pred)]
    return label, pred

while True:
    txt = input("\nEnter text: ")
    if txt.lower() == "exit":
        break
    emotion, probs = predict(txt)
    print("Emotion:", emotion)
    print("Raw probs:", probs)
```

#### Critical Artifacts

- **First `AttentionLayer` appearance**: This is where the custom attention mechanism was born. The implementation uses `tf.tensordot` to compute attention scores, `tf.nn.softmax` for normalization, and `tf.reduce_sum` for weighted pooling. The weight shape is `(dim,)` — a 1D vector, making this a simplified "additive attention" variant.
- **`AttentionLayer` evolution**: In `phase04_text_attention_tester.py`, the weight shape is `(dim,)` with `glorot_uniform` initializer. In `phase07_text_api_standalone.txt`, it becomes `(input_shape[-1], 1)` with `normal` initializer, and the computation changes to `K.tanh(K.dot(x, self.W) + self.b)` — a more standard Bahdanau-style attention. In Phase 9's `main_text.py`, the ops change to `keras.ops.tanh`/`ops.matmul`/`ops.softmax`/`ops.sum` for Keras 3 compatibility.
- **`glob.glob("trained_models/aa/...")`**: Dynamic model file discovery by sorting glob matches and taking the last (most recent) file. This was replaced by explicit path constants in all later phases.
- **`PorterStemmer` — imported, never used**: `stemmer = PorterStemmer()` is defined on line 15 but `stemmer` is never called anywhere in the script. This is evidence of an abandoned NLP preprocessing step — the developer considered stemming words before tokenization but decided against it (likely because the tokenizer was trained on unstemmed text, so stemming at inference time would cause vocabulary mismatches).
- **`maxlen=60`**: Different from `phase04_text_bilstm_trainer.py`'s 50 and `phase04_text_cnn_predictor.py`'s 100. This is the third distinct `maxlen` value, indicating that model retraining happened multiple times with different configurations.
- **`compile=False`**: Model is loaded without recompiling the optimizer. This is correct for inference-only usage and avoids warnings about missing optimizer state.

---

### File: `phase04_text_negation_engine.py` (Complete Source — 183 lines)

```python
# ===================================
# 1) IMPORTS
# ===================================
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import re
from nltk.tokenize import sent_tokenize

# ===================================
# 2) MODEL PATHS (TF 2.12 FIXED)
# ===================================
MODEL_PATH = r"trained_models/emotion_model_tf212_fixed.h5"
TOKENIZER_PATH = r"trained_models/tokenizer_20251203_192244.pkl"
LABEL_ENCODER_PATH = r"trained_models/label_encoder_20251203_192244.pkl"

MAX_LEN = 50

# ===================================
# 3) LOAD MODEL
# ===================================
print("Loading TF 2.12 compatible model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✔ Model loaded successfully!\n")

# ===================================
# 4) LOAD TOKENIZER & LABEL ENCODER
# ===================================
print("Loading tokenizer & label encoder...")
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

print("✔ Tokenizer & label encoder loaded.")
print("Classes:", list(label_encoder.classes_))
print("\nReady. Type 'exit' to quit.\n")

# ===================================
# 5) HELPER: IS CONTEXT CLEAR?
# ===================================
def is_context_clear(sentence):
    s = sentence.strip()

    if len(s.split()) <= 2:
        return False

    if len(re.sub(r"[a-zA-Z]", "", s)) > len(s) / 2:
        return False

    emotion_words = [
        "happy","sad","angry","upset","frustrated",
        "fear","scared","worried","terrified",
        "love","hate","calm","excited","hurt","broken","feeling"
    ]

    negators = ["not", "never", "no", "isn't", "aren't", "won't", "didn't"]

    if not any(w in s.lower() for w in emotion_words) and \
       not any(n in s.lower() for n in negators):
        return False

    return True

# ===================================
# 6) NEGATION REWRITE ENGINE
# ===================================
NEGATION_MAP = {
    "happy": "sad",
    "sad": "happy",
    "angry": "calm",
    "upset": "calm",
    "excited": "disappointed",
    "calm": "anxious",
    "love": "dislike",
    "hate": "like",
    "proud": "ashamed",
    "confident": "insecure",
    "brave": "afraid"
}

def rewrite_sentence(sentence):
    s = sentence.lower()

    for word in NEGATION_MAP:
        pattern = r"not\s+" + word
        if re.search(pattern, s):
            return re.sub(pattern, NEGATION_MAP[word], s)

    patterns = [
        r"not feeling\s+(\w+)",
        r"not really\s+(\w+)",
        r"not exactly\s+(\w+)"
    ]

    for p in patterns:
        m = re.search(p, s)
        if m:
            original = m.group(1)
            if original in NEGATION_MAP:
                return re.sub(original, NEGATION_MAP[original], s)

    m2 = re.search(r"i(\'m| am) not (\w+)", s)
    if m2:
        target = m2.group(2)
        if target in NEGATION_MAP:
            return re.sub(r"not " + target, NEGATION_MAP[target], s)

    return sentence

# ===================================
# 7) PREDICT FUNCTION
# ===================================
def predict_emotion(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    pred = model.predict(padded, verbose=0)[0]

    emotion = label_encoder.inverse_transform([np.argmax(pred)])[0]

    probs = {
        label_encoder.inverse_transform([i])[0]: float(pred[i])
        for i in range(len(pred))
    }
    return emotion, probs

# ===================================
# 8) INTERACTIVE PREDICT LOOP
# ===================================
while True:
    paragraph = input("\nEnter text (or 'exit'): ")

    if paragraph.lower() == "exit":
        break

    sentences = sent_tokenize(paragraph)

    vote_counter = {}
    prob_accumulator = {cls: 0 for cls in label_encoder.classes_}

    print("\n--- Sentence-level Analysis ---\n")

    for s in sentences:

        if not is_context_clear(s):
            print(f"Sentence : {s}")
            print("⚠ Context unclear — please be more specific.\n")
            continue

        rewritten = rewrite_sentence(s)
        emotion, probs = predict_emotion(rewritten)

        print(f"Original : {s}")
        print(f"Rewritten: {rewritten}")
        print(f"Predicted: {emotion}")
        print("Probs    :", ", ".join([f"{k}:{v*100:.1f}%" for k, v in probs.items()]))
        print("")

        vote_counter[emotion] = vote_counter.get(emotion, 0) + 1
        for k in prob_accumulator:
            prob_accumulator[k] += probs[k]

    if vote_counter:
        print("--- Paragraph Summary ---")
        print("Votes:", vote_counter)

        total = sum(vote_counter.values())
        avg_probs = {k: f"{(prob_accumulator[k]/total)*100:.2f}%" for k in prob_accumulator}

        print("Weighted probabilities:", avg_probs)

        final_emotion = max(vote_counter, key=vote_counter.get)
        print("Final Emotion:", final_emotion)

    print("\n==============================================\n")
```

#### Block-by-Block Annotation

**`is_context_clear()` (Lines 46-67)**
- Rejects sentences ≤ 2 words: "I'm sad" (2 words) → **rejected**. "Help me" (2 words) → **rejected**.
- Rejects sentences with >50% non-alphabetic characters: "I'm so angry!!!" → 4 exclamations + apostrophe = 5 non-alpha out of 16 chars → passes. But "😭😭😭" → 100% non-alpha → rejected (reasonable).
- Requires presence of hardcoded emotion keywords OR negators: "The project deadline is tomorrow and I can barely breathe" → no keyword match → **rejected**, even though it clearly expresses anxiety.
- **Phase 9 resolution**: `is_context_clear()` is completely removed. The retrained BiLSTM+Attention model handles ambiguous sentences by predicting with low confidence, which is caught by the dual-threshold filter (`confidence < 0.50` OR `gap < 0.15`).

**`NEGATION_MAP` and `rewrite_sentence()` (Lines 73-114)**
- 11 word-pair mappings. Total English negation patterns: infinite.
- `"not happy"` → regex `r"not\s+happy"` → replaced with `"sad"`. This works for the exact phrase "not happy" but fails for: "not particularly happy", "not at all happy", "not feeling happy about this", "unhappy", "far from happy".
- Adjacent patterns attempt to catch "not feeling X", "not really X", "not exactly X", and "I'm not X" / "I am not X". Each requires the target word to exist in `NEGATION_MAP`.
- **Logical flaw**: If a sentence contains a negation of a word NOT in the map (e.g., "not peaceful"), the function returns the original sentence unchanged, sending "not peaceful" to the model — which may predict "joy" because it sees "peaceful".
- **Phase 9 resolution**: `rewrite_sentence()` is completely removed. The BiLSTM+Attention model was retrained with a larger dataset that includes negated sentences, so it learns negation patterns implicitly.

**`predict_emotion()` (Lines 120-131)**
- Single sentence → single `model.predict()` call. When called inside the `for s in sentences` loop, this means N sentences = N `model.predict()` calls.
- **`label_encoder.inverse_transform([i])[0]`**: Called once per class per prediction to build the probabilities dictionary. For 6 classes and 5 sentences, this is 30 `inverse_transform` calls — wasteful compared to a pre-built mapping dict.

**Paragraph Voting System (Lines 145-180)**
- `vote_counter`: Counts how many sentences predict each emotion. The paragraph's final emotion is the mode (most frequent vote).
- `prob_accumulator`: Sums raw probabilities across sentences, then averages them.
- This dual voting+averaging approach survives conceptually into Phase 9's `main_text.py` (probability accumulation + final emotion selection).

---

### File: `phase04_text_bilstm_attention_v1.py` & `phase04_text_api_bilstm_keyword.py` (The API Bridge)

These two scripts form the final evolution of Phase 4 before the unified APIs.

**`phase04_text_bilstm_attention_v1.py`**:
- **Role**: Desktop text predictor with the mature `AttentionLayer` (`K.tanh`/`K.dot` ops).
- **Evolution**: Shows `AttentionLayer` migrating from `tf.tensordot` (in `phase04_text_attention_tester.py`) to the Bahdanau-style `K.tanh(K.dot(x, W) + b)` used in Phase 7.
- **Voting**: Uses Python's `collections.Counter` for majority voting, simpler than the manual dictionary in `phase04_text_negation_engine.py`.

**`phase04_text_api_bilstm_keyword.py`**:
- **Role**: The **first text emotion FastAPI endpoint** (Port 8001). The missing link between Phase 4 desktop scripts and Phase 7 distributed APIs.
- **Key Innovation**: Replaces `is_context_clear()` with a massive `rawemotionwords` list (150+ keywords). The `sentence_has_emotion()` function requires at least one keyword match before running the model.
- **Why abandoned**: Phase 9's dual-threshold confidence filter proved far superior to relying on hardcoded vocabulary arrays.

---

## Header 3: Micro-Decision Log

| Decision | `phase04_text_bilstm_trainer.py` | `phase04_text_cnn_predictor.py` | `phase04_text_attention_tester.py` | `phase04_text_negation_engine.py` | Phase 9 (`main_text.py`) |
|---|---|---|---|---|---|
| Architecture | BiLSTM | CNN | BiLSTM+Attention | BiLSTM (model loaded) | BiLSTM+Attention |
| `max_len` | 50 | 100 | 60 | 50 | 100 |
| `max_words` | 20,000 | Unknown | Unknown | Unknown | 30,000 |
| Attention | None | None | `tf.tensordot` | None (model may have it) | `ops.tanh`+`ops.matmul` |
| Negation handling | None | None | None | `rewrite_sentence()` | Removed (model handles) |
| Context filter | None | None | None | `is_context_clear()` | Removed |
| Sentence splitting | None | None | None | `sent_tokenize` | `sent_tokenize` |
| Batch inference | No | No | No | No | Yes (`pad_sequences` on all) |
| `@tf.function` | No | No | No | No | Yes |
| `asyncio.to_thread` | No | No | No | No | Yes |
| Confidence sharpening | None | None | None | None | `predictions ** 1.5` |

---

## Header 4: Inter-Phase Diff Analysis

| Phase 4 Artifact | Where it migrated | What replaced it |
|---|---|---|
| `phase04_text_bilstm_trainer.py` (trainer) | → Kaggle notebook (`training_models_notebook_dump.txt` lines 1-800) | Retrained with Attention, class weights, `max_len=100` |
| `phase04_text_cnn_predictor.py` (CNN predictor) | → Abandoned | CNN architecture was not pursued; BiLSTM+Attention won |
| `phase04_text_attention_tester.py` `AttentionLayer` | → `phase07_text_api_standalone.txt` → `main_text.py` | Evolved from `tf.tensordot` → `K.tanh/K.dot` → `ops.tanh/ops.matmul` |
| `phase04_text_attention_tester.py` `PorterStemmer` | → Abandoned | Never used; removed silently |
| `phase04_text_attention_tester.py` `glob.glob()` discovery | → Replaced by explicit paths | All Phase 7+ scripts use hardcoded model paths |
| `phase04_text_negation_engine.py` `is_context_clear()` | → Removed entirely in Phase 9 | Replaced by dual-threshold confidence filter |
| `phase04_text_negation_engine.py` `NEGATION_MAP` | → Removed entirely in Phase 9 | Model handles negation natively |
| `phase04_text_negation_engine.py` `rewrite_sentence()` | → Removed entirely in Phase 9 | Model handles negation natively |
| `phase04_text_negation_engine.py` vote counter | → `main_text.py` accumulator logic | Preserved conceptually |
| `phase04_text_negation_engine_copy.py` | → Byte-identical copy; co-located for Phase 6 monolith | Replaced by `main_text.py` endpoint |
