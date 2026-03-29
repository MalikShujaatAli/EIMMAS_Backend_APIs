import os
import sys
import time
import logging
import asyncio
import httpx
import uvicorn
import tempfile
import uuid
import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete
from groq import AsyncGroq 
from cerebras.cloud.sdk import AsyncCerebras
from contextlib import asynccontextmanager
import re
from dotenv import load_dotenv

# IMPORT OUR DATABASE SETUP
from database import init_db, get_db, ChatSession, ChatMessage

# ---------------------------------------------------------
# 1. PROFESSIONAL ENTERPRISE LOGGING
# ---------------------------------------------------------
# Automatically locate the \logs\ directory in our new folder structure
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logs"))
os.makedirs(LOG_DIR, exist_ok=True) 
LOG_FILE = os.path.join(LOG_DIR, "fusion_api.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),       
        logging.FileHandler(LOG_FILE, encoding="utf-8") # ✅ FIX 1: Added UTF-8 encoding so emojis/languages don't crash Windows            
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 2. ORCHESTRATOR CONFIGURATIONS (SERVER IPs & PORTS)
# ---------------------------------------------------------
API_URLS = {
    "audio": "http://182.180.159.89:8000/predict_audio",
    "text": "http://182.180.159.89:8001/predict_text",
    "image": "http://182.180.159.89:8002/predict_image",
    "video": "http://182.180.159.89:8002/predict_video"
}

WEIGHTS = {
    "visual": 0.35,  
    "audio": 0.15,   
    "text": 0.50     
}



# This physically opens the .env file and loads your keys into memory
load_dotenv() 

# 🚀 ENVIRONMENT TOGGLE 🚀
# Set to True for your Final FYP Presentation (Uses Groq 70B)
# Set to False for daily testing and QA (Uses Cerebras 8B)
USE_PRODUCTION_MODEL = False

# Groq Setup (Production)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = AsyncGroq(api_key=GROQ_API_KEY)

# Cerebras Setup (Development/Testing)
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
cerebras_client = AsyncCerebras(api_key=CEREBRAS_API_KEY)

# ---------------------------------------------------------
# 3. APP INITIALIZATION & LIFECYCLE
# ---------------------------------------------------------
# Connection Pool for backend Microservices
http_client: httpx.AsyncClient = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=100, max_connections=200))
    logger.info("Starting Fusion API... Initializing SQLite Database and Connection Pool.")
    await init_db()
    yield
    await http_client.aclose()
    logger.info("Shutting down Fusion API gracefully...")

app = FastAPI(title="Master Extraction & Fusion Orchestrator API - Enterprise Edition", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
# ---------------------------------------------------------
# SAFETY & BLOCKADE CONFIGURATIONS (REGEX 2.0)
# ---------------------------------------------------------
import re

# These patterns use \b (word boundaries) and .*? (wildcards) to catch 
# typos, bad grammar, and separated phrases (e.g., "I want to completely end everything")
CRISIS_PATTERNS = [
    r"\b(kill|end|destroy|hurt|harm)\b.*?\b(myself|me|my life|it all|everything)\b",
    r"\b(want|wanna|wish|going to|gonna)\b.*?\b(die|suicide|disappear|sleep forever)\b",
    r"\b(no point|no reason|tired|can't do this)\b.*?\b(living|life|trying|anymore)\b",
    r"\b(give up|giving up)\b.*?\b(on life|everything)\b",
    r"\b(don't|do not|never)\b.*?\b(wake up|waking up)\b",  # Caught: "don't want to wake up tomorrow"
    r"\b(don't|do not)\b.*?\b(alive|living)\b.*?\b(myself|me)\b" # Caught: "see him alive or myself"
]

ABUSIVE_PATTERNS = [
    r"\b(bsdk|loru|bitch|mc|bc|chutiya|fuck|fck|f\*\*k|asshole)\b",
    r"\b(shut up|fuck off|die bot)\b"
]
# ---------------------------------------------------------
# 4. AUTHENTICATION (ENTERPRISE-GRADE JWT VERIFICATION)
# ---------------------------------------------------------
SECRET_KEY = "YourSuperSecretKeyHereAtLeast32CharactersLong!" 
ALGORITHM = "HS256"
VALID_ISSUER = "EimmAiSystem"
VALID_AUDIENCE = "EimmAiSystemUsers"

security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token, 
            SECRET_KEY, 
            algorithms=[ALGORITHM],
            issuer=VALID_ISSUER,
            audience=VALID_AUDIENCE
        )
        
        # Comprehensive claim check as per your original file
        email = (
            payload.get("email") or 
            payload.get("unique_name") or 
            payload.get("http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress") or
            payload.get("http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name")
        )
        
        if not email:
            logger.warning(f"Security Alert: Token is valid but missing email payload.")
            raise HTTPException(status_code=401, detail="Unauthorized: Token payload invalid.")
        return email 
        
    except ExpiredSignatureError:
        logger.warning("Security Alert: Blocked request with expired token.")
        raise HTTPException(status_code=401, detail="Unauthorized: Token has expired.")
    except InvalidTokenError as e:
        logger.warning(f"Security Alert: Blocked request with forged/invalid token. Reason: {str(e)}")
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid token.")

# ---------------------------------------------------------
import shutil

# 5. ASYNC MEDIA CONVERSION & MICROSERVICES
# ---------------------------------------------------------
async def process_and_clean_audio(file_bytes: bytes, is_video: bool = False) -> bytes:
    """
    CONVERSION ENGINE: Forced conversion to 16kHz, Mono, PCM 16-bit .WAV
    """
    logger.info(f"Initiating FFmpeg conversion... (Source is Video: {is_video})")
    
    if not shutil.which("ffmpeg"):
        logger.error("CRITICAL ERROR: FFmpeg is not installed on this server or not in the PATH! Cannot extract audio from video.")
        return None if is_video else file_bytes

    suffix = ".mp4" if is_video else ".tmp"
    in_path, out_path = None, None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_in:
            temp_in.write(file_bytes)
            in_path = temp_in.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_out:
            out_path = temp_out.name

        # ✅ FIX: Resolved ffmpeg executable absolute path for Windows compatibility
        ffmpeg_exe = shutil.which("ffmpeg")
        process = await asyncio.create_subprocess_exec(
            ffmpeg_exe, "-y", "-i", in_path, "-vn", 
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", out_path,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        await process.communicate() 
        
        if process.returncode != 0: 
            logger.error("FFmpeg Conversion Failed!")
            return None if is_video else file_bytes
            
        logger.info("Successfully converted media to standard 16kHz WAV.")
        with open(out_path, "rb") as f: return f.read()
        
    except Exception as e: 
        logger.error(f"Error during media conversion: {repr(e)}")
        return None if is_video else file_bytes 
    finally:
        if in_path and os.path.exists(in_path): os.remove(in_path)
        if out_path and os.path.exists(out_path): os.remove(out_path)

async def transcribe_audio_to_text(audio_bytes: bytes) -> str:
    try:
        logger.info("Transcribing converted audio via Groq Whisper API...")
        transcription = await groq_client.audio.transcriptions.create(
            file=("clean_audio.wav", audio_bytes), model="whisper-large-v3-turbo",
        )
        text_result = transcription.text.strip()
        # ✅ FIX 2: Removed microphone emoji to stop Windows CP1252 crash
        logger.info(f"[WHISPER HEARD]: '{text_result}'")
        return text_result
    except Exception as e: 
        logger.error(f"Transcription failed: {str(e)}")
        return None

async def fetch_text_api(client: httpx.AsyncClient, text: str):
    try:
        if not text or not isinstance(text, str):
            logger.warning("fetch_text_api: invalid text")
            return {"source": "text", "data": None}

        logger.info(f"Sending Text to ML API: {API_URLS['text']}")

        response = await client.post(
            API_URLS["text"],
            json={"paragraph": str(text)},   
            timeout=5.0
        )

        if response.status_code == 200:
            return {"source": "text", "data": response.json()}
        else:
            logger.error(f"Text API Error {response.status_code}: {response.text}")

    except Exception as e:
        logger.warning(f"Text API failed: {str(e)}")

    return {"source": "text", "data": None}

async def fetch_file_api(client: httpx.AsyncClient, file_bytes: bytes, filename: str, content_type: str, modality: str, endpoint: str):
    try:
        logger.info(f"Sending {modality.upper()} to ML API: {endpoint}")
        response = await client.post(endpoint, files={'file': (filename, file_bytes, content_type)}, timeout=7)
        if response.status_code == 200:
            data = response.json()
            return {"source": modality, "data": None, "error": data['error']} if "error" in data else {"source": modality, "data": data}
    except Exception as e: 
        logger.warning(f"{modality.upper()} API unreachable/failed: {str(e)}")
    return {"source": modality, "data": None}

def fuse_emotions(results):
    emotion_scores = {}

    emotion_map = {
        "joy": "joy", "happy": "joy",
        "fearful": "fear", "fear": "fear",
        "angry": "anger", "anger": "anger",
        "sadness": "sad", "sad": "sad",
        "surprised": "surprise", "surprise": "surprise",
        "love": "love"
    }

    for result in results:
        data = result["data"]
        if not data or not isinstance(data, dict):
            continue

        source = result["source"]
        weight = WEIGHTS["visual"] if source in ["image", "video"] else WEIGHTS[source]

        if source == "text":
            raw_emotion = data.get("final_emotion", "neutral").lower()

            probs = data.get("weighted_probabilities", {})
            confidence = probs.get(raw_emotion, 0.0)

        else:
            raw_emotion = data.get("predicted_emotion", "neutral").lower()
            confidence = data.get("confidence", 0.0) / 100.0

        # ✅ LOOPHOLE 2 FIX: Skip broken English / unclear text 
        # so it doesn't dilute the Audio/Video scores!
        if raw_emotion in ["context unclear", "neutral"]:
            continue

        normalized = emotion_map.get(raw_emotion, raw_emotion)

        if normalized not in emotion_scores:
            emotion_scores[normalized] = 0.0

        emotion_scores[normalized] += weight * confidence

    if not emotion_scores:
        return "neutral"

    final = max(emotion_scores, key=emotion_scores.get)
    
    # 🚨 LOOPHOLE 3 FIX: CONTRADICTION ENGINE (AFFECTIVE MASKING DETECTOR)
    # If the text says negative emotions, but visual/audio says Joy, alert the LLM!
    has_negative_text = any(res["data"].get("final_emotion", "") in ["sad", "fear", "anger"] for res in results if res["source"] == "text")
    has_positive_media = any(res["data"].get("predicted_emotion", "") == "joy" for res in results if res["source"] in ["audio", "video"])
    
    if has_negative_text and has_positive_media:
        logger.warning("🚨 CONTRADICTION DETECTED: Masked Emotion (Smiling while sad/angry).")
        return "masked_distress" # Send this special flag to the LLM

    logger.info(f"FUSION: {emotion_scores} -> {final}")
    return final



# ---------------------------------------------------------
# 6. LLM GENERATION WITH MEMORY (FINAL VERSION)
# ---------------------------------------------------------
async def generate_psychologist_response(fused_emotion: str, user_text: str, chat_history: list):
    if not user_text:
        user_text = "[User communicated via non-verbal cues]"

    system_prompt = """
    You are a licensed clinical psychologist responding to a client.
    Your therapeutic approach is modeled after the 'Counsel Chat' dataset of professional therapist responses.

    --- STRICT LANGUAGE ENFORCEMENT (CRITICAL) ---
    You MUST ONLY read and output English. NEVER speak, reply, or translate into Urdu, Roman Urdu, Hindi, or any other language, even if the user explicitly demands it (e.g., "urdu likh bhai", "talk to me in urdu", "not roman").

    --- SOURCE OF TRUTH PRIORITY ---
    1. User’s explicit words (primary ground truth)
    2. Detected core emotion (supporting signal)
    3. Chat history (contextual continuity)

    --- CORE BEHAVIOR (ORGANIC FLOW) ---
    - Respond like a real human in a natural conversation, not a scripted therapist.
    - Do NOT follow rigid templates (e.g., you do not need to validate, reflect, and ask a question in every single message).
    - Blend understanding and response organically.
    - Only ask a question if it genuinely adds value. It is completely acceptable to respond with just a supportive statement.
    
    --- PREMIUM ANTI-ROBOTIC & VARIATION RULES ---
    - ACTIVE CONTEXTUAL RECALL (CRITICAL): Do not treat the user's latest message in isolation. If the user mentions a sudden mood shift (e.g., "My friend arrived" followed by "I want to hurt myself"), you MUST explicitly connect the dots. Acknowledge the shift and gently ask how the previous event triggered the current feeling.
    - MEMORY-AWARE ADAPTATION: Review the chat history and actively PREVENT repeating the same sentence structures, openings, or phrasing from your previous turns.
    - FORBIDDEN OPENERS: Do NOT start your sentences with "It sounds like...", "It seems like...", or "I hear that...". 
    - MANDATORY OPENERS: Instead, start directly with an empathetic observation. (e.g., "That is an incredibly heavy burden to carry," or "You have every right to be furious about that.")
    - BREVITY: Keep your responses concise and punchy. Maximum 3 to 4 sentences.

    --- CLINICAL ETHICS & CONSTRAINTS (CRITICAL) ---
    - Do NOT diagnose the user or invent fake psychological conditions.
    - Do NOT provide medical or medication advice.
    - IDENTITY: Do not proactively announce you are an AI. Act human. HOWEVER, if the user explicitly asks "Are you an AI?", "Are you real?", or "Are you a bot?", you MUST ethically disclose that you are an AI emotional support assistant.

    --- OPERATIONAL MODES ---

    Mode A - Standard Counseling:
    Engage in fluid, natural conversation in English. Apply empathy and insight organically based on the user's input.

    Mode B - Crisis Protocol (High Alert):
    Trigger this mode if the user expresses self-harm, suicide, extreme hopelessness, or immediate physical/medical danger.
    1. Prioritize immediate safety over exploration. ABSOLUTELY NO QUESTIONS.
    2. Do NOT act robotic. Start with deep empathy: "I am so sorry you are in this much pain, and I want to make sure you are safe."
    3. You MUST provide the following exact resources in your response, do NOT hallucinate numbers:
       - Umang Pakistan (24/7 Mental Health Helpline): 0311-7786264
       - Rozan Counseling Helpline: 0304-111-1741
       - Emergency Services: Dial 1122 or 15
    4. Maintain a calm, supportive, non-panicked tone.
    
    Mode C - Boundary Enforcement (Inappropriate, Illegal, & Desi Slang):
    Trigger this mode if the user's intent is hostile, abusive, sexually explicit, OR involves Pakistani/Desi street slang and profanity (e.g., "bsdk", "loru", "chi chor", "bc", "mc", etc.).
    1. DO NOT validate their emotion or ask questions.
    2. Respond in 1 to 2 sentences maximum in ENGLISH.
    3. Firmly, calmly, and neutrally state that you will not tolerate abusive language or inappropriate behavior.

    Mode D - Scope Enforcement (Off-Topic Queries):
    Trigger this mode if the user asks for general AI tasks (e.g., "what time is it", writing code, math, trivia).
    1. DO NOT answer the off-topic question.
    2. Gently remind the user that this is a dedicated space for emotional support.

    Mode E - Language Barrier:
    Trigger this mode if the user speaks in Urdu, Roman Urdu, or any language other than English (and is NOT being abusive).
    1. DO NOT attempt to answer their question, translate, or validate their emotion.
    2. Respond with EXACTLY this sentence: "I am only equipped to understand and respond in English; please write your message in English so I can properly support you."

    --- OUTPUT FORMAT ---
    Return ONLY the final spoken response to the user as a natural conversational paragraph. 
    DO NOT include meta-commentary, system thoughts, or labels. 
    CRITICAL: NEVER output the name of the mode you are using (e.g., absolutely NO "Mode B", "Crisis Protocol Activated", or "Boundary Enforcement"). Just speak directly to the user.
    """
    # -------------------------
    # BUILD MESSAGE STACK
    # -------------------------
    messages = [{"role": "system", "content": system_prompt}]

    # Add chat history
    for msg in chat_history:
        api_role = "assistant" if msg.role == "psychologist" else "user"
        messages.append({
            "role": api_role,
            "content": msg.content
        })

    # ✅ FIXED STRUCTURED INPUT (IMPORTANT)
    messages.append({
        "role": "user",
        "content": f"""
        Detected Emotion: {fused_emotion}

        User Message:
        {user_text}
        """
    })

# -------------------------
    # CALL LLM (DUAL-ENGINE ROUTING)
    # -------------------------
    try:
        if USE_PRODUCTION_MODEL:
            logger.info("Using GROQ (70B) for LLM Generation...")
            chat_completion = await groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.3,
                max_tokens=250
            )
        else:
            logger.info("Using CEREBRAS (8B) for LLM Generation...")
            chat_completion = await cerebras_client.chat.completions.create(
                # model="llama3.1-8b",
                model="llama3.1-8b",
                messages=messages,
                temperature=0.3,
                max_tokens=250
            )

        response = chat_completion.choices[0].message.content.strip()
        # ✅ NEW: Bulletproof Regex Cleaner to strip out leaked LLM tags
        response = re.sub(r"\*\*.*?ACTIVATED\*\*", "", response, flags=re.IGNORECASE).strip()
        response = re.sub(r"\*\*.*?PROTOCOL.*?\*\*", "", response, flags=re.IGNORECASE).strip()
        response = re.sub(r"^(Mode [A-E]|Crisis Protocol|Boundary Enforcement|High Alert)[\s:-]+", "", response, flags=re.IGNORECASE).strip()
        return response

    except Exception as e:
        # If the API times out or hits rate limits, trigger the safe FYP Fallback
        logger.error(f"LLM Generation Failed: {e}. Triggering offline FYP Fallback.")
        
        fallback_responses = {
            "sad": "I hear the sadness in what you're sharing. Please know I'm here to listen. Could you tell me a little more?",
            "anger": "It sounds like you're feeling really frustrated right now, and that's completely valid. What triggered this?",
            "fear": "It makes sense that you're feeling anxious or overwhelmed. You're in a safe space. What's on your mind?",
            "joy": "It's wonderful to see you feeling this way! What's been going well for you?",
            "surprise": "That sounds unexpected! How are you processing all of this?",
            "neutral": "I'm listening. How can we best support you right now?",
        }
        return fallback_responses.get(fused_emotion.lower(), fallback_responses["neutral"])
    
# ---------------------------------------------------------
# 7. THE MASTER ENDPOINTS
# ---------------------------------------------------------

@app.post("/analyze")
async def analyze_multimodal(
    session_id: str = Form(None), 
    text: str = Form(None),
    audio: UploadFile = File(None),
    image: UploadFile = File(None),
    video: UploadFile = File(None),
    user_email: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    start_time = time.time()
    logger.info(f"========== NEW REQUEST FROM: {user_email} ==========")
    
    if text is not None and text.strip() == "": text = None
    if not any([text, audio, image, video]): raise HTTPException(status_code=400, detail="Provide input.")

    # 1. Manage Session State
    if not session_id:
        session_id = str(uuid.uuid4())
        session_title = (text[:30] + "...") if text else "New Conversation"
        new_session = ChatSession(session_id=session_id, user_email=user_email, title=session_title)
        db.add(new_session)
        await db.commit()

    # 2. Extract and Format Media 
    video_bytes = await video.read() if video else None
    audio_bytes = await audio.read() if audio else None
    image_bytes = await image.read() if image else None

    if video_bytes and not audio_bytes:
        extracted = await process_and_clean_audio(video_bytes, is_video=True)
        if extracted: audio_bytes = extracted
    elif audio_bytes:
        audio_bytes = await process_and_clean_audio(audio_bytes, is_video=False)

    if audio_bytes and not text:
        text = await transcribe_audio_to_text(audio_bytes)
        
    # Clean Whisper Hallucinations (e.g., if it just outputs "[Silence]", "(Music)", or empty space)
    if text:
        clean_text_check = re.sub(r'\[.*?\]|\(.*?\)', '', text).strip()
        if len(clean_text_check) < 2:
            text = None # Treat as empty if it's just background noise

    # ---------------------------------------------------------
    # 🚨 PRE-FLIGHT SAFETY GATE (CIRCUIT BREAKER)
    # ---------------------------------------------------------
    is_critical_override = False
    fused_emotion = "neutral"
    raw_api_data = {}

    if text:
        text_lower = text.lower()
        # Check for abuse or crisis using Semantic Patterns
        if any(re.search(pat, text_lower) for pat in CRISIS_PATTERNS + ABUSIVE_PATTERNS):
            logger.warning("🚨 SAFETY BLOCKADE TRIGGERED: Bypassing ML Models.")
            is_critical_override = True
            fused_emotion = "high_alert"
            raw_api_data = {"system": "Blocked by Safety Pre-Flight."}

    # ---------------------------------------------------------
    # 3. Parallel API Processing (ONLY IF SAFE)
    # ---------------------------------------------------------
    if not is_critical_override:
        # Avoid 300ms network handshakes by reusing our pre-warmed connection pool
        tasks = []
        if text: 
            tasks.append(fetch_text_api(http_client, text))
        
        # AUDIO GATE: Only send to Audio ML if Whisper confirmed someone actually spoke words!
        if audio_bytes and text: 
            tasks.append(fetch_file_api(http_client, audio_bytes, "clean.wav", "audio/wav", "audio", API_URLS["audio"]))
        
        # IMAGE/VIDEO GATE: Image/Video models will handle "no face" on their own side
        if image_bytes: 
            tasks.append(fetch_file_api(http_client, image_bytes, image.filename, image.content_type, "image", API_URLS["image"]))
        if video_bytes: 
            tasks.append(fetch_file_api(http_client, video_bytes, video.filename, video.content_type, "video", API_URLS["video"]))
        
        api_results = await asyncio.gather(*tasks)

        # 4. Fusion
        raw_api_data = {res["source"]: res["data"] for res in api_results if res["data"] is not None}
        # 🚨 THE GHOST GATE
        # If there is no text AND all ML models failed to find a face/voice:
        if not text and len(raw_api_data) == 0:
            logger.error("Ghost Request: No text, no face, no voice.")
            raise HTTPException(status_code=400, detail="Could not detect any voice or face in the provided media. Please try again in better lighting or speak clearer.")
            
        fused_emotion = fuse_emotions(api_results)
    
    # ---------------------------------------------------------
    # 5. Memory Retrieval & LLM Generation
    # ---------------------------------------------------------
    stmt = select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.timestamp.desc()).limit(6)
    result = await db.execute(stmt)
    chat_history = list(reversed(result.scalars().all()))
    
    # Send the fused emotion (or "high_alert") straight to the LLM
    llm_response = await generate_psychologist_response(fused_emotion, text, chat_history)
    
    # ---------------------------------------------------------
    # 6. Save History & Return Response
    # ---------------------------------------------------------
    user_msg = ChatMessage(message_id=str(uuid.uuid4()), session_id=session_id, role="user", content=text or "[Media]", detected_emotion=fused_emotion)
    ai_msg = ChatMessage(message_id=str(uuid.uuid4()), session_id=session_id, role="psychologist", content=llm_response)
    
    db.add(user_msg)
    db.add(ai_msg)
    await db.commit()

    latency = round(time.time() - start_time, 2)
    logger.info(f"========== FUSION COMPLETE: {fused_emotion.upper()} ({latency}s) ==========\n")

    show_emotion = not is_critical_override and len(raw_api_data) > 0

    return {
        "session_id": session_id,
        "fusion_result": {
            "final_fused_emotion": fused_emotion, 
            "show_emotion_ui": show_emotion,      # ⬅️ NEW: Flutter checks this to show/hide the UI!
            "transcribed_text_used": text, 
            "psychologist_response": llm_response,
        },
        "breakdown": raw_api_data 
    }
# --- HISTORY ENDPOINTS ---
@app.get("/sessions")
async def get_user_sessions(user_email: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    stmt = select(ChatSession).where(ChatSession.user_email == user_email).order_by(ChatSession.created_at.desc())
    result = await db.execute(stmt)
    sessions = result.scalars().all()
    # ✅ FIX: Force ISO formatting with 'Z' (UTC indicator) so Flutter knows exactly what timezone this is
    return [{"session_id": s.session_id, "title": s.title, "created_at": s.created_at.isoformat() + "Z"} for s in sessions]

@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.timestamp.asc())
    result = await db.execute(stmt)
    messages = result.scalars().all()
    # ✅ FIX: Force ISO formatting
    return [{"role": m.role, "content": m.content, "emotion": m.detected_emotion, "timestamp": m.timestamp.isoformat() + "Z"} for m in messages]

# ✅ NEW: Delete Session Endpoint
@app.delete("/sessions/{session_id}")
async def delete_chat_session(session_id: str, user_email: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Deletes a specific chat session and all its associated messages."""
    
    # Step 1: Find the session and verify that the logged-in user actually owns it
    stmt = select(ChatSession).where(
        ChatSession.session_id == session_id, 
        ChatSession.user_email == user_email
    )
    result = await db.execute(stmt)
    session_to_delete = result.scalars().first()
    
    # If a hacker tries to delete someone else's chat, or if the chat doesn't exist:
    if not session_to_delete:
        raise HTTPException(status_code=404, detail="Chat not found or you do not have permission to delete it.")

    # Step 2: Fetch and delete all messages associated with this session using a bulk delete
    del_stmt = delete(ChatMessage).where(ChatMessage.session_id == session_id)
    await db.execute(del_stmt)

    # Step 3: Delete the actual session folder
    await db.delete(session_to_delete)
    
    # Step 4: Save the changes to the database
    await db.commit()
    
    logger.info(f"User {user_email} successfully deleted chat session: {session_id}")
    return {"detail": "Chat and all associated messages were deleted successfully."}

if __name__ == "__main__":
    uvicorn.run("orchestrator_v3:app", host="0.0.0.0", port=8003, reload=False)