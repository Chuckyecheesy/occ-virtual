import os
import tempfile
import threading
import subprocess
from collections import deque
from functools import lru_cache
import json

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from gtts import gTTS
from speech_recognition import Recognizer, AudioData
from word2number import w2n
from elevenlabs.client import ElevenLabs

from affordability_model import (
    load_historical_data,
    train_model,
    load_sublets,
    recommend_apartments,
    predict_safe_rent
)

load_dotenv()  # ✅ Must be first before accessing os.getenv

# -------------------------
# ElevenLabs Setup (lazy to reduce latency on import)
# -------------------------
def _parse_voice_list(env_key, fallback_key=None):
    raw = os.getenv(env_key, "")
    if raw:
        return [v.strip() for v in raw.split(",") if v.strip()]
    if fallback_key:
        fallback = os.getenv(fallback_key)
        return [fallback] if fallback else []
    return []

_VOICE_IDS = {
    "friendly": _parse_voice_list("VOICE_FRIENDLY_IDS", "VOICE_FRIENDLY"),
    "professional": _parse_voice_list("VOICE_PROFESSIONAL_IDS", "VOICE_PROFESSIONAL"),
    "neutral": _parse_voice_list("VOICE_NEUTRAL_IDS")
}

@lru_cache(maxsize=1)
def _get_elevenlabs_client():
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        return None
    return ElevenLabs(api_key=api_key)

def _get_voice_id(tone: str):
    voice_list = _VOICE_IDS.get(tone, [])
    if not voice_list:
        return None
    return voice_list[0]

# -------------------------
# Speech Recognition Setup
# -------------------------
recognizer = Recognizer()

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import requests
except ImportError:
    requests = None

def listen_and_transcribe(duration=5, fs=44100):
    """
    Record audio using sounddevice and transcribe via Google STT.
    """
    print(f"Listening for {duration} seconds... speak now.")
    try:
        # Record audio
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()

        # Convert to int16 bytes
        audio_int16 = np.int16(recording * 32767)
        audio_bytes = audio_int16.tobytes()

        # Wrap in AudioData for Recognizer
        audio_data = AudioData(audio_bytes, fs, 2)  # 2 bytes per sample

        # Recognize speech
        text = recognizer.recognize_google(audio_data)
        print("You said:", text)
        return text

    except Exception as e:
        print("Could not understand audio:", e)
        return ""

# -------------------------
# Text-to-Speech Function
# -------------------------
_AUDIO_QUEUE = deque()
_AUDIO_QUEUE_LOCK = threading.Lock()
_AUDIO_QUEUE_EVENT = threading.Event()
_AUDIO_THREAD = None

def _audio_worker():
    while True:
        _AUDIO_QUEUE_EVENT.wait()
        while True:
            with _AUDIO_QUEUE_LOCK:
                if not _AUDIO_QUEUE:
                    _AUDIO_QUEUE_EVENT.clear()
                    break
                path = _AUDIO_QUEUE.popleft()
            _play_audio_file(path, async_playback=False)
            try:
                os.remove(path)
            except OSError:
                pass

def _ensure_audio_worker():
    global _AUDIO_THREAD
    if _AUDIO_THREAD and _AUDIO_THREAD.is_alive():
        return
    _AUDIO_THREAD = threading.Thread(target=_audio_worker, daemon=True)
    _AUDIO_THREAD.start()

def _enqueue_audio(path):
    _ensure_audio_worker()
    with _AUDIO_QUEUE_LOCK:
        _AUDIO_QUEUE.append(path)
        _AUDIO_QUEUE_EVENT.set()

def _cleanup_audio_file(path, process, async_playback):
    if async_playback:
        _enqueue_audio(path)
    else:
        try:
            os.remove(path)
        except OSError:
            pass

def _play_audio_file(path, async_playback=False):
    if async_playback:
        process = subprocess.Popen(["afplay", path])
        return process
    subprocess.run(["afplay", path], check=False)
    return None

def speak_text(text, tone="neutral", async_playback=False, voice_id=None):
    # Always fall back to gTTS if ElevenLabs fails.
    def _play_gtts():
        tts = gTTS(text=text, lang="en")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            tts.save(f.name)
            tmp_file = f.name
        if async_playback:
            _cleanup_audio_file(tmp_file, None, async_playback=True)
        else:
            process = _play_audio_file(tmp_file, async_playback=False)
            _cleanup_audio_file(tmp_file, process, async_playback=False)

    try:
        if voice_id is None:
            voice_id = _get_voice_id(tone)
        client = _get_elevenlabs_client()
        if tone in ["friendly", "professional", "neutral"] and voice_id and client:
            try:
                audio_generator = client.text_to_speech.convert(
                    text=text,
                    voice_id=voice_id,
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128"
                )
                audio_bytes = b"".join(audio_generator)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    f.write(audio_bytes)
                    tmp_file = f.name
                if async_playback:
                    _cleanup_audio_file(tmp_file, None, async_playback=True)
                else:
                    process = _play_audio_file(tmp_file, async_playback=False)
                    _cleanup_audio_file(tmp_file, process, async_playback=False)
            except Exception as e:
                print(f"[ElevenLabs TTS failed] {e}")
                _play_gtts()
        else:
            _play_gtts()
    except Exception as e:
        print(f"[TTS failed] {e}")
        print(text)


# -------------------------
# Ask Question Helper
# -------------------------
def ask_question(prompt, tone="neutral"):
    """Ask a question via TTS and return text answer"""
    print(prompt)
    speak_text(prompt + " You can speak or type your answer.", tone=tone)
    choice = input(f"{prompt} (type 'speak' to answer by voice, anything else to type): ").strip().lower()
    if choice == "speak":
        answer = listen_and_transcribe()
        while not answer:
            speak_text("Sorry, I did not catch that. Please try again.", tone=tone)
            answer = listen_and_transcribe()
        return answer
    else:
        return input(prompt + " ")

def clean_float(val):
    """Robustly converts mixed strings (e.g., '1 million', 'eight') to floats."""
    val = str(val).lower().strip().replace("$", "").replace(",", "")
    
    if not val or val in ["none", "nothing", "nil"]:
        return 0.0

    try:
        return float(val)
    except ValueError:
        try:
            return float(w2n.word_to_num(val))
        except ValueError:
            gemini_val = _gemini_parse_number(val)
            if gemini_val is not None:
                return gemini_val
            print(f"⚠️ Could not parse '{val}', defaulting to 0.0")
            return 0.0

@lru_cache(maxsize=1)
def _get_gemini_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or genai is None:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

def _gemini_parse_number(text):
    """
    Use Gemini to parse spoken-number text into a numeric value.
    Returns float or None when unavailable.
    """
    model = _get_gemini_model()
    if model is None:
        return None
    prompt = (
        "Convert the following spoken number into a numeric value. "
        "Return only the number with no extra text.\n"
        f"Input: {text}"
    )
    try:
        result = model.generate_content(prompt)
        raw = (result.text or "").strip()
        return float(raw)
    except Exception:
        return None

def openrouter_clarify_number(text):
    """
    Use OpenRouter to clarify spoken input into a numeric value.
    Returns dict: {"normalized": str|None, "number": float|None, "confidence": float|None}
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or requests is None:
        return None

    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    system_prompt = (
        "You convert spoken financial inputs to numeric values. "
        "Return ONLY valid JSON with keys: normalized, number, confidence."
    )
    user_prompt = f"Input: {text}"
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "OCCVirtual"
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.0
            },
            timeout=15
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        data = json.loads(content)
        return {
            "normalized": data.get("normalized"),
            "number": data.get("number"),
            "confidence": data.get("confidence")
        }
    except Exception:
        return None

def gather_user_input(tone="neutral"):
    """Gather financial and housing info from the user."""
    speak_text("Please provide the following details for your rent calculation.", tone=tone)

    # Now all calls below are properly aligned and reachable
    tuition = clean_float(ask_question("Enter your annual tuition fee:", tone=tone))
    bank_balance = clean_float(ask_question("Enter your current bank balance:", tone=tone))
    part_time_income = clean_float(ask_question("Enter your monthly part-time income:", tone=tone))
    internship_income = clean_float(ask_question("Enter your monthly internship income:", tone=tone))
    scholarships = clean_float(ask_question("Enter total received scholarships:", tone=tone))
    loans = clean_float(ask_question("Enter total available loans:", tone=tone))
    
    # Use clean_float then convert to int for months
    months_str = ask_question("Enter the number of months for which you need housing:", tone=tone)
    months = int(clean_float(months_str))

    return {
        'tuition': tuition,
        'bank_balance': bank_balance,
        'part_time_income': part_time_income,
        'internship_income': internship_income,
        'scholarships': scholarships,
        'loans': loans,
        'months': months
    }




# -------------------------
# Main Process
# -------------------------
def process_speech_interaction():
    # Ask user to choose voice tone
    print("Select voice tone: friendly, professional, neutral")
    selected_tone = input("Tone: ").strip().lower()
    if selected_tone not in ["friendly", "professional", "neutral"]:
        selected_tone = "neutral"

    speak_text("Let's start with your financial information.", tone=selected_tone)
    user_input = gather_user_input(tone=selected_tone)

    print("Loading historical data ...")
    df_train = load_historical_data()
    print("Training model ...")
    model = train_model(df_train)

    safe_rent = predict_safe_rent(model, user_input)
    print(f"Safe rent calculated: ${safe_rent:,.2f}")
    speak_text(f"Your suggested budget is ${safe_rent:,.2f} per month.", tone=selected_tone)

    print("Loading sublets ...")
    sublets_df = load_sublets()
    print("Generating recommendations ...")
    recommendations = recommend_apartments(safe_rent, sublets_df)

    if not recommendations.empty:
        addresses = ", ".join(recommendations["address"].tolist())
        print("Recommended apartments:", addresses)
        speak_text(f"Recommended housing options are: {addresses}", tone=selected_tone)
    else:
        print("No suitable housing options found.")
        speak_text("No suitable housing options were found within your budget.", tone=selected_tone)

# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    process_speech_interaction()



