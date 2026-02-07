import pandas as pd
import os
from gtts import gTTS
from speech_recognition import Recognizer, Microphone, AudioData
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
import sounddevice as sd
import numpy as np
from word2number import w2n
from affordability_model import (
    load_historical_data, 
    train_model, 
    load_sublets, 
    recommend_apartments
)

# -------------------------
# ElevenLabs Setup
# -------------------------
ELEVENLABS_API_KEY = "sk_282178210d030b5f4510ef76647e9620e4536efef1fea724"
# Initialize client
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

all_voices = client.voices.get_all()

for v in all_voices:
    print(v)


# Hardcoded voice IDs for Rachel and Sean
# You can find these in your ElevenLabs Dashboard or via client.voices.get_all()
voices = {
    "friendly": "CwhRBWXzGAHq8TQ4Fs17",      # Roger
    "professional": "EXAVITQu4vr4xnSDxMaL"    # Sarah
}

# Verification check
if not voices["friendly"]:
    print("Friendly voice ID missing!")

# Step 4: Check if voices were found
if not "friendly":
    print("Friendly voice 'Rachel' not found!")
if not "professional":
    print("Professional voice 'Sean' not found!")

# -------------------------
# Speech Recognition Setup
# -------------------------
recognizer = Recognizer()

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
def speak_text(text, tone="neutral"):
    """Speak text using selected tone: friendly, professional, or neutral."""
    if tone in ["friendly", "professional"]:
        # Get the ID string from your voices dictionary
        voice_id = voices[tone]
        
        # Correct SDK call: Use the client's text_to_speech.convert method
        audio_generator = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2", # Standard high-quality model
            output_format="mp3_44100_128"      # Optional: set quality
        )
        
        # client.text_to_speech.convert returns a generator; join chunks into bytes
        audio_bytes = b"".join(audio_generator)
        
        with open("temp.mp3", "wb") as f:
            f.write(audio_bytes)
        
        # Play the file using Mac's afplay
        os.system("afplay temp.mp3")
        os.remove("temp.mp3")
    else:
        # Neutral uses Google TTS (gTTS)
        tts = gTTS(text=text, lang="en")
        filename = "temp.mp3"
        tts.save(filename)
        os.system(f"afplay {filename}")
        os.remove(filename)
# -------------------------
# Ask Question Helper
# -------------------------
def ask_question(prompt, tone="neutral"):
    """
    Ask a question via TTS and get answer via voice or text input.
    """
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
        # text2num handles "1 million", "8", and "eight" natively
        return float(text2num(val, lang="en"))
    except ValueError:
        # Fallback for standard float parsing
        try:
            return float(val)
        except ValueError:
            print(f"⚠️ Could not parse '{val}', defaulting to 0.0")
            return 0.0

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
# Predict Safe Rent
# -------------------------
def predict_safe_rent(model, user_input):
    """Convert user input into DataFrame and predict safe rent."""
    input_df = pd.DataFrame([user_input])
    safe_rent = model.predict(input_df)[0]
    return safe_rent

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



