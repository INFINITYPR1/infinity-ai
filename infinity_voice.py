import asyncio
import os
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import edge_tts
import whisper
from openai import OpenAI

# 1. Initialize local OpenAI client (Ollama backend)
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'
)

# 2. Load Whisper local model on startup (uses small 'base' model for fast response)
print("Loading Whisper STT model...")
stt_model = whisper.load_model("base")

VOICE_NAME = "en-GB-RyanNeural"

async def speak_async(text):
    """Generate and play natural neural speech using edge-tts"""
    print(f"\nInfinity: {text}\n")
    output_file = "temp_response.mp3"
    communicate = edge_tts.Communicate(text, VOICE_NAME)
    await communicate.save(output_file)
    os.system(f"afplay {output_file}")

def speak(text):
    """Synchronous wrapper for async speech"""
    asyncio.run(speak_async(text))

def record_audio(duration=5, sample_rate=16000):
    """Record microphone input"""
    print(f"\n[Listening...] Speak now ({duration}s)...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    file_path = "temp_input.wav"
    wav.write(file_path, sample_rate, audio)
    print("[Processing audio...]")
    return file_path

def run_infinity_loop():
    speak("Infinity voice system online. I am listening.")
    
    while True:
        try:
            # Record 5 seconds of spoken audio
            audio_file = record_audio(duration=5)
            
            # Transcribe recorded voice to text using Whisper
            result = stt_model.transcribe(audio_file)
            user_text = result["text"].strip()
            
            if not user_text:
                print("(No speech detected)")
                continue

            print(f"\nYou said: '{user_text}'")

            # Exit command check
            if "exit" in user_text.lower() or "shutdown" in user_text.lower():
                speak("Shutting down voice core. Goodbye!")
                break

            # Send transcribed speech to Ollama
            response = client.chat.completions.create(
                model="llama3.2",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are Infinity, an elite personal AI assistant. Keep responses brief (1-2 sentences) so they sound natural when spoken."
                    },
                    {"role": "user", "content": user_text}
                ]
            )

            reply = response.choices[0].message.content
            speak(reply)

        except KeyboardInterrupt:
            speak("Voice system offline.")
            break

if __name__ == "__main__":
    run_infinity_loop()