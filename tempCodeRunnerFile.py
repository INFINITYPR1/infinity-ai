import asyncio
import os
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from openai import OpenAI

# Initialize local OpenAI client (Ollama backend)
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'
)

# Choose a natural neural voice (e.g., British male 'en-GB-RyanNeural' or US 'en-US-ChristopherNeural')
VOICE_NAME = "en-GB-RyanNeural"

async def speak_async(text):
    """Generate and play natural neural speech using edge-tts"""
    print(f"\nInfinity: {text}\n")
    output_file = "temp_response.mp3"
    
    # Generate speech file
    communicate = edge_tts.Communicate(text, VOICE_NAME)
    await communicate.save(output_file)
    
    # Play audio natively on macOS using 'afplay'
    os.system(f"afplay {output_file}")

def speak(text):
    """Synchronous wrapper for async speech"""
    asyncio.run(speak_async(text))

def record_audio(duration=5, sample_rate=16000):
    """Record microphone input for a specified duration in seconds"""
    print(f"\n[Listening...] Speak now ({duration}s)...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    wav.write("temp_input.wav", sample_rate, audio)
    print("[Audio captured]")

def run_infinity_loop():
    speak("Infinity voice system online. Neural speech active. How can I assist you?")
    
    while True:
        try:
            record_audio(duration=5)
            
            user_input = input("Press ENTER to send audio command (or type 'exit' to quit): ")
            if user_input.lower() == 'exit':
                speak("Shutting down voice core.")
                break

            prompt_text = user_input if user_input.strip() != "" else "Status report."

            response = client.chat.completions.create(
                model="llama3.2",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are Infinity, an elite personal AI assistant. Keep all responses brief (1-2 sentences) so they sound natural when spoken."
                    },
                    {"role": "user", "content": prompt_text}
                ]
            )

            reply = response.choices[0].message.content
            speak(reply)

        except KeyboardInterrupt:
            speak("Voice system offline.")
            break

if __name__ == "__main__":
    run_infinity_loop()