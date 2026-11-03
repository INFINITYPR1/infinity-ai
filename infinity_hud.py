import asyncio
import os
import json
import re
import requests
import concurrent.futures
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
import whisper
import edge_tts
from openai import OpenAI
from duckduckgo_search import DDGS

app = FastAPI()

# Initialize local OpenAI client (Ollama backend)
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'
)

print("Loading Whisper STT model for HUD...")
stt_model = whisper.load_model("base")
VOICE_NAME = "en-GB-RyanNeural"
AI_MODEL = "llama3.1"

MEMORY_FILE = "memory.json"

def load_memories():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                data = json.load(f)
                return data.get("user_facts", [])
        except Exception:
            return []
    return []

def save_memory(fact):
    facts = load_memories()
    if fact not in facts:
        facts.append(fact)
        with open(MEMORY_FILE, "w") as f:
            json.dump({"user_facts": facts}, f, indent=4)

def get_kiel_weather():
    """Instant weather fetch via free Open-Meteo API for Kiel"""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": 54.3217,
            "longitude": 10.1348,
            "current": ["temperature_2m", "wind_speed_10m"]
        }
        res = requests.get(url, params=params, timeout=2.0).json()
        current = res.get("current", {})
        temp = current.get("temperature_2m", "N/A")
        wind = current.get("wind_speed_10m", "N/A")
        return f"Current Kiel Weather: {temp}°C, Wind Speed: {wind} km/h."
    except Exception as e:
        print(f"[Weather API Error] {e}")
        return None

def web_search(query):
    """Fallback search with strict 4-second hard timeout to prevent server hanging"""
    def _search():
        try:
            with DDGS() as ddgs:
                results = [r.get('body', '') for r in ddgs.text(query, max_results=2)]
                return "\n".join(results) if results else "No live results found."
        except Exception as e:
            return f"Search error: {str(e)}"

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_search)
        try:
            return future.result(timeout=4.0)
        except concurrent.futures.TimeoutError:
            print("[Infinity Tool] Search timed out after 4 seconds.")
            return "Web search timed out."

def get_system_prompt():
    facts = load_memories()
    facts_str = "\n".join([f"- {f}" for f in facts]) if facts else "None yet."
    return {
        "role": "system",
        "content": (
            "You are Infinity, an elite personal AI assistant powered by local intelligence. Keep responses brief, sharp, and conversational (1-2 sentences max).\n\n"
            f"Permanent User Memories & Facts:\n{facts_str}\n\n"
            "Capabilities:\n"
            "1. If the user tells you to remember a permanent fact or detail, include '[SAVE: <fact>]' anywhere in your response.\n"
            "2. If you need real-time data, current events, or web information to answer a user's question, output '[SEARCH: <search query>]' and wait for results."
        )
    }

# Short-term chat history buffer
conversation_history = []

async def handle_ai_response(messages):
    last_user_msg = messages[-1]["content"].lower()
    
    # Fast-path: Direct weather check without unnecessary search
    if "weather" in last_user_msg:
        weather_data = get_kiel_weather()
        if weather_data:
            messages.append({"role": "system", "content": f"Live Fast-Path Data: {weather_data}"})

    # Pass 1: Get model response
    response = client.chat.completions.create(
        model=AI_MODEL,
        messages=messages
    )
    raw_reply = response.choices[0].message.content
    
    # Check if Infinity requested a web search
    search_match = re.search(r'\[SEARCH:\s*(.*?)\]', raw_reply, re.IGNORECASE)
    if search_match:
        query = search_match.group(1).strip()
        print(f"[Infinity Tool] Executing live web search for: '{query}'")
        search_results = web_search(query)
        
        messages.append({"role": "assistant", "content": raw_reply})
        messages.append({"role": "system", "content": f"Live Web Search Results for '{query}':\n{search_results}\n\nProvide a final concise answer based on these results."})
        
        # Pass 2: Final response with web data included
        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=messages
        )
        raw_reply = response.choices[0].message.content

    # Extract [SAVE: ...] tags and save them to memory.json
    save_matches = re.findall(r'\[SAVE:\s*(.*?)\]', raw_reply, re.IGNORECASE)
    for match in save_matches:
        save_memory(match.strip())
    
    # Clean all system tags out of the final reply sent to text/voice
    clean_reply = re.sub(r'\[(SAVE|SEARCH):\s*.*?\]', '', raw_reply).strip()
    return clean_reply

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Infinity AI Core - HUD</title>
    <style>
        body { 
            background-color: #0b0f19; 
            color: #00ffcc; 
            font-family: 'Courier New', monospace; 
            margin: 0; 
            padding: 0; 
            display: flex; 
            flex-direction: column; 
            align-items: center; 
            justify-content: center; 
            height: 100vh; 
            overflow: hidden; 
        }
        .hud-container { 
            width: 850px; 
            border: 2px solid #00ffcc; 
            border-radius: 12px; 
            padding: 20px; 
            background: rgba(0, 20, 30, 0.85); 
            box-shadow: 0 0 30px rgba(0, 255, 204, 0.3); 
            display: flex; 
            flex-direction: column; 
            height: 700px; 
            position: relative;
        }
        .header-section {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px dashed #005544;
            padding-bottom: 10px;
            margin-bottom: 10px;
        }
        h1 { 
            letter-spacing: 4px; 
            margin: 0; 
            font-size: 24px;
            text-shadow: 0 0 12px #00ffcc; 
        }
        .status { 
            font-size: 12px; 
            color: #88aaff; 
        }
        .visualizer-box {
            height: 120px;
            width: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 10px;
            background: rgba(0, 5, 12, 0.6);
            border-radius: 6px;
            border: 1px solid #004433;
        }
        canvas {
            width: 100%;
            height: 100%;
        }
        .chat-box { 
            flex: 1; 
            border: 1px solid #008866; 
            border-radius: 6px; 
            padding: 15px; 
            overflow-y: auto; 
            margin-bottom: 15px; 
            background: rgba(0, 5, 10, 0.95); 
        }
        .message { 
            margin-bottom: 12px; 
            line-height: 1.5; 
        }
        .user { color: #ff007f; }
        .infinity { color: #00ffcc; }
        .controls { 
            display: flex; 
            gap: 10px; 
            align-items: center; 
        }
        button { 
            background: #00ffcc; 
            color: #0b0f19; 
            border: none; 
            padding: 12px 22px; 
            font-weight: bold; 
            cursor: pointer; 
            border-radius: 6px; 
            font-family: 'Courier New', monospace; 
            font-size: 16px; 
            transition: 0.2s; 
            box-shadow: 0 0 10px #00ffcc; 
        }
        button:hover { 
            background: #ff007f; 
            color: #fff; 
            box-shadow: 0 0 15px #ff007f; 
        }
        button.recording { 
            background: #ff007f; 
            color: #fff; 
            animation: pulse 1.2s infinite; 
        }
        .btn-clear {
            background: transparent;
            border: 1px solid #ff007f;
            color: #ff007f;
            padding: 6px 12px;
            font-size: 12px;
            box-shadow: none;
        }
        .btn-clear:hover {
            background: #ff007f;
            color: #fff;
        }
        input[type="text"] { 
            flex: 1; 
            padding: 12px; 
            background: #050b14; 
            border: 1px solid #00ffcc; 
            color: #00ffcc; 
            border-radius: 6px; 
            font-family: 'Courier New', monospace; 
            font-size: 16px; 
        }
        @keyframes pulse { 
            0% { opacity: 1; box-shadow: 0 0 10px #ff007f; } 
            50% { opacity: 0.4; box-shadow: 0 0 25px #ff007f; } 
            100% { opacity: 1; box-shadow: 0 0 10px #ff007f; } 
        }
    </style>
</head>
<body>
    <div class="hud-container">
        <div class="header-section">
            <h1>INFINITY // NEURAL HUD</h1>
            <button class="btn-clear" onclick="clearMemory()">Clear Memory</button>
        </div>
        <div class="status" id="status">System Status: ONLINE | Fast-Path APIs & Timeout Active</div>
        
        <div class="visualizer-box">
            <canvas id="visualizerCanvas"></canvas>
        </div>

        <div class="chat-box" id="chatBox">
            <div class="message infinity"><strong>Infinity:</strong> Optimizations loaded. Weather fast-path and strict search timeouts ready.</div>
        </div>
        
        <div class="controls">
            <input type="text" id="textInput" placeholder="Ask about weather, news, or type command..." onkeydown="if(event.key==='Enter') sendText()">
            <button id="micBtn" onclick="toggleRecord()">🎤 Auto VAD</button>
            <button onclick="sendText()">Send</button>
        </div>
    </div>

    <audio id="audioPlayer" style="display:none;"></audio>

    <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        
        let audioCtx;
        let analyser;
        let micSource;
        let silenceTimer = null;
        let hasSpoken = false;
        const SILENCE_THRESHOLD = 15; 
        const SILENCE_DURATION = 1500; 

        const canvas = document.getElementById('visualizerCanvas');
        const ctx = canvas.getContext('2d');
        const player = document.getElementById('audioPlayer');

        function resizeCanvas() {
            canvas.width = canvas.parentElement.clientWidth;
            canvas.height = canvas.parentElement.clientHeight;
        }
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

        function initAudioContext() {
            if (!audioCtx) {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioCtx.createAnalyser();
                analyser.fftSize = 128;
            }
            if (audioCtx.state === 'suspended') {
                audioCtx.resume();
            }
        }

        function checkSilence(dataArray) {
            if (!isRecording) return;

            let sum = 0;
            for (let i = 0; i < dataArray.length; i++) {
                sum += dataArray[i];
            }
            let averageVolume = sum / dataArray.length;

            if (averageVolume > SILENCE_THRESHOLD) {
                hasSpoken = true;
                if (silenceTimer) {
                    clearTimeout(silenceTimer);
                    silenceTimer = null;
                }
            } else if (hasSpoken && !silenceTimer) {
                document.getElementById('status').innerText = "Silence detected... finalizing speech";
                silenceTimer = setTimeout(() => {
                    if (isRecording) {
                        stopRecording();
                    }
                }, SILENCE_DURATION);
            }
        }

        function drawVisualizer() {
            requestAnimationFrame(drawVisualizer);
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const bufferLength = analyser ? analyser.frequencyBinCount : 0;
            const dataArray = new Uint8Array(bufferLength);

            if (analyser && isRecording) {
                analyser.getByteFrequencyData(dataArray);
                checkSilence(dataArray);
            }

            const barWidth = (canvas.width / (bufferLength || 32)) * 2;
            let barHeight;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                barHeight = (dataArray[i] / 255) * canvas.height;

                if (isRecording) {
                    ctx.fillStyle = `rgb(${dataArray[i] + 100}, 0, 127)`;
                } else {
                    ctx.fillStyle = `rgb(0, 180, 204)`;
                }

                ctx.fillRect(x, canvas.height - barHeight, barWidth - 2, barHeight);
                x += barWidth;
            }

            if (!isRecording || dataArray.every(val => val === 0)) {
                ctx.beginPath();
                ctx.strokeStyle = '#004433';
                ctx.lineWidth = 2;
                ctx.moveTo(0, canvas.height / 2);
                ctx.lineTo(canvas.width, canvas.height / 2);
                ctx.stroke();
            }
        }
        drawVisualizer();

        async function toggleRecord() {
            if (!isRecording) {
                startRecording();
            } else {
                stopRecording();
            }
        }

        async function startRecording() {
            initAudioContext();
            const micBtn = document.getElementById('micBtn');
            const status = document.getElementById('status');
            hasSpoken = false;

            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                micSource = audioCtx.createMediaStreamSource(stream);
                micSource.connect(analyser);

                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    if (micSource) micSource.disconnect();
                    status.innerText = "Transcribing audio via Whisper...";
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    const formData = new FormData();
                    formData.append("file", audioBlob, "voice_input.webm");

                    appendMessage("You (Voice)", "Processing voice input...", "user");

                    const response = await fetch('/process_voice', { method: 'POST', body: formData });
                    const data = await response.json();
                    
                    updateLastUserMessage(data.user_text);
                    appendMessage("Infinity", data.reply, "infinity");
                    playAudio(data.audio_url);
                    status.innerText = "System Status: ONLINE | Ready";
                };

                mediaRecorder.start();
                isRecording = true;
                micBtn.classList.add('recording');
                micBtn.innerText = "Listening...";
                status.innerText = "Listening... speak now";
            } catch (err) {
                alert("Microphone permission denied or unavailable.");
            }
        }

        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                isRecording = false;
                if (silenceTimer) clearTimeout(silenceTimer);
                const micBtn = document.getElementById('micBtn');
                micBtn.classList.remove('recording');
                micBtn.innerText = "🎤 Auto VAD";
            }
        }

        async function sendText() {
            const input = document.getElementById('textInput');
            const text = input.value.trim();
            if (!text) return;

            appendMessage("You", text, "user");
            input.value = "";
            document.getElementById('status').innerText = "Processing query...";

            const formData = new FormData();
            formData.append("text", text);

            const response = await fetch('/process_text', { method: 'POST', body: formData });
            const data = await response.json();

            appendMessage("Infinity", data.reply, "infinity");
            playAudio(data.audio_url);
            document.getElementById('status').innerText = "System Status: ONLINE | Complete";
        }

        async function clearMemory() {
            await fetch('/clear_memory', { method: 'POST' });
            document.getElementById('chatBox').innerHTML = '<div class="message infinity"><strong>Infinity:</strong> Chat buffer cleared. Long-term memory persists.</div>';
            document.getElementById('status').innerText = "System Status: ONLINE | Chat Cleared";
        }

        function appendMessage(sender, text, className) {
            const chatBox = document.getElementById('chatBox');
            const div = document.createElement('div');
            div.className = `message ${className}`;
            div.innerHTML = `<strong>${sender}:</strong> ${text}`;
            chatBox.appendChild(div);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        function updateLastUserMessage(text) {
            const chatBox = document.getElementById('chatBox');
            const messages = chatBox.getElementsByClassName('user');
            if (messages.length > 0) {
                messages[messages.length - 1].innerHTML = `<strong>You (Voice):</strong> ${text}`;
            }
        }

        function playAudio(url) {
            player.src = url + "?t=" + new Date().getTime();
            player.play();
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_hud():
    return HTML_CONTENT

@app.post("/process_text")
async def process_text(text: str = Form(...)):
    global conversation_history
    conversation_history.append({"role": "user", "content": text})

    messages = [get_system_prompt()] + conversation_history
    reply = await handle_ai_response(messages)
    
    conversation_history.append({"role": "assistant", "content": reply})

    audio_path = "static_response.mp3"
    communicate = edge_tts.Communicate(reply, VOICE_NAME)
    await communicate.save(audio_path)

    return {"reply": reply, "audio_url": "/get_audio"}

@app.post("/process_voice")
async def process_voice(file: UploadFile = File(...)):
    global conversation_history
    temp_audio = "temp_web_input.webm"
    with open(temp_audio, "wb") as buffer:
        buffer.write(await file.read())

    result = stt_model.transcribe(temp_audio)
    user_text = result["text"].strip()
    if not user_text:
        user_text = "[No speech detected]"

    conversation_history.append({"role": "user", "content": user_text})

    messages = [get_system_prompt()] + conversation_history
    reply = await handle_ai_response(messages)
    
    conversation_history.append({"role": "assistant", "content": reply})

    audio_path = "static_response.mp3"
    communicate = edge_tts.Communicate(reply, VOICE_NAME)
    await communicate.save(audio_path)

    return {"user_text": user_text, "reply": reply, "audio_url": "/get_audio"}

@app.post("/clear_memory")
async def clear_memory():
    global conversation_history
    conversation_history = []
    return {"status": "cleared"}

@app.get("/get_audio")
async def get_audio():
    return FileResponse("static_response.mp3", media_type="audio/mpeg")