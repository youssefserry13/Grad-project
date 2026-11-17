from flask import Flask, request, jsonify
import base64
import wave
import ollama
from faster_whisper import WhisperModel
from difflib import SequenceMatcher

app = Flask(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────
OLLAMA_HOST = "http://127.0.0.1:11434"   # Ollama now runs inside WSL, same as this script
TEXT_MODEL = "qwen3:8b"
VISION_MODEL = "qwen2.5vl:7b"
WHISPER_MODEL = "large-v3-turbo"
PORT = 5001
WAKE_WORD = "nova"
WAKE_THRESHOLD = 0.85

client = ollama.Client(host=OLLAMA_HOST)

# ─── Load Whisper once at startup ────────────────────────────────────────────
print("🔄 Loading Whisper model...")
whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
print("✅ Whisper ready on GPU")


# ─── Fuzzy Wake Word Detection ───────────────────────────────────────────────
def is_similar(word, target=WAKE_WORD, threshold=WAKE_THRESHOLD):
    return SequenceMatcher(None, word.lower(), target).ratio() >= threshold


def wake_word_detected(transcript):
    return any(is_similar(w.strip(".,!?").lower()) for w in transcript.split())


def extract_command(transcript):
    words = transcript.split()
    for i, word in enumerate(words):
        if is_similar(word.strip(".,!?").lower()):
            command = " ".join(words[i + 1:]).strip()
            return command if command else "Hello, what can you help me with?"
    return transcript


# ─── Audio Endpoint ──────────────────────────────────────────────────────────
@app.route("/audio", methods=["POST"])
def handle_audio():
    data = request.json
    if data.get("state") != 101:
        return "Error: Wrong State", 400

    audio_bytes = base64.b64decode(data["payload"])
    print(f"✅ Received {len(audio_bytes)} bytes of audio")

    # Detect format by magic bytes
    is_wav = audio_bytes[:4] == b"RIFF"
    is_m4a = audio_bytes[4:8] == b"ftyp" if len(audio_bytes) > 8 else False

    if is_wav:
        path = "received_audio.wav"
        with open(path, "wb") as f:
            f.write(audio_bytes)
        print("📁 Detected: WAV file")
    elif is_m4a:
        path = "received_audio.m4a"
        with open(path, "wb") as f:
            f.write(audio_bytes)
        print("📁 Detected: M4A file")
    else:
        # Raw PCM from ESP32 (16 kHz, mono, 16-bit) → wrap in WAV header
        path = "received_audio.wav"
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_bytes)
        print("📁 Detected: Raw PCM from ESP32, wrapped in WAV")

    # Transcribe (auto language detection)
    segments, info = whisper.transcribe(path)
    transcript = " ".join(s.text for s in segments).strip().lower()
    print(f"🎤 Transcribed ({info.language}): '{transcript}'")

    if not wake_word_detected(transcript):
        print("💤 Wake word not detected, ignoring.")
        return jsonify({"status": "ignored", "transcript": transcript}), 200

    command = extract_command(transcript)
    print(f"💬 Command: '{command}'")

    print("🤖 Querying Ollama...")
    response = client.chat(
        model=TEXT_MODEL,
        keep_alive=-1,
        messages=[{"role": "user", "content": command}],
        think=False,  # thinking off for voice: latency matters
    )
    reply = response["message"]["content"]
    print(f"🤖 Ollama: {reply}")

    return jsonify({"status": "ok", "transcript": command, "reply": reply}), 200


# ─── Image Endpoint ──────────────────────────────────────────────────────────
@app.route("/image", methods=["POST"])
def handle_image():
    data = request.json
    if data.get("state") != 102:
        return "Error: Wrong State", 400

    image_bytes = base64.b64decode(data["payload"])
    with open("received_frame.jpg", "wb") as f:
        f.write(image_bytes)
    print(f"📷 Received image frame ({len(image_bytes)} bytes)")

    command = data.get("command", "Describe what you see in this image.")
    print(f"💬 Vision command: '{command}'")

    image_b64 = base64.b64encode(image_bytes).decode()
    response = client.chat(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": command, "images": [image_b64]}],
    )
    reply = response["message"]["content"]
    print(f"🤖 Vision: {reply}")

    return jsonify({"status": "ok", "reply": reply}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
