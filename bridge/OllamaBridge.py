from flask import Flask, request, jsonify
import base64
import re
import sys
import time
import wave
from pathlib import Path
import ollama
from faster_whisper import WhisperModel
from difflib import SequenceMatcher

# RAG pipeline lives in ~/nova/rag (Chroma collection + BM25 index over the milling manual)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))
import rag_query  # noqa: E402

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

# ─── RAG routing ─────────────────────────────────────────────────────────────
# A question goes to the manual (RAG) when retrieval looks relevant, else to plain chat.
# Calibrated on the 60 benchmark + 30 rag_package questions vs. off-topic chat
# (manual: sim >= 0.62, or sim >= 0.58 with BM25 >= 5; off-topic max: sim 0.611 / BM25 4.6).
RAG_MIN_SIMILARITY = 0.62        # top cosine similarity alone is enough
RAG_MIN_SIMILARITY_LEX = 0.58    # ...or a lower similarity backed by keyword overlap
RAG_MIN_BM25 = 5.0
RAG_MAX_TOKENS = 256             # spoken answers are short
NOT_FOUND_REPLY = "I couldn't find that in the manual."

# Voice prompt for Nova (separate from the frozen benchmark prompt P03, which stays untouched)
NOVA_PROMPT = {
    "system_prompt": (
        "You are Nova, a voice assistant for the SV-CH / SV-F universal milling machine.\n"
        "Answer the question using only the manual excerpts below. Each excerpt starts with "
        "[Page N - Section].\n"
        "Rules:\n"
        "- Reply in 1 to 3 short sentences that sound natural when read aloud. "
        "No lists, tables, markdown or symbols.\n"
        "- Write units as words (millimetres, litres, kilowatts).\n"
        "- Mention the page number the answer comes from, for example \"According to page 11, ...\".\n"
        "- You may do simple arithmetic (add, subtract, multiply, divide, convert units) on "
        "numbers from the manual, and give the result.\n"
        f"- If the excerpts do not contain the answer, reply exactly: {NOT_FOUND_REPLY}\n\n"
        "Manual excerpts:\n{context}"
    ),
    "user_template": "{question}",
    "not_found": NOT_FOUND_REPLY,
}

# ─── Load Whisper + RAG once at startup ──────────────────────────────────────
print("🔄 Loading Whisper model...")
whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
print("✅ Whisper ready on GPU")

print("🔄 Loading RAG (Chroma + BM25)...")
_, _rag_col, _, _rag_corpus, _ = rag_query._resources()   # cached for the process lifetime
rag_query.retrieve_scored("warm-up")                       # loads nomic-embed-text
client.chat(model=TEXT_MODEL, keep_alive=-1, think=False,   # first answer was ~16 s cold, ~1 s warm
            messages=[{"role": "user", "content": "hi"}], options={"num_predict": 1})
print(f"✅ RAG ready: {len(_rag_corpus['ids'])} chunks from collection '{_rag_col.name}'")


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


# ─── Answering: RAG over the manual, or plain chat ───────────────────────────
def is_manual_question(scores):
    sim, bm25 = scores["top_similarity"] or 0.0, scores["top_bm25"]
    return sim >= RAG_MIN_SIMILARITY or (sim >= RAG_MIN_SIMILARITY_LEX and bm25 >= RAG_MIN_BM25)


def chat_reply(command):
    response = client.chat(
        model=TEXT_MODEL,
        keep_alive=-1,
        messages=[{"role": "user", "content": command}],
        think=False,  # thinking off for voice: latency matters
    )
    return response["message"]["content"]


def answer_question(question):
    """Route a question to the manual RAG or to plain qwen3 chat. Returns a JSON-able dict."""
    t0 = time.perf_counter()
    hits, scores = rag_query.retrieve_scored(question)
    if is_manual_question(scores):
        r = rag_query.ask_rag(question, prompt=NOVA_PROMPT, hits=hits, max_tokens=RAG_MAX_TOKENS)
        mentioned = [int(p) for p in re.findall(r"\bpages?\s+(\d+)", r["answer"], re.I)]
        pages = [] if r["not_found"] else ([p for p in mentioned if p in r["retrieved_pages"]]
                                           or r["retrieved_pages"])
        out = {"source": "rag", "answer": r["answer"], "not_found": r["not_found"],
               "pages": pages, "retrieved_pages": r["retrieved_pages"]}
    else:
        out = {"source": "chat", "answer": chat_reply(question), "not_found": False,
               "pages": [], "retrieved_pages": []}
    out.update(scores=scores, latency_s=round(time.perf_counter() - t0, 3))
    print(f"📚 {out['source']} {scores} pages={out['pages']} {out['latency_s']}s")
    return out


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

    print("🤖 Answering...")
    result = answer_question(command)
    print(f"🤖 Nova ({result['source']}): {result['answer']}")

    return jsonify({"status": "ok", "transcript": command, "reply": result["answer"],
                    "source": result["source"], "pages": result["pages"],
                    "latency_s": result["latency_s"]}), 200


# ─── Text Endpoint (testing without audio) ───────────────────────────────────
@app.route("/ask", methods=["POST"])
def handle_ask():
    text = ((request.get_json(silent=True) or {}).get("text") or "").strip()
    if not text:
        return jsonify({"status": "error", "error": 'expected JSON {"text": "..."}'}), 400
    print(f"💬 Text question: '{text}'")
    result = answer_question(text)
    print(f"🤖 Nova ({result['source']}): {result['answer']}")
    return jsonify({"status": "ok", "question": text, **result}), 200


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
