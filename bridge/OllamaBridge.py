from flask import Flask, request, jsonify
import base64
import io
import os
import queue
import re
import sys
import threading
import time
import urllib.request
import wave
from pathlib import Path
import numpy as np
import ollama
from faster_whisper import WhisperModel
from difflib import SequenceMatcher

# RAG pipeline lives in ~/nova/rag (Chroma collection + BM25 index over the milling manual)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))
import rag_query  # noqa: E402
import image_pipeline  # noqa: E402
from nova_prompt import NOVA_PROMPT, answer_pages, label_pages  # noqa: E402

app = Flask(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────
OLLAMA_HOST = "http://127.0.0.1:11434"   # Ollama now runs inside WSL, same as this script
# MODEL_MODE=single (default): qwen3-vl:8b-instruct answers text, voice and images, pinned
#   (keep_alive=-1), num_ctx 4096, no model swaps; images always use Mode B2.
# MODEL_MODE=dual (fallback): qwen3:8b for text + qwen2.5vl:7b for images, swapped per image.
MODEL_MODE = os.environ.get("MODEL_MODE", "single").lower()
assert MODEL_MODE in ("single", "dual"), "MODEL_MODE must be 'single' or 'dual'"
if MODEL_MODE == "single":
    image_pipeline.use_single_model()
TEXT_MODEL = image_pipeline.TEXT_MODEL       # single: qwen3-vl:8b-instruct | dual: qwen3:8b
VISION_MODEL = image_pipeline.VISION_MODEL   # single: qwen3-vl:8b-instruct | dual: qwen2.5vl:7b
MODEL_OPTIONS = image_pipeline.OPTIONS       # single: {"num_ctx": 4096} on every call
EMBED_MODEL = "nomic-embed-text"
WHISPER_MODEL = "large-v3-turbo"
# Whisper language for /live and ESP streams ("" = auto-detect). Short push-to-talk clips are
# often mis-detected, and the manual questions are English.
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en") or None
PORT = 5001
WAKE_WORD = "nova"
WAKE_THRESHOLD = 0.85

# ESP32 streaming on /audio: 256 ms packets of 16 kHz 16-bit mono PCM (8192 bytes), buffered per
# client IP. An utterance ends when no packet arrives for STREAM_SILENCE_S (or at STREAM_MAX_S).
STREAM_RATE = 16000
STREAM_SILENCE_S = float(os.environ.get("STREAM_SILENCE_S", 1.0))
STREAM_MAX_S = float(os.environ.get("STREAM_MAX_S", 30))
STREAM_PACKET_MAX_BYTES = 32000          # JSON raw-PCM payloads up to 1 s count as stream packets
# Camera for ESP "look" questions: nova_live.py on Windows serves its latest frame on port 5002.
# From WSL (NAT mode) Windows is the default gateway, not localhost. Override with LIVE_FRAME_URL.
LIVE_FRAME_PORT = 5002
FRAME_TIMEOUT_S = 1.5

client = ollama.Client(host=OLLAMA_HOST)

# Image questions: "A" = qwen2.5vl describes -> RAG -> qwen3 answers (model swap),
#                  "B2" = like B with the stricter spoken/page/honesty prompt + score gate,
#                  "B" = RAG -> qwen2.5vl answers from image + chunks. Per request: form "mode".
# Single mode always uses B2 (the request's "mode" is ignored).
IMAGE_MODE = "B2" if MODEL_MODE == "single" else os.environ.get("IMAGE_MODE", "A").upper()
assert IMAGE_MODE in image_pipeline.MODES, f"IMAGE_MODE must be one of {image_pipeline.MODES}"
# Dual mode: only one big model fits in 8 GB VRAM, so image swaps and text answers must not
# overlap. Single mode has no swaps; the lock just serialises GPU work.
GPU_LOCK = threading.Lock()

# ─── RAG routing ─────────────────────────────────────────────────────────────
# A question goes to the manual (RAG) when retrieval looks relevant, else to plain chat.
# Calibrated on the 60 benchmark + 30 rag_package questions vs. off-topic chat
# (manual: sim >= 0.62, or sim >= 0.58 with BM25 >= 5; off-topic max: sim 0.611 / BM25 4.6).
RAG_MIN_SIMILARITY = 0.62        # top cosine similarity alone is enough
RAG_MIN_SIMILARITY_LEX = 0.58    # ...or a lower similarity backed by keyword overlap
RAG_MIN_BM25 = 5.0
RAG_MAX_TOKENS = 256             # spoken answers are short
# Any of these words in the question forces the RAG path, even when the retrieval score is low
# (e.g. "What does error code E21 mean?"). Ambiguous words like "speed"/"power" are left out
# so general questions ("speed of light") still go to chat. Plurals are matched too.
MACHINE_KEYWORDS = {
    "error", "code", "alarm", "fault", "trouble", "warning", "spindle", "coolant", "feed",
    "motor", "table", "lubrication", "lubricant", "oil", "grease", "machine", "milling", "mill",
    "handwheel", "handle", "lever", "quill", "knee", "saddle", "column", "arbor", "cutter",
    "tool", "belt", "gear", "clutch", "brake", "bearing", "fuse", "breaker", "switch", "button",
    "panel", "display", "emergency", "workpiece", "vise", "clamp", "bolt", "nut", "screw",
    "rpm", "maintenance", "inspection", "manual", "warranty", "software", "cutting", "machining",
}
# Requests Nova can't do offline go to the chat route, which says so (instead of the manual's
# "I couldn't find that"). Actions always win, even with a machine word ("remind me to oil the
# machine"). Live-data words only win without a machine word, so "What is the price of this
# milling machine?" stays a manual question. "alarm" alone stays a machine word ("alarm lamp").
ACTION_RE = re.compile(
    r"\b(set|start|create)\b.{0,25}\b(timers?|alarms?|reminders?)\b|\btimers?\b"
    r"|\bremind (me|us)\b|\breminders?\b|\bwake me\b"
    r"|\b(send|write|reply to)\b.{0,30}\b(e-?mails?|messages?|texts?|sms)\b"
    r"|\be-?mail (to|my|him|her|them)\b|\bcall (my|me|him|her|them|someone)\b", re.I)
LIVE_DATA_RE = re.compile(r"\b(weather|forecast|news|headlines?|prices?)\b", re.I)
# /live and ESP: questions about what the camera sees go to the image route (Mode B2)
LOOK_RE = re.compile(
    r"\b(see|seeing|look|looking|camera|in front)\b|\bwhat(?:'s| is| are) (this|these|that)\b"
    r"|\bthis (machine|part|thing|lever|button|handle|switch|component|piece|one|plate|label|"
    r"sign|page|tool)\b", re.I)

# Plain-chat fallback: short and speakable (the reply is read aloud)
CHAT_SYSTEM = ("You are Nova, a voice assistant. Answer in one or two short sentences that sound "
               "natural when spoken. Plain text only: no markdown, lists, headings, emojis or symbols.\n"
               "Nova is fully offline. It has no internet, weather, news, prices, clock or timers, "
               "alarms, reminders, email, messages, phone calls, or device control. If asked for any "
               "of these, say briefly that you are offline and can't do that. Never invent live data "
               "(weather, news, prices, scores, the current time or date) and never claim to perform "
               "an action.")

# ─── Load Whisper + RAG once at startup ──────────────────────────────────────
print("🔄 Loading Whisper model...")
whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
WHISPER_LOCK = threading.Lock()   # one transcription at a time (/audio, /live, ESP stream worker)
_t = time.perf_counter()          # warm-up: the first transcription (+ VAD model) is ~5 s slower
_noise = (np.random.default_rng(0).standard_normal(16000) * 0.01).astype(np.float32)
list(whisper.transcribe(_noise, language="en")[0])                     # loads the decoder
list(whisper.transcribe(_noise, language="en", vad_filter=True)[0])     # loads the VAD model
print(f"✅ Whisper ready on GPU (warm-up {time.perf_counter() - _t:.1f}s)")

print("🔄 Loading RAG (Chroma + BM25)...")
_, _rag_col, _, _rag_corpus, _ = rag_query._resources()   # cached for the process lifetime
rag_query.retrieve_scored("warm-up")                       # loads nomic-embed-text
if MODEL_MODE == "single":   # free VRAM held by models from a previous dual-mode run
    for _m in image_pipeline.loaded_models() - {TEXT_MODEL, EMBED_MODEL, f"{EMBED_MODEL}:latest"}:
        client.generate(model=_m, prompt="", keep_alive=0)
        print(f"⏏️  Unloaded {_m}")
client.chat(model=TEXT_MODEL, keep_alive=-1, think=False,   # first answer was ~16 s cold, ~1 s warm
            messages=[{"role": "user", "content": "hi"}],
            options={**MODEL_OPTIONS, "num_predict": 1})
print(f"✅ RAG ready: {len(_rag_corpus['ids'])} chunks from collection '{_rag_col.name}'")
print(f"✅ MODEL_MODE={MODEL_MODE}: text {TEXT_MODEL}, images {VISION_MODEL} (mode {IMAGE_MODE}), "
      f"options {MODEL_OPTIONS or 'server default'}")


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
def keyword_hits(question):
    words = re.findall(r"[a-z]+", question.lower())
    return sorted({w for w in words if w in MACHINE_KEYWORDS or
                   (w.endswith("s") and w[:-1] in MACHINE_KEYWORDS)})


def route(scores, keywords, question=""):
    """Why the question goes where: "keyword"/"score" -> manual RAG;
    "action" (offline action / live data) or "low_score" -> plain chat."""
    if ACTION_RE.search(question) or (LIVE_DATA_RE.search(question) and not keywords):
        return "action"
    if keywords:
        return "keyword"
    sim, bm25 = scores["top_similarity"] or 0.0, scores["top_bm25"]
    if sim >= RAG_MIN_SIMILARITY or (sim >= RAG_MIN_SIMILARITY_LEX and bm25 >= RAG_MIN_BM25):
        return "score"
    return "low_score"


def strip_markdown(text):
    """Make a reply speakable: drop markdown emphasis, headings, bullets, code and links."""
    t = re.sub(r"```.*?```", " ", text, flags=re.S)                  # code blocks
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)                   # [text](url) -> text
    t = re.sub(r"(\*\*|__)(.+?)\1", r"\2", t)                        # bold
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", t)  # *italic* (keeps 2 * 3)
    t = re.sub(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)", r"\1", t)       # _italic_
    t = t.replace("`", "")
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)              # headings
    t = re.sub(r"^\s*(?:[-*+•]|\d+[.)])\s+", "", t, flags=re.M)      # list markers
    t = re.sub(r"^\s*([-*_=]\s*){3,}$", "", t, flags=re.M)           # horizontal rules
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    t = " ".join(l if re.search(r"[.!?:;]$", l) else l + "." for l in lines[:-1])
    t = (t + " " + lines[-1]).strip() if lines else ""
    return re.sub(r"\s{2,}", " ", t)


def chat_reply(command):
    response = client.chat(
        model=TEXT_MODEL,
        keep_alive=-1,
        messages=[{"role": "system", "content": CHAT_SYSTEM},
                  {"role": "user", "content": command}],
        think=False,  # thinking off for voice: latency matters
        options=dict(MODEL_OPTIONS) or None,
    )
    return response["message"]["content"]


def answer_question(question):
    with GPU_LOCK:   # wait if an image request is swapping models
        return _answer_question(question)


def _answer_question(question):
    """Route a question to the manual RAG or to plain qwen3 chat. Returns a JSON-able dict."""
    t0 = time.perf_counter()
    hits, scores = rag_query.retrieve_scored(question)
    keywords = keyword_hits(question)
    reason = route(scores, keywords, question)
    if reason in ("keyword", "score"):
        r = rag_query.ask_rag(question, prompt=NOVA_PROMPT, hits=label_pages(hits),
                              max_tokens=RAG_MAX_TOKENS, model=TEXT_MODEL, options=MODEL_OPTIONS)
        pages = answer_pages(r["answer"], r["retrieved_pages"], r["not_found"])
        out = {"source": "rag", "answer": r["answer"], "not_found": r["not_found"],
               "pages": pages, "retrieved_pages": r["retrieved_pages"]}
    else:
        out = {"source": "chat", "answer": chat_reply(question), "not_found": False,
               "pages": [], "retrieved_pages": []}
    out["answer"] = strip_markdown(out["answer"])
    out.update(route_reason=reason, keywords=keywords, scores=scores,
               latency_s=round(time.perf_counter() - t0, 3))
    print(f"📚 {out['source']} ({out['route_reason']}) {scores} kw={keywords} "
          f"pages={out['pages']} {out['latency_s']}s")
    return out


# ─── Speech in, answer out (shared by /live and the ESP stream) ──────────────
def transcribe(audio):
    """audio: file path, file-like (any format av can read) or float32 16 kHz numpy array.
    VAD drops silence so push-to-talk gaps don't turn into hallucinated words."""
    t0 = time.perf_counter()
    with WHISPER_LOCK:
        segments, info = whisper.transcribe(audio, language=WHISPER_LANGUAGE, vad_filter=True)
        text = " ".join(s.text for s in segments).strip()
    return text, info.language, round(time.perf_counter() - t0, 3)


def windows_host_ip():
    """Windows as seen from WSL (NAT mode): the default gateway in /proc/net/route."""
    for line in open("/proc/net/route").read().splitlines()[1:]:
        f = line.split()
        if f[1] == "00000000":
            return ".".join(str(int(f[2][i:i + 2], 16)) for i in (6, 4, 2, 0))
    return "127.0.0.1"


LIVE_FRAME_URL = os.environ.get("LIVE_FRAME_URL") or \
    f"http://{windows_host_ip()}:{LIVE_FRAME_PORT}/frame"


def fetch_frame():
    """Latest camera frame from nova_live.py. Returns (jpeg bytes or None, note)."""
    try:
        with urllib.request.urlopen(LIVE_FRAME_URL, timeout=FRAME_TIMEOUT_S) as r:
            data = r.read()
        return (data, "ok") if data else (None, "empty frame")
    except Exception as e:
        return None, f"no frame from {LIVE_FRAME_URL}: {e}"


# Last answer for clients that poll (nova_live.py shows + speaks answers to ESP questions too)
_last = {"id": 0, "answer": None, "heard": None}
_last_lock = threading.Lock()


def publish(result):
    with _last_lock:
        _last["id"] += 1
        result["id"] = _last["id"]
        _last["answer"] = result
    return result


def respond(transcript, origin, image_bytes=None, frame_from_camera=False):
    """Route a spoken question. "Look" questions + a frame -> image route (Mode B2 in single
    mode); everything else -> the voice route (manual RAG / chat / offline refusal)."""
    t0 = time.perf_counter()
    look = bool(LOOK_RE.search(transcript))
    frame_note = None
    if look and image_bytes is None and frame_from_camera:
        image_bytes, frame_note = fetch_frame()
    out = {"origin": origin, "transcript": transcript, "look_question": look,
           "frame": frame_note or ("sent" if image_bytes else "none")}
    if look and image_bytes:
        try:
            image_b64, img_info = image_pipeline.preprocess(image_bytes)
        except image_pipeline.BadImage as e:
            image_b64, out["frame"] = None, f"bad image: {e}"
    else:
        image_b64 = None
    if image_b64:
        with GPU_LOCK:
            if IMAGE_MODE == "B2":
                r = image_pipeline.run_mode_b2_look(image_b64, transcript, RAG_MAX_TOKENS)
            else:
                r = image_pipeline.run(image_b64, transcript, IMAGE_MODE, RAG_MAX_TOKENS)
        t = r["timings"]
        fallback = " + describe-retrieval" if r.get("look_fallback") else ""
        out.update(route="image", route_reason=f"look question, mode {IMAGE_MODE}{fallback}",
                   answer=strip_markdown(r["answer"]), pages=r["pages"],
                   not_found=r["not_found"], image=img_info,
                   timings={"answer_s": round(time.perf_counter() - t0, 3),
                            "vision_s": t["vision_s"], "rag_s": t["rag_s"],
                            "swap_s": t["swap_s"]})
    else:
        r = answer_question(transcript)
        out.update(route=r["source"], route_reason=r["route_reason"], answer=r["answer"],
                   pages=r["pages"], not_found=r["not_found"],
                   timings={"answer_s": round(time.perf_counter() - t0, 3)})
    print(f"🗣️  [{origin}] {out['route']} ({out['route_reason']}, frame {out['frame']}): "
          f"{out['answer']}")
    return out


# ─── ESP32 stream: per-IP packet buffers, closed after STREAM_SILENCE_S ──────
_streams = {}                    # ip -> {"pcm": bytearray, "first": t, "last": t, "packets": n}
_streams_lock = threading.Lock()
_utterances = queue.Queue()      # closed utterances, processed one at a time


def add_packet(ip, pcm):
    now = time.monotonic()
    with _streams_lock:
        s = _streams.setdefault(ip, {"pcm": bytearray(), "first": now, "last": now, "packets": 0})
        s["pcm"] += pcm
        s["last"] = now
        s["packets"] += 1
        buffered_s = len(s["pcm"]) / (2 * STREAM_RATE)
        if buffered_s >= STREAM_MAX_S:             # never-ending stream: cut it here
            _utterances.put((ip, _streams.pop(ip), "max length"))
    return buffered_s


def _stream_watchdog():
    while True:
        time.sleep(0.1)
        now = time.monotonic()
        with _streams_lock:
            for ip in [ip for ip, s in _streams.items() if now - s["last"] >= STREAM_SILENCE_S]:
                _utterances.put((ip, _streams.pop(ip), "silence"))


def _utterance_worker():
    while True:
        ip, s, why = _utterances.get()
        try:
            process_utterance(ip, s, why)
        except Exception as e:      # keep the worker alive whatever happens
            print(f"❌ ESP utterance from {ip} failed: {e!r}")


def process_utterance(ip, s, why):
    t0 = time.perf_counter()
    pcm = bytes(s["pcm"][:len(s["pcm"]) // 2 * 2])
    dur = len(pcm) / (2 * STREAM_RATE)
    with wave.open(str(Path(__file__).with_name("last_utterance.wav")), "wb") as wf:
        wf.setnchannels(1), wf.setsampwidth(2), wf.setframerate(STREAM_RATE), wf.writeframes(pcm)
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    text, lang, whisper_s = transcribe(audio)
    heard = {"ip": ip, "transcript": text, "language": lang, "audio_s": round(dur, 2),
             "packets": s["packets"], "closed_by": why, "whisper_s": whisper_s,
             "time": time.strftime("%H:%M:%S")}
    print(f"🎤 ESP {ip}: {s['packets']} packets, {dur:.2f}s ({why}) -> '{text}' {whisper_s}s")
    wake = wake_word_detected(text.lower())
    with _last_lock:
        _last["heard"] = {**heard, "wake_word": wake}
    if not wake:
        print("💤 Wake word not detected, ignoring.")
        return
    command = extract_command(text)
    out = respond(command, origin="esp", frame_from_camera=True)
    out["timings"] = {"whisper_s": whisper_s, **out["timings"],
                      "total_s": round(time.perf_counter() - t0, 3)}
    out.update(heard=heard, pages=out["pages"])
    publish(out)


threading.Thread(target=_stream_watchdog, daemon=True, name="stream-watchdog").start()
threading.Thread(target=_utterance_worker, daemon=True, name="utterance-worker").start()


# ─── Audio Endpoint ──────────────────────────────────────────────────────────
# ESP stream (new): each POST is one packet of raw 16 kHz 16-bit mono PCM, either as the raw body
#   (Content-Type: application/octet-stream) or as JSON {"state": 101, "payload": <base64>} of at
#   most 1 s. Returns at once; the utterance is answered after STREAM_SILENCE_S without packets
#   and the answer is published on /last_answer.
# Legacy: JSON {"state": 101, "payload": <base64 WAV / M4A / longer raw PCM>} = one whole recording.
@app.route("/audio", methods=["POST"])
def handle_audio():
    ip = request.remote_addr
    if not request.is_json:
        pcm = request.get_data()
        if not pcm:
            return jsonify({"status": "error", "error": "empty body"}), 400
        return jsonify({"status": "buffering", "buffered_s": round(add_packet(ip, pcm), 3),
                        "last_answer_id": _last["id"]}), 200

    data = request.get_json(silent=True) or {}
    if data.get("state") != 101 or not data.get("payload"):
        return jsonify({"status": "error", "error": "expected state 101 + payload"}), 400
    audio_bytes = base64.b64decode(data["payload"])

    # Detect format by magic bytes
    is_wav = audio_bytes[:4] == b"RIFF"
    is_m4a = audio_bytes[4:8] == b"ftyp" if len(audio_bytes) > 8 else False
    if not (is_wav or is_m4a) and len(audio_bytes) <= STREAM_PACKET_MAX_BYTES:
        return jsonify({"status": "buffering", "buffered_s": round(add_packet(ip, audio_bytes), 3),
                        "last_answer_id": _last["id"]}), 200

    print(f"✅ Received {len(audio_bytes)} bytes of audio")
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

    # Transcribe (auto language detection, as before)
    with WHISPER_LOCK:
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


# ─── Live endpoint (push-to-talk from nova_live.py) ─────────────────────────
# multipart: audio=<wav, 16 kHz mono>, image=<jpg, optional>. No wake word.
@app.route("/live", methods=["POST"])
def handle_live():
    t0 = time.perf_counter()
    f = request.files.get("audio")
    if f is None:
        return jsonify({"status": "error", "error": "multipart field 'audio' (wav) is required"}), 400
    audio_bytes = f.read()
    image_bytes = request.files["image"].read() if request.files.get("image") else None
    try:
        text, lang, whisper_s = transcribe(io.BytesIO(audio_bytes))
    except Exception as e:
        return jsonify({"status": "error", "error": f"cannot decode audio: {e}"}), 400
    print(f"🎙️  /live ({lang}, {whisper_s}s): '{text}'  image={'yes' if image_bytes else 'no'}")
    if not text:
        return jsonify({"status": "empty", "transcript": "", "route": None, "answer": "",
                        "pages": [], "timings": {"whisper_s": whisper_s}}), 200
    out = respond(text, origin="live", image_bytes=image_bytes)
    out["timings"] = {"whisper_s": whisper_s, **out["timings"],
                      "total_s": round(time.perf_counter() - t0, 3)}
    publish(out)
    return jsonify({"status": "ok", **out}), 200


# ─── Last answer (polled by nova_live.py; ESP answers show up here) ─────────
@app.route("/last_answer", methods=["GET"])
def handle_last_answer():
    with _last_lock:
        return jsonify({"id": _last["id"], "answer": _last["answer"], "heard": _last["heard"]}), 200

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
# New:    multipart form  image=<jpg/png file>, question=<text, optional>, mode=A|B (optional)
# Legacy: JSON {"state": 102, "payload": <base64 image>, "command": <text>, "mode": optional}
#         (esp_simulator.py and the camera script)
@app.route("/image", methods=["POST"])
def handle_image():
    if request.files.get("image"):
        image_bytes = request.files["image"].read()
        question, mode = request.form.get("question"), request.form.get("mode")
        legacy = False
    else:
        data = request.get_json(silent=True) or {}
        if data.get("state") != 102 or not data.get("payload"):
            return jsonify({"status": "error", "error": "send multipart 'image' (+ 'question'), "
                            "or JSON {state: 102, payload: <base64>, command}"}), 400
        try:
            image_bytes = base64.b64decode(data["payload"], validate=True)
        except ValueError:
            return jsonify({"status": "error", "error": "payload is not valid base64"}), 400
        question, mode = data.get("command"), data.get("mode")
        legacy = True

    question = (question or "").strip() or image_pipeline.DEFAULT_QUESTION
    if MODEL_MODE == "single":       # one model: always B2, no swaps
        if mode and mode.strip().upper() != IMAGE_MODE:
            print(f"ℹ️  mode {mode!r} requested; single-model mode uses {IMAGE_MODE}")
        mode = IMAGE_MODE
    mode = (mode or IMAGE_MODE).strip().upper()
    if mode not in image_pipeline.MODES:
        return jsonify({"status": "error", "error": f"mode must be one of {image_pipeline.MODES}, got {mode!r}"}), 400
    try:
        image_b64, img_info = image_pipeline.preprocess(image_bytes)
    except image_pipeline.BadImage as e:
        return jsonify({"status": "error", "error": str(e)}), 400

    print(f"📷 Image {img_info['orig_size']} -> {img_info['proc_size']} ({img_info['proc_kb']} KB), "
          f"mode {mode}{' (legacy JSON)' if legacy else ''}: '{question}'")
    with GPU_LOCK:
        r = image_pipeline.run(image_b64, question, mode, RAG_MAX_TOKENS)
    answer = strip_markdown(r["answer"])
    t = r["timings"]
    if mode == "A":
        print(f"👁️  Description: {r['description']}")
    print(f"🤖 Nova image ({mode}): {answer}")
    print(f"⏱️  vision {t['vision_s']}s, rag {t['rag_s']}s, answer {t['answer_s']}s, "
          f"swap {t['swap_s']}s, total {t['total_s']}s, pages={r['pages']}")

    out = {"status": "ok", "answer": answer, "pages": r["pages"], "mode": mode,
           "question": question, "not_found": r["not_found"],
           "retrieved_pages": r["retrieved_pages"], "image": img_info,
           "timings": {k: t[k] for k in ("vision_s", "rag_s", "answer_s", "swap_s", "total_s")}}
    if mode == "A":
        out["description"] = r["description"]
    if legacy:
        out["reply"] = answer          # old clients read "reply"
    return jsonify(out), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
