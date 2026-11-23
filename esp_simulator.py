import requests
import base64
import sys
import time
import wave

# ─── Config ──────────────────────────────────────────────────────────────────
BRIDGE_URL = "http://127.0.0.1:5001"  # Change to Ubuntu IP if running from Windows
AUDIO_FILE = "received_audio.wav"

# ─── Test 1: Audio ───────────────────────────────────────────────────────────
def test_audio(wav_path=AUDIO_FILE):
    print(f"📤 Sending audio file: {wav_path}")
    with open(wav_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "state": 101,
        "payload": audio_b64
    }

    response = requests.post(f"{BRIDGE_URL}/audio", json=payload)
    print(f"📥 Server response ({response.status_code}):")
    print(response.json())

# ─── Test 2: Image ───────────────────────────────────────────────────────────
def test_image(image_path="test_frame.jpg", command="What do you see in this image?"):
    print(f"📤 Sending image file: {image_path}")
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "state": 102,
        "payload": image_b64,
        "command": command
    }

    response = requests.post(f"{BRIDGE_URL}/image", json=payload)
    print(f"📥 Server response ({response.status_code}):")
    print(response.json())

# ─── Test 3: ESP stream (256 ms PCM packets) ─────────────────────────────────
PACKET_MS = 256


def test_stream(wav_path=AUDIO_FILE, as_json=False, fast=False, wait_s=60):
    """Send a 16 kHz 16-bit mono WAV like the ESP32: 256 ms packets (8192 bytes) of raw PCM,
    one HTTP POST each over a reused connection, in real time (fast=True: no pacing). Then wait
    for the bridge to close the utterance (silence timeout) and print /last_answer."""
    with wave.open(wav_path, "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), \
            "need 16 kHz, mono, 16-bit WAV"
        pcm = w.readframes(w.getnframes())
    size = 16000 * 2 * PACKET_MS // 1000
    packets = [pcm[i:i + size] for i in range(0, len(pcm), size)]
    s = requests.Session()                                   # keep-alive: socket reused
    prev = s.get(f"{BRIDGE_URL}/last_answer").json()
    before, before_heard = prev["id"], prev.get("heard")
    print(f"📤 Streaming {wav_path}: {len(packets)} packets of {PACKET_MS} ms "
          f"({'JSON base64' if as_json else 'raw octet-stream'})")
    t0 = time.perf_counter()
    for i, p in enumerate(packets):
        if as_json:
            r = s.post(f"{BRIDGE_URL}/audio",
                       json={"state": 101, "payload": base64.b64encode(p).decode()})
        else:
            r = s.post(f"{BRIDGE_URL}/audio", data=p,
                       headers={"Content-Type": "application/octet-stream"})
        r.raise_for_status()
        if not fast:                                         # real-time pacing
            time.sleep(max(0.0, t0 + (i + 1) * PACKET_MS / 1000 - time.perf_counter()))
    sent = time.perf_counter()
    print(f"✅ Sent in {sent - t0:.2f}s, last packet reply: {r.json()}")
    while time.perf_counter() - sent < wait_s:
        time.sleep(0.25)
        last = s.get(f"{BRIDGE_URL}/last_answer").json()
        heard = last.get("heard") or {}
        if last["id"] > before:
            a = last["answer"]
            print(f"📥 after {time.perf_counter() - sent:.2f}s from last packet:")
            print(f"   transcript: {a['transcript']!r}\n   route: {a['route']} "
                  f"({a['route_reason']}, frame {a['frame']})\n   answer: {a['answer']!r}\n"
                  f"   pages: {a['pages']}\n   timings: {a['timings']}")
            return last
        if heard and heard != before_heard and not heard.get("wake_word"):
            print(f"💤 heard without wake word after {time.perf_counter() - sent:.2f}s: "
                  f"{heard['transcript']!r} (whisper {heard['whisper_s']}s)")
            return last
    print("⏱️  no answer within", wait_s, "s")


# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "audio"

    if mode == "audio":
        wav = sys.argv[2] if len(sys.argv) > 2 else AUDIO_FILE
        test_audio(wav)
    elif mode in ("stream", "stream-json"):
        wav = sys.argv[2] if len(sys.argv) > 2 else AUDIO_FILE
        test_stream(wav, as_json=mode == "stream-json", fast="--fast" in sys.argv)
    elif mode == "image":
        img = sys.argv[2] if len(sys.argv) > 2 else "test_frame.jpg"
        cmd = sys.argv[3] if len(sys.argv) > 3 else "What do you see in this image?"
        test_image(img, cmd)
    else:
        print("Usage:")
        print("  python esp_simulator.py audio [path/to/file.wav or .m4a]")
        print("  python esp_simulator.py stream [file.wav] [--fast]   # 256 ms raw PCM packets")
        print("  python esp_simulator.py stream-json [file.wav]       # same, JSON base64 packets")
        print("  python esp_simulator.py image [path/to/frame.jpg] ['your command']")
