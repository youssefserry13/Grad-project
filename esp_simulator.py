import requests
import base64
import sys
import threading
import time
import wave

import numpy as np

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


# ─── Test 4: continuous stream (firmware noise gate 0: packets never stop) ──
DEFAULT_PARTS = ["live_stream.wav", "live/test_audio/esp_nova_tank.wav@-20"]


def read_part(spec):
    """'file.wav' or 'file.wav@-20' (gain in dB). 16 kHz mono 16-bit -> float32 samples."""
    path, _, gain = spec.partition("@")
    with wave.open(path, "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), path
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
    return path, x * 10 ** (float(gain or 0) / 20)


def test_continuous(parts, noise_dbfs=-74.0, gap_s=2.0, tail_s=4.0, fast=False, wait_s=40):
    """One gap-free stream: noise, part 1, noise, part 2, ..., noise tail. Background noise at
    noise_dbfs is added everywhere (the real ESP floor is ~ -74 dBFS). Prints every utterance
    the bridge closes (/last_answer 'heard') and every answer, with timings."""
    rng = np.random.default_rng(0)
    noise = lambda sec: rng.standard_normal(int(sec * 16000)) * 32768 * 10 ** (noise_dbfs / 20)
    pieces, marks, t = [noise(1.0)], [], 1.0
    for spec in parts:
        path, x = read_part(spec)
        pieces += [x + noise(len(x) / 16000), noise(gap_s)]
        marks.append((path.split("/")[-1], t, t + len(x) / 16000))
        t += len(x) / 16000 + gap_s
    pieces.append(noise(tail_s))
    pcm = np.clip(np.concatenate(pieces), -32768, 32767).astype(np.int16).tobytes()
    size = 16000 * 2 * PACKET_MS // 1000
    packets = [pcm[i:i + size] for i in range(0, len(pcm), size)]
    for name, a, b in marks:
        print(f"   {name}: {a:5.2f}s - {b:5.2f}s of the stream")
    print(f"📤 Continuous stream: {len(packets)} packets, {len(pcm) / 32000:.1f}s, noise "
          f"{noise_dbfs} dBFS, no gaps{' (fast)' if fast else ' (real time)'}")

    s = requests.Session()
    prev = s.get(f"{BRIDGE_URL}/last_answer").json()
    seen = {"id": prev["id"], "heard": prev.get("heard")}
    t0 = time.perf_counter()
    done = threading.Event()

    def poll():
        while not done.is_set():
            d = requests.get(f"{BRIDGE_URL}/last_answer", timeout=5).json()
            now = time.perf_counter() - t0
            h = d.get("heard")
            if h and h != seen["heard"]:
                seen["heard"] = h
                print(f"  [{now:5.1f}s] 🎤 utterance {h['audio_s']}s ({h['closed_by']}, voiced "
                      f"{h['voiced_s']}s) wake={h['wake_word']}: {h['transcript']!r}")
            if d["id"] > seen["id"]:
                seen["id"] = d["id"]
                a = d["answer"]
                print(f"  [{now:5.1f}s] 📥 {a['route']} ({a['route_reason']}): {a['answer']!r} "
                      f"pages {a['pages']} timings {a['timings']}")
            time.sleep(0.2)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    for i, p in enumerate(packets):
        s.post(f"{BRIDGE_URL}/audio", data=p,
               headers={"Content-Type": "application/octet-stream"}).raise_for_status()
        if not fast:
            time.sleep(max(0.0, t0 + (i + 1) * PACKET_MS / 1000 - time.perf_counter()))
    print(f"  [{time.perf_counter() - t0:5.1f}s] last packet sent (stream stops here)")
    time.sleep(wait_s if fast else min(wait_s, 15))
    done.set()
    th.join()


# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "audio"

    if mode == "audio":
        wav = sys.argv[2] if len(sys.argv) > 2 else AUDIO_FILE
        test_audio(wav)
    elif mode in ("stream", "stream-json"):
        wav = sys.argv[2] if len(sys.argv) > 2 else AUDIO_FILE
        test_stream(wav, as_json=mode == "stream-json", fast="--fast" in sys.argv)
    elif mode == "continuous":
        args = [a for a in sys.argv[2:] if not a.startswith("--")]
        opt = lambda name, d: float(next((a.split("=")[1] for a in sys.argv
                                          if a.startswith(f"--{name}=")), d))
        test_continuous(args or DEFAULT_PARTS, noise_dbfs=opt("noise-dbfs", -74),
                        gap_s=opt("gap-s", 2.0), tail_s=opt("tail-s", 4.0),
                        fast="--fast" in sys.argv)
    elif mode == "image":
        img = sys.argv[2] if len(sys.argv) > 2 else "test_frame.jpg"
        cmd = sys.argv[3] if len(sys.argv) > 3 else "What do you see in this image?"
        test_image(img, cmd)
    else:
        print("Usage:")
        print("  python esp_simulator.py audio [path/to/file.wav or .m4a]")
        print("  python esp_simulator.py stream [file.wav] [--fast]   # 256 ms raw PCM packets")
        print("  python esp_simulator.py stream-json [file.wav]       # same, JSON base64 packets")
        print("  python esp_simulator.py continuous [a.wav b.wav@-20 ...] [--noise-dbfs=-74]")
        print("        [--gap-s=2] [--tail-s=4] [--fast]   # gap-free stream with background noise")
        print("  python esp_simulator.py image [path/to/frame.jpg] ['your command']")
