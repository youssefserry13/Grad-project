"""Nova live demo client (Windows Python, not WSL).

    py nova_live.py                       # webcam 0, bridge http://localhost:5001
    py nova_live.py --camera 1 --mic 3    # other camera / microphone (see --list-mics)
    py nova_live.py --list-mics

Hold SPACE (with this window focused) to talk, release to send the recording + the current
camera frame to the bridge's /live endpoint. The transcript and answer are drawn on the video
and every answer is spoken, in order, with the offline Windows voice (SAPI). Q or Esc quits.
--require-wake: only answer when you say "Nova ..." (default: push-to-talk needs no wake word).

Also, for the ESP32 microphone:
  - serves the latest camera frame on http://<this laptop>:5002/frame, so the bridge can answer
    "look" questions asked through the ESP;
  - polls the bridge's /last_answer and shows + speaks answers to ESP questions as well.

pip install opencv-python sounddevice numpy requests pywin32
"""
import argparse
import ctypes
import io
import queue
import textwrap
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import requests
import sounddevice as sd

WINDOW = "Nova Live"
RATE = 16000
MIN_CLIP_S = 0.3
VK_SPACE = 0x20
user32 = ctypes.windll.user32


# ─── Shared state ────────────────────────────────────────────────────────────
class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_jpg = None          # latest camera frame (JPEG bytes) for /live and :5002
        self.status = "Hold SPACE to talk"
        self.you = ""
        self.nova = ""
        self.meta = ""
        self.busy = False
        self.shown_id = 0              # last /last_answer id already shown


S = State()


# ─── Speech output: SAPI directly, in its own thread, every answer in order ──
# (pyttsx3's runAndWait only spoke the first answer on Windows.) Speak() blocks only this
# thread, so the camera window keeps running while Nova talks.
speech_q = queue.Queue()
TTS_RATE = 1                           # SAPI rate -10..10 (0 = normal)


def speaker():
    import pythoncom
    import win32com.client
    pythoncom.CoInitialize()           # COM must be initialised in the thread that uses it
    voice = win32com.client.Dispatch("SAPI.SpVoice")
    voice.Rate = TTS_RATE
    while True:
        text = speech_q.get()
        try:
            voice.Speak(text)          # synchronous (flags 0): finish this answer, then the next
        except Exception as e:
            print(f"TTS error: {e}")
            try:                       # recreate the voice if SAPI got into a bad state
                voice = win32com.client.Dispatch("SAPI.SpVoice")
                voice.Rate = TTS_RATE
            except Exception:
                pass


def say(text):
    if text:
        speech_q.put(text)


# ─── Microphone (push-to-talk) ───────────────────────────────────────────────
class Recorder:
    def __init__(self, device):
        self.device, self.chunks, self.stream = device, [], None

    def start(self):
        self.chunks = []
        self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype="int16",
                                     device=self.device,
                                     callback=lambda data, *_: self.chunks.append(data.copy()))
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()
        pcm = np.concatenate(self.chunks) if self.chunks else np.zeros((0, 1), np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(pcm.tobytes())
        return buf.getvalue(), len(pcm) / RATE


def space_held():
    """SPACE is down and our window has focus (no key-up events in OpenCV, so ask Windows)."""
    hwnd = user32.FindWindowW(None, WINDOW)
    return bool(hwnd) and user32.GetForegroundWindow() == hwnd and \
        bool(user32.GetAsyncKeyState(VK_SPACE) & 0x8000)


# ─── Talking to the bridge ───────────────────────────────────────────────────
def show_answer(a, prefix=""):
    pages = ", ".join(map(str, a.get("pages") or []))
    with S.lock:
        S.you = prefix + (a.get("transcript") or "")
        S.nova = a.get("answer") or ""
        S.meta = f"route: {a.get('route')}" + (f" | pages {pages}" if pages else "") + \
                 f" | {a.get('timings', {}).get('total_s', '?')} s"
        S.shown_id = max(S.shown_id, a.get("id") or 0)
    print(f"\n{prefix}You : {a.get('transcript')}\nNova: {a.get('answer')}\n"
          f"route={a.get('route')} ({a.get('route_reason')}) frame={a.get('frame')} "
          f"pages={a.get('pages')}\ntimings={a.get('timings')}")
    say(a.get("answer"))


def post_status(bridge, state):
    """Tell the bridge's /display page that SPACE is held (listening) or released."""
    def go():
        try:
            requests.post(f"{bridge}/status", json={"state": state}, timeout=2)
        except requests.RequestException:
            pass
    threading.Thread(target=go, daemon=True).start()


def send_live(bridge, wav, jpg, require_wake=False):
    t0 = time.perf_counter()
    try:
        files = {"audio": ("question.wav", wav, "audio/wav")}
        if jpg:
            files["image"] = ("frame.jpg", jpg, "image/jpeg")
        r = requests.post(f"{bridge}/live", files=files, timeout=180,
                          data={"require_wake": "1" if require_wake else "0"})
        a = r.json()
        if a.get("status") == "empty":
            with S.lock:
                S.status = "Didn't catch that - hold SPACE and try again"
        elif a.get("status") == "ignored":
            with S.lock:
                S.status = f"No wake word - start with \"Nova\" (heard: {a.get('transcript', '')[:40]})"
            print(f"ignored (no wake word): {a.get('transcript')}")
        elif a.get("status") != "ok":
            with S.lock:
                S.status = f"Bridge error: {a.get('error')}"
        else:
            show_answer(a)
            with S.lock:
                S.status = "Hold SPACE to talk"
        print(f"round trip {time.perf_counter() - t0:.2f}s")
    except requests.RequestException as e:
        with S.lock:
            S.status = f"Bridge not reachable: {e.__class__.__name__}"
        print(f"bridge error: {e}")
    finally:
        with S.lock:
            S.busy = False
            if S.status.startswith("Thinking"):
                S.status = "Hold SPACE to talk"


def poll_bridge(bridge):
    """Answers to ESP questions (and what the ESP heard without the wake word)."""
    last_heard = None
    while True:
        time.sleep(0.5)
        try:
            d = requests.get(f"{bridge}/last_answer", timeout=2).json()
        except requests.RequestException:
            continue
        a = d.get("answer")
        if a and d["id"] > S.shown_id and a.get("origin") != "live":
            show_answer(a, prefix="[ESP] ")
        elif a and d["id"] > S.shown_id:
            with S.lock:
                S.shown_id = d["id"]
        h = d.get("heard")
        if h and h != last_heard:
            if last_heard is not None and not h.get("wake_word"):
                with S.lock:
                    S.status = f"ESP heard (no wake word): {h['transcript'][:60]}"
            last_heard = h


# ─── Frame server for the bridge (ESP "look" questions) ──────────────────────
class FrameHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        with S.lock:
            jpg = S.frame_jpg
        if self.path.startswith("/frame") and jpg:
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.end_headers()
            self.wfile.write(jpg)
        else:
            self.send_response(404 if self.path.startswith("/frame") else 200)
            self.end_headers()
            self.wfile.write(b"Nova live frame server: GET /frame")

    def log_message(self, *args):
        pass


# ─── Drawing ─────────────────────────────────────────────────────────────────
def draw_text_box(img, lines, top, color=(255, 255, 255)):
    h = 26 * len(lines) + 12
    overlay = img.copy()
    cv2.rectangle(overlay, (0, top), (img.shape[1], top + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    for i, (text, c) in enumerate(lines):
        cv2.putText(img, text, (12, top + 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    c or color, 1, cv2.LINE_AA)


def draw(frame, recording):
    img = frame.copy()
    with S.lock:
        status, you, nova, meta = S.status, S.you, S.nova, S.meta
    width = max(30, img.shape[1] // 12)
    if recording:
        cv2.circle(img, (img.shape[1] - 30, 30), 12, (0, 0, 255), -1)
        status = "Listening... release SPACE to send"
    draw_text_box(img, [(status, (0, 255, 255))], 0)
    lines = []
    for t in textwrap.wrap(f"You: {you}", width)[:2]:
        lines.append((t, (200, 200, 200)))
    for t in textwrap.wrap(f"Nova: {nova}", width)[:5]:
        lines.append((t, (255, 255, 255)))
    if meta:
        lines.append((meta, (150, 220, 150)))
    if you or nova:
        draw_text_box(img, lines, img.shape[0] - (26 * len(lines) + 12))
    return img


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", default="http://localhost:5001")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--mic", type=int, default=None, help="input device index (--list-mics)")
    ap.add_argument("--frame-port", type=int, default=5002)
    ap.add_argument("--no-image", action="store_true", help="never send the camera frame")
    ap.add_argument("--require-wake", action="store_true",
                    help='only answer when the question starts with "Nova" (default off)')
    ap.add_argument("--list-mics", action="store_true")
    a = ap.parse_args()
    if a.list_mics:
        print(sd.query_devices())
        return

    try:
        r = requests.get(f"{a.bridge}/last_answer", timeout=3).json()
        S.shown_id = r["id"]
        print(f"Bridge OK at {a.bridge} (last answer id {r['id']})")
    except Exception as e:
        print(f"WARNING: bridge not reachable at {a.bridge}: {e}")
        S.status = "Bridge not reachable - start it in WSL"

    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {a.camera}")

    threading.Thread(target=speaker, daemon=True).start()
    threading.Thread(target=poll_bridge, args=(a.bridge,), daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", a.frame_port), FrameHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Frame server on port {a.frame_port}. Hold SPACE to talk, Q to quit."
          + (' Wake word required: start with "Nova".' if a.require_wake else ""))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    rec, recording, last_jpg_t = Recorder(a.mic), False, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        now = time.monotonic()
        if now - last_jpg_t > 0.2:                       # keep a fresh JPEG (~5 fps)
            jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
            with S.lock:
                S.frame_jpg = jpg
            last_jpg_t = now

        held = space_held()
        if held and not recording and not S.busy:
            rec.start()
            recording = True
            post_status(a.bridge, "listening")
        elif recording and not held:
            recording = False
            wav, dur = rec.stop()
            if dur < MIN_CLIP_S:
                S.status = "Too short - hold SPACE while you speak"
                post_status(a.bridge, "ready")
            else:
                with S.lock:
                    S.busy, S.status = True, f"Thinking... ({dur:.1f} s of audio)"
                    jpg = None if a.no_image else S.frame_jpg
                threading.Thread(target=send_live, args=(a.bridge, wav, jpg, a.require_wake),
                                 daemon=True).start()

        cv2.imshow(WINDOW, draw(frame, recording))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()
    server.shutdown()


if __name__ == "__main__":
    main()
