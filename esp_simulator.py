import requests
import base64
import sys

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

# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "audio"

    if mode == "audio":
        wav = sys.argv[2] if len(sys.argv) > 2 else AUDIO_FILE
        test_audio(wav)
    elif mode == "image":
        img = sys.argv[2] if len(sys.argv) > 2 else "test_frame.jpg"
        cmd = sys.argv[3] if len(sys.argv) > 3 else "What do you see in this image?"
        test_image(img, cmd)
    else:
        print("Usage:")
        print("  python esp_simulator.py audio [path/to/file.wav or .m4a]")
        print("  python esp_simulator.py image [path/to/frame.jpg] ['your command']")
