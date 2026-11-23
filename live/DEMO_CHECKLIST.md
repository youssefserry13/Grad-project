# Nova lab demo: checklist

**Links:** ESP32 mic ──Wi-Fi──▶ laptop `<Wi-Fi IP>:5001` ──portproxy──▶ WSL bridge `:5001`
· webcam client `nova_live.py` (Windows) ──localhost:5001──▶ bridge · bridge ──▶ `<WSL gateway>:5002/frame` (camera frame)

## The night before (home)
- [ ] Windows: `winget install -e --id Python.Python.3.12`, then a **new** terminal: `py -m pip install opencv-python sounddevice numpy requests pywin32`
- [ ] Run the client once (below). When Windows Firewall asks about Python, click **Allow** (tick both networks). If you cancel it, Windows adds a block rule that wins over the setup script's allow rule for 5002.
- [ ] Laptop on charger, power mode "Best performance", Windows updates paused, notifications off.

## Start order (lab)
1. **WSL** `nvidia-smi` (GPU visible) → `cd ~/nova/bridge && ./start_bridge.sh` → wait for `Running on http://127.0.0.1:5001` (~40 s).
   `MODEL_MODE=single` line must say qwen3-vl, ctx 4096.
2. **Join the lab Wi-Fi**, then **admin PowerShell**: `powershell -ExecutionPolicy Bypass -File C:\Grad\live\esp_network_setup.ps1` → note the green `ESP firmware: http://<Wi-Fi IP>:5001/audio`.
3. **ESP32**: set that URL (if the firmware has the IP hard-coded, it must match; see fallback B).
4. **Windows terminal**: `py C:\Grad\live\nova_live.py` (other camera: `--camera 1`; mic: `--list-mics`, then `--mic N`; only answer to "Nova ...": `--require-wake`).

**Display for the audience**: open `http://localhost:5001/display` in a browser on the projector/screen, press F11 (full screen).
Shows the last turns (You / Nova), source badge (Manual p.N / Camera / Chat), input (ESP / Laptop), response time,
the camera thumbnail for camera answers, and Listening… / Thinking… / Ready.

## Tune the ESP speech detection in the lab (1 minute, once)
The ESP streams continuously; the bridge closes an utterance after speech + 0.8 s silence.
1. Start the bridge with the level print: `STREAM_CALIBRATE=1 ./start_bridge.sh`
2. Stay quiet 5 s, then talk 5 s. Read the `📊 ESP ... level median X max Y | floor F | threshold T` lines.
   Quiet seconds should say `silence`, talking seconds `SPEECH`. Home values: floor −74 dBFS, speech −60 to −25, threshold ≈ −62 (auto).
3. If quiet shows `SPEECH` (noise too high): `STREAM_SPEECH_DBFS=-45 ./start_bridge.sh` (a fixed value a few dB under your talking `max`),
   or raise the auto margin: `STREAM_AUTO_MARGIN_DB=18`. If talking shows `silence`: speak closer / louder, or lower the fixed value.
4. Answers come too early (cut mid-sentence): `STREAM_END_SILENCE_S=1.2`. Too slow: `0.6`. Then restart without `STREAM_CALIBRATE`.

## Quick test for each link (2 minutes)
| # | Link | Test | Expect |
|---|---|---|---|
| 1 | Bridge up | Windows: `curl.exe http://localhost:5001/last_answer` | JSON with `"id"` |
| 2 | Speech + RAG | `cd C:\Grad\live` then `curl.exe -F "audio=@test_audio\live_tank.wav" http://localhost:5001/live` | "According to page 11 ... 0.35 litres" |
| 3 | Webcam client | Hold SPACE: "What is the capacity of the lubrication tank?" | Answer on screen + spoken |
| 4 | Camera route | Point at a manual page or part, hold SPACE: "What is this part?" | `route: image`, names a page (~10 s) |
| 5 | Frame for ESP | WSL: `curl -s -o /tmp/f.jpg -w "%{http_code}\n" http://$(ip route \| awk '/default/{print $3}'):5002/frame` | `200` |
| 6 | Wi-Fi path | Phone on the same Wi-Fi: `http://<Wi-Fi IP>:5001/last_answer` | JSON |
| 7 | ESP voice | Say "Nova, what is the capacity of the lubrication tank?" | Bridge log `🎤 ESP ... packets`, answer on the laptop window + spoken |
| 8 | ESP + camera | Say "Nova, what is this part?" with the camera on a part | `[ESP]`, `route: image` |
| 9 | Offline honesty | "What's the weather today?" | "I'm offline and can't check the weather." |

## If something fails
- **1 fails**: bridge not running or still loading → check the WSL terminal. Right after a (re)start, Windows `localhost` forwarding can need ~5 s: retry. `ollama ps` should show qwen3-vl `14%/86% CPU/GPU`.
- **Bridge log shows "mode B2" but answer "couldn't find that"**: question too vague for what the camera shows; ask about a specific part, or move closer.
- **5 fails**: firewall blocked Python → Windows Security > Firewall > Allow an app > Python: tick Private + Public; rerun the setup script.
- **6 fails**: lab Wi-Fi isolates devices → **fallback A**: phone hotspot, connect laptop + ESP to it, rerun step 2 (new IP!).
- **7: nothing in the bridge log**: ESP not reaching the laptop (wrong IP / not on same Wi-Fi), see 6.
- **7: log shows the transcript but "Wake word not detected"**: say "Nova" clearly at the start; the window shows "ESP heard (no wake word): ...".
- **7: answer only after ~30 s**: speech never "ends" (noise above the threshold) → see *Tune the ESP speech detection*; the hard cut is `STREAM_MAX_S=30`.
- **7: bridge log shows `🔇 dropped ... without speech`**: threshold too high for your voice level → tune (above).
- **Fallback B (ESP dead)**: demo everything with the webcam client (push-to-talk), same answers.
- After a reboot or Wi-Fi change: rerun step 2 (WSL and laptop IPs change).

## Talking points (measured)
One model (qwen3-vl 8B) for voice, text and camera; no model swaps. Text answers ~2–5 s, camera answers ~5–10 s,
Whisper ~1–2 s. Fully offline: Nova says so for weather, news, timers, email.
