"""Checks against a RUNNING bridge (http://127.0.0.1:5001), e.g. after `MODEL_MODE=single`.

    python bridge_check.py ram      # 20 mixed requests (/audio, /ask, /image) + RAM sampling
    python bridge_check.py voice    # voice_sanity.json through /ask (the bridge's voice route)
    python bridge_check.py grade    # bridge STOPPED: UNOFFICIAL qwen3:8b judge on the manual cases

RAM is read from /proc/meminfo like `free`: used = total - free - buffers - cache - SReclaimable
(real process memory), buff/cache separately (file cache incl. mmapped model files, reclaimable),
available = what the kernel can still hand out. Outputs: rag/results/bridge_check/.
"""
import argparse
import base64
import json
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import httpx

from rag_common import RESULTS_DIR, load_benchmark_questions

URL = "http://127.0.0.1:5001"
NOVA = Path(__file__).resolve().parent.parent
BENCH_DIR = Path.home() / "benchmark_v2"
OUT = RESULTS_DIR / "bridge_check"
VOICE_SET = Path(__file__).resolve().parent / "voice_sanity.json"
REFUSE_RE = re.compile(r"can['’]?t|cannot|can not|unable|not able|don['’]?t have|do not have|"
                       r"offline|no (internet|access|connection)|not connected", re.I)
ACTION_CLAIM_RE = re.compile(r"\b(i['’]?ve|i have|i will|i['’]ll)\s+(set|sent|scheduled|created|"
                             r"added|remind)|\b(timer|reminder|alarm|email) (is |has been )?(set|sent)\b",
                             re.I)


def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ─── Memory ──────────────────────────────────────────────────────────────────
def mem():
    m = {l.split(":")[0]: int(l.split()[1]) // 1024 for l in open("/proc/meminfo")}
    cache = m["Buffers"] + m["Cached"] + m["SReclaimable"]
    return {"total_mb": m["MemTotal"], "used_mb": m["MemTotal"] - m["MemFree"] - cache,
            "buff_cache_mb": cache, "free_mb": m["MemFree"], "available_mb": m["MemAvailable"],
            "swap_used_mb": m["SwapTotal"] - m["SwapFree"]}


def proc_rss():
    out = subprocess.run(["ps", "-eo", "rss,args"], capture_output=True, text=True).stdout
    rss = {"bridge": 0, "ollama_runners": 0}
    for line in out.splitlines()[1:]:
        kb, args = line.strip().split(None, 1)
        if "OllamaBridge.py" in args:
            rss["bridge"] += int(kb) // 1024
        elif "llama-server" in args or "ollama runner" in args:
            rss["ollama_runners"] += int(kb) // 1024
    return rss


def snapshot():
    return {**mem(), "rss_mb": proc_rss(),
            "free_h": subprocess.run(["free", "-h"], capture_output=True, text=True).stdout,
            "ollama_ps": subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout}


class Sampler:
    def __init__(self):
        self.samples, self._stop = [], threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.samples.append({"t": time.time(), **mem(), **proc_rss()})
            self._stop.wait(1.0)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    def peaks(self):
        keys = ("used_mb", "buff_cache_mb", "swap_used_mb", "bridge", "ollama_runners")
        return {**{f"max_{k}": max(s[k] for s in self.samples) for k in keys},
                "min_available_mb": min(s["available_mb"] for s in self.samples),
                "n_samples": len(self.samples)}


# ─── Requests ────────────────────────────────────────────────────────────────
def ask(c, text):
    return c.post(f"{URL}/ask", json={"text": text}).json()


def audio(c, path):
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return c.post(f"{URL}/audio", json={"state": 101, "payload": b64}).json()


def image(c, path, question):
    with open(path, "rb") as f:
        return c.post(f"{URL}/image", files={"image": (Path(path).name, f, "image/jpeg")},
                      data={"question": question}).json()


def mixed_requests():
    """20 requests: 4 audio (Whisper), 8 text voice-route, 8 image (B2)."""
    qs = load_benchmark_questions()
    pick = lambda i: (BENCH_DIR / qs[i]["processed_path"], qs[i]["question"])
    audio_files = [NOVA / "live_stream.wav", NOVA / "bridge" / "received_audio.wav",
                   NOVA / "received_audio.m4a", NOVA / "live_stream.wav"]
    texts = ["What is the capacity of the lubrication tank?", "What's the weather today?",
             "How often should the oil of the feed gear box be exchanged?",
             "What does error code E21 mean on the display?", "Set a timer for 5 minutes.",
             "What is the rated power of the main spindle motor?", "Tell me a fun fact.",
             "What must be confirmed before the power is turned on?"]
    images = [pick(i) for i in (0, 6, 12, 18, 24, 30, 42)] + \
             [("/mnt/c/Windows/Web/Screen/img102.jpg", "What is this?")]
    reqs = []
    for i in range(8):                      # interleave: text, image, (audio every other round)
        reqs.append(("ask", texts[i]))
        reqs.append(("image", images[i]))
        if i % 2 == 1:
            reqs.append(("audio", audio_files[i // 2]))
    return reqs


def step_ram(_):
    before = snapshot()
    print("BEFORE\n" + before["free_h"] + before["ollama_ps"])
    rows = []
    with httpx.Client(timeout=300) as c, Sampler() as s:
        for kind, arg in mixed_requests():
            t0 = time.perf_counter()
            if kind == "ask":
                r = ask(c, arg)
                short = f"{r.get('source')}: {r.get('answer', '')[:70]}"
            elif kind == "image":
                r = image(c, *arg)
                short = f"{r.get('mode')} {r.get('timings', {}).get('total_s')}s: {r.get('answer', '')[:60]}"
            else:
                r = audio(c, arg)
                short = f"{r.get('status')}: {r.get('transcript', '')[:60]}"
            dt = round(time.perf_counter() - t0, 2)
            m = mem()
            rows.append({"kind": kind, "arg": str(arg), "wall_s": dt, "response": r, "mem_after": m})
            print(f"{len(rows):>2} {kind:<5} {dt:6.2f}s used {m['used_mb']:>5} cache "
                  f"{m['buff_cache_mb']:>5} avail {m['available_mb']:>5} swap "
                  f"{m['swap_used_mb']:>4}  {short!r}")
    after = snapshot()
    print("AFTER\n" + after["free_h"] + after["ollama_ps"])
    print("PEAKS", json.dumps(s.peaks()))
    save("ram.json", {"when": datetime.now().isoformat(timespec="seconds"), "before": before,
                      "after": after, "peaks": s.peaks(), "requests": rows,
                      "samples": s.samples})


def step_voice(_):
    cases = json.loads(VOICE_SET.read_text(encoding="utf-8"))["cases"]
    rows = {}
    with httpx.Client(timeout=300) as c:
        for case in cases:
            r = ask(c, case["question"])
            row = {**case, **{k: r.get(k) for k in ("source", "answer", "not_found", "pages",
                                                     "route_reason", "latency_s")}}
            if case["kind"] == "trick":
                row["rule_pass"] = bool(r["not_found"])
            elif case["kind"] == "offline_honesty":
                row["rule_pass"] = r["source"] == "chat" and bool(REFUSE_RE.search(r["answer"])) \
                    and not ACTION_CLAIM_RE.search(r["answer"])
            else:
                row["rule_pass"] = r["source"] == "rag" and not r["not_found"]
            rows[case["id"]] = row
            print(f"{case['id']:<3} {r['source']:<4} {r['route_reason']:<9} "
                  f"{'PASS' if row['rule_pass'] else 'FAIL'} {r['latency_s']:5.2f}s  {r['answer']!r}")
    save("voice_sanity.json", rows)


def step_grade(_):
    """Run with the bridge stopped: loads qwen3:8b (the usual UNOFFICIAL judge)."""
    import ollama
    from rag_benchmark60 import judge
    path = OUT / "voice_sanity.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    c = ollama.Client()
    for r in rows.values():
        if r["kind"] == "manual":
            r["judge"] = "WRONG" if r["not_found"] else judge(c, r["question"], r["expected"],
                                                              r["answer"])
            r["page_named"] = r["page"] in r["pages"] and \
                bool(re.search(rf"\bpages?\s+{r['page']}\b", r["answer"], re.I))
            r["rule_pass"] = r["source"] == "rag" and r["judge"] in ("CORRECT", "PARTIAL")
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    for kind in ("manual", "trick", "offline_honesty"):
        k = [r for r in rows.values() if r["kind"] == kind]
        print(f"{kind}: {sum(r['rule_pass'] for r in k)}/{len(k)}")
    for r in rows.values():
        if r["kind"] == "manual":
            print(f"  {r['id']} {r['judge']} page_named={r['page_named']} {r['answer'][:90]!r}")


STEPS = {"ram": step_ram, "voice": step_voice, "grade": step_grade}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=STEPS)
    STEPS[ap.parse_args().step](None)
