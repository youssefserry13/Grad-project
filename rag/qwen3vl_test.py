"""Can qwen3-vl:8b-instruct replace BOTH qwen3:8b (text) and qwen2.5vl:7b (vision)? No swaps.

    python qwen3vl_test.py vram                 # step 1: fit at ctx 8192 / 4096, alone + with Whisper
    python qwen3vl_test.py text  --ctx N        # step 2: 60 benchmark Qs, R1 pipeline + frozen P03
    python qwen3vl_test.py image --ctx N        # step 3: Mode B2 (60 + honesty) with qwen3-vl
    python qwen3vl_test.py image-base           # step 3 baseline: same B2 v2 prompt with qwen2.5vl
    python qwen3vl_test.py voice --ctx N        # step 4: voice_sanity.json, bridge voice route
    python qwen3vl_test.py voice-base           # step 4 baseline: same with qwen3:8b
    python qwen3vl_test.py grade                # UNOFFICIAL qwen3:8b judge pass (one load)
    python qwen3vl_test.py report               # step 5 table

Nothing frozen is touched: P03 + the 60 questions are read from the manifest (read-only), the
bridge and the shared modules are not edited; the answer model is swapped in-process only
(rag_query.LLM_MODEL / image_pipeline.VISION_MODEL), and num_ctx is injected for qwen3-vl calls.
The judge stays qwen3:8b (same judge as R1 / image Level 1), run as a separate pass.
Outputs: results/rag_R1_qwen3vl.json (+ _run_meta), everything else in results/qwen3vl/.
"""
import argparse
import ast
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import image_level1 as L1
import image_pipeline as ip
import rag_query
from nova_prompt import B2_PROMPT_ID, NOVA_PROMPT, answer_pages, is_not_found, label_pages
from rag_benchmark60 import judge
from rag_common import (MAX_TOKENS, RESULTS_DIR, TEMPERATURE, TOP_K, load_benchmark_questions,
                        load_prompt)
from rag_export_r1 import clean_answer, hardware, ollama_model_info

VL = "qwen3-vl:8b-instruct"
QWEN3 = "qwen3:8b"
QWEN25VL = "qwen2.5vl:7b"
EMBED = "nomic-embed-text"
OUT = RESULTS_DIR / "qwen3vl"
R1_BASE = Path("/mnt/c/Grad/to_send/rag_R1.json")
R1_OUT = RESULTS_DIR / "rag_R1_qwen3vl.json"
NOVA = Path(__file__).resolve().parent.parent
BRIDGE = NOVA / "bridge" / "OllamaBridge.py"
VENV_PY = NOVA / "bridge" / "venv" / "bin" / "python"
SITE = next((NOVA / "bridge" / "venv" / "lib").glob("python3*/site-packages"))
WHISPER_ENV = {**os.environ, "LD_LIBRARY_PATH": f"{SITE}/nvidia/cublas/lib:{SITE}/nvidia/cudnn/lib"}
TEST_AUDIO = NOVA / "bridge" / "received_audio.wav"
VOICE_SET = Path(__file__).resolve().parent / "voice_sanity.json"
UNOFFICIAL = "UNOFFICIAL auto-grade (qwen3:8b judge, temp 0), not the official scoring"

ip.BIG_MODELS = (QWEN3, QWEN25VL, VL)          # use_only() must unload every big model
ip.KEEP_ALIVE[VL] = -1                          # the single model would stay pinned
ip.KEEP_ALIVE[QWEN25VL] = -1                    # batch runs: no reload between calls
CTX = {"value": None}


def client():
    return rag_query._resources()[0]


def patch_client():
    """Every chat/generate call to qwen3-vl gets options.num_ctx = CTX (same value on every call,
    otherwise Ollama reloads the model). Other models keep the server default (8192)."""
    c = client()
    if getattr(c, "_ctx_patched", False):
        return
    for name in ("chat", "generate"):
        orig = getattr(c, name)

        def wrapped(*a, _orig=orig, **kw):
            if kw.get("model") == VL and CTX["value"]:
                kw["options"] = {**(kw.get("options") or {}), "num_ctx": CTX["value"]}
            return _orig(*a, **kw)
        setattr(c, name, wrapped)
    c._ctx_patched = True


def load(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# ─── Measurements ────────────────────────────────────────────────────────────
def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def meminfo():
    mi = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
    return {"ram_used_mb": (mi["MemTotal"] - mi["MemAvailable"]) // 1024,
            "swap_used_mb": (mi["SwapTotal"] - mi["SwapFree"]) // 1024,
            "ram_total_mb": mi["MemTotal"] // 1024}


def ollama_rss_mb():
    rss = 0
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            if b"ollama" in open(f"/proc/{pid}/cmdline", "rb").read():
                rss += next(int(l.split()[1]) for l in open(f"/proc/{pid}/status")
                            if l.startswith("VmRSS")) // 1024
        except (OSError, StopIteration):
            pass
    return rss


def gpu_used_mb():
    out = sh(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
    return int(out.split(",")[0]) if out else None


def ps_split():
    """Per loaded model: size, VRAM part, GPU %, context (from the Ollama API) + raw `ollama ps`."""
    models = []
    for m in client().ps().models:
        size, vram = m.size or 0, m.size_vram or 0
        models.append({"model": m.model, "size_mb": size >> 20, "vram_mb": vram >> 20,
                       "gpu_pct": round(100 * vram / size, 1) if size else None,
                       "context": getattr(m, "context_length", None)})
    return {"models": models, "ollama_ps": sh(["ollama", "ps"]), "gpu_used_mb": gpu_used_mb(),
            **meminfo(), "ollama_rss_mb": ollama_rss_mb()}


class Peak:
    """Background sampler: peak RAM used, swap, Ollama RSS, GPU memory over a phase."""

    def __init__(self):
        self.peak = {"ram_used_mb": 0, "swap_used_mb": 0, "ollama_rss_mb": 0, "gpu_used_mb": 0}
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        i = 0
        while not self._stop.is_set():
            s = {**meminfo(), "ollama_rss_mb": ollama_rss_mb()}
            if i % 4 == 0:
                s["gpu_used_mb"] = gpu_used_mb() or 0
            for k in self.peak:
                self.peak[k] = max(self.peak[k], s.get(k, 0))
            i += 1
            self._stop.wait(0.5)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def record_phase(name, peak, ps_after, extra=None):
    phases = load(OUT / "phases.json", {})
    phases[name] = {"when": datetime.now().isoformat(timespec="seconds"), "ctx": CTX["value"],
                    "peak": peak, "ps_after": ps_after, **(extra or {})}
    save(OUT / "phases.json", phases)


def unload_all():
    loaded = ip.loaded_models()
    for m in loaded:
        client().generate(model=m, prompt="", keep_alive=0)
    deadline = time.monotonic() + 60
    while ip.loaded_models() and time.monotonic() < deadline:
        time.sleep(0.2)


def show_split(tag, snap):
    print(f"\n[{tag}] GPU used {snap['gpu_used_mb']} MiB | RAM used {snap['ram_used_mb']} MiB, "
          f"swap {snap['swap_used_mb']} MiB, ollama RSS {snap['ollama_rss_mb']} MiB")
    print(snap["ollama_ps"])


# ─── Step 1: VRAM fit ────────────────────────────────────────────────────────
WHISPER_CODE = """
import sys
from faster_whisper import WhisperModel
m = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
segs, _ = m.transcribe(sys.argv[1])
print("transcript:", " ".join(s.text for s in segs)[:80], flush=True)
print("READY", flush=True)
sys.stdin.read()
"""


def start_whisper():
    """Whisper large-v3-turbo on the GPU exactly like the bridge (cuda, float16), after one
    transcription so its working buffers are allocated. Kept alive until stop_whisper()."""
    p = subprocess.Popen([str(VENV_PY), "-c", WHISPER_CODE, str(TEST_AUDIO)], env=WHISPER_ENV,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    for line in p.stdout:
        print("  whisper:", line.rstrip())
        if line.startswith("READY"):
            return p
    raise RuntimeError("Whisper did not start")


def stop_whisper(p):
    p.stdin.close()
    p.wait(timeout=30)


def probe_requests():
    """One realistic text request (R1 pipeline, P03) and one B2 image request, both on VL."""
    q = load_benchmark_questions()[0]
    rag_query.LLM_MODEL = VL
    ip.VISION_MODEL = VL
    r = rag_query.ask_rag(q["question"])
    b64, info = L1.image_for(q["processed_path"])
    b = ip.answer_b2(b64, q["question"], MAX_TOKENS)
    return {"text": {"q": q["question_id"], "s": r["latency_s"], "usage": r["usage"]},
            "image": {"q": q["question_id"], "image": info, "s": b["vision_s"],
                      "usage": b["usage"], "answer": b["answer"][:120]}}


def step_vram(_):
    results = load(OUT / "vram.json", {})
    for with_whisper in (False, True):
        wp = None
        try:
            for ctx in (8192, 4096):
                tag = f"ctx{ctx}_{'with_whisper' if with_whisper else 'alone'}"
                unload_all()
                if with_whisper and wp is None:
                    wp = start_whisper()
                CTX["value"] = ctx
                before = ps_split()
                with Peak() as pk:
                    rag_query.retrieve_scored("warm-up")      # nomic-embed-text, like the bridge
                    t0 = time.perf_counter()
                    client().generate(model=VL, prompt="", keep_alive=-1)
                    load_s = round(time.perf_counter() - t0, 2)
                    after_load = ps_split()
                    probe = probe_requests()
                    after_req = ps_split()
                results[tag] = {"ctx": ctx, "whisper_on_gpu": with_whisper, "load_s": load_s,
                                "before": before, "after_load": after_load,
                                "after_requests": after_req, "peak": pk.peak, "probe": probe}
                save(OUT / "vram.json", results)
                show_split(f"{tag} after load ({load_s}s)", after_load)
                show_split(f"{tag} after text+image request", after_req)
                print(f"  peak {pk.peak}\n  probe tokens: text {probe['text']['usage']}, "
                      f"image {probe['image']['usage']}")
        finally:
            if wp:
                stop_whisper(wp)
    unload_all()


# ─── Step 2: text R1 pipeline ────────────────────────────────────────────────
def step_text(a):
    rag_query.LLM_MODEL = VL
    questions = load_benchmark_questions()
    assert len(questions) == 60
    records = load(R1_OUT, {})
    ip.use_only(VL)
    rag_query.ask_rag("warm-up")
    started = datetime.now().isoformat(timespec="seconds")
    with Peak() as pk:
        for q in questions:
            if q["question_id"] in records:
                continue
            r = rag_query.ask_rag(q["question"], k=TOP_K)
            records[q["question_id"]] = {
                "question_id": q["question_id"], "source": q["source"],
                "page": int(q["page"]) if q["page"] not in (None, "") else None,
                "image_id": None, "condition": "R1_qwen3vl", "prompt_id": "P03", "model": VL,
                "answer_raw": r["content"], "answer_clean": clean_answer(r["content"]),
                "latency_seconds": round(r["latency_s"], 2), "finish_reason": r["done_reason"],
                "usage": r["usage"], "attempts": 1, "system_merged": False,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            save(R1_OUT, records)
            print(f"{q['question_id']} {r['latency_s']:.2f}s  "
                  f"{records[q['question_id']]['answer_clean'][:80]!r}")
    snap = ps_split()
    record_phase("text", pk.peak, snap)
    show_split("text done", snap)
    p03 = load_prompt("P03")
    lat = [x["latency_seconds"] for x in records.values()]
    meta = {"R1_qwen3vl": {
        "description": "R1 pipeline (hybrid retrieval, top-5, frozen P03) with the answer model "
                       "swapped from qwen3:8b to qwen3-vl:8b-instruct; nothing else changed",
        "model": VL, "llm": ollama_model_info(VL), "embedding_model": ollama_model_info(EMBED),
        "think": False, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "num_ctx": CTX["value"],
        "keep_alive": -1, "prompt_id": "P03", "prompt_system": p03["system_prompt"],
        "prompt_user_template": p03["user_template"], "n_questions": len(records),
        "avg_latency_seconds": round(statistics.mean(lat), 3),
        "not_found_count": sum("NOT FOUND" in x["answer_clean"].upper() for x in records.values()),
        "ollama_ps": snap["models"], "peak_memory": pk.peak, "hardware": hardware(),
        "ollama_version": sh(["ollama", "--version"]).replace("ollama version is ", ""),
        "baseline": str(R1_BASE), "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
    }}
    save(R1_OUT.with_name("rag_R1_qwen3vl_run_meta.json"), meta)
    print(f"Saved {R1_OUT}")


# ─── Step 3: images, Mode B2 ─────────────────────────────────────────────────
def image_paths(tag):
    return {"B2": OUT / f"image_B2_{tag}.json", "HB2": OUT / f"image_honesty_B2_{tag}.json"}


def run_b2(model, tag):
    ip.VISION_MODEL = model
    questions = load_benchmark_questions()
    P = image_paths(tag)
    with Peak() as pk:
        L1.run_vision_mode("B2", questions, P, MAX_TOKENS)
        L1.run_honesty("B2", P, MAX_TOKENS)
    snap = ps_split()
    record_phase(f"image_{tag}", pk.peak, snap, {"prompt_id": B2_PROMPT_ID,
                                                  "image_max_side": ip.MAX_SIDE})
    show_split(f"image {tag} done", snap)


def step_image(a):
    run_b2(VL, "qwen3vl")


def step_image_base(a):
    CTX["value"] = None
    run_b2(QWEN25VL, "qwen25vl")


# ─── Step 4: voice route ─────────────────────────────────────────────────────
BRIDGE_NAMES = {"RAG_MIN_SIMILARITY", "RAG_MIN_SIMILARITY_LEX", "RAG_MIN_BM25", "RAG_MAX_TOKENS",
                "MACHINE_KEYWORDS", "CHAT_SYSTEM", "keyword_hits", "route", "strip_markdown",
                "chat_reply", "_answer_question"}


def bridge_voice_route(model):
    """The bridge's own voice-route code (routing, NOVA_PROMPT RAG, chat fallback, markdown
    strip), taken from OllamaBridge.py source without starting Flask/Whisper."""
    tree = ast.parse(BRIDGE.read_text(encoding="utf-8"))
    keep = [n for n in tree.body if
            (isinstance(n, ast.FunctionDef) and n.name in BRIDGE_NAMES) or
            (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in BRIDGE_NAMES
                                               for t in n.targets))]
    ns = {"re": re, "time": time, "rag_query": rag_query, "NOVA_PROMPT": NOVA_PROMPT,
          "answer_pages": answer_pages, "label_pages": label_pages, "client": client(),
          "TEXT_MODEL": model}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(BRIDGE), "exec"), ns)
    assert BRIDGE_NAMES <= ns.keys(), BRIDGE_NAMES - ns.keys()
    return ns["_answer_question"]


REFUSE_RE = re.compile(r"can['’]?t|cannot|can not|unable|not able|don['’]?t have|do not have|"
                       r"offline|no (internet|access|connection)|not connected|isn['’]?t possible",
                       re.I)
ACTION_RE = re.compile(r"\b(i['’]?ve|i have|i will|i'll)\s+(set|sent|scheduled|created|added)|"
                       r"\b(timer|reminder|alarm) (is )?set\b|\bdone\b", re.I)


def run_voice(model, tag):
    rag_query.LLM_MODEL = model
    answer = bridge_voice_route(model)
    cases = json.loads(VOICE_SET.read_text(encoding="utf-8"))["cases"]
    ip.use_only(model)
    answer("warm-up question about the spindle")       # not timed / not saved
    rows = {}
    with Peak() as pk:
        for c in cases:
            r = answer(c["question"])
            row = {**c, **r, "model": model}
            if c["kind"] == "trick":
                row["pass_rule_result"] = bool(r["not_found"])
            elif c["kind"] == "offline_honesty":
                row["pass_rule_result"] = (r["source"] == "chat" or r["not_found"]) and \
                    bool(REFUSE_RE.search(r["answer"])) and not ACTION_RE.search(r["answer"])
            rows[c["id"]] = row
            print(f"  {c['id']:<3} {r['source']:<4} {r['latency_s']:5.2f}s  {r['answer'][:110]!r}")
    snap = ps_split()
    record_phase(f"voice_{tag}", pk.peak, snap)
    save(OUT / f"voice_{tag}.json", rows)


def step_voice(a):
    run_voice(VL, "qwen3vl")


def step_voice_base(a):
    CTX["value"] = None
    run_voice(QWEN3, "qwen3")


# ─── Grading (UNOFFICIAL, qwen3:8b, separate pass) ───────────────────────────
def step_grade(a):
    questions = {q["question_id"]: q for q in load_benchmark_questions()}
    grades = load(OUT / "grades.json", {"note": UNOFFICIAL})
    ip.use_only(QWEN3)
    c = client()
    sets = {"R1": {k: v["answer_clean"] for k, v in load(R1_BASE, {}).items()},
            "R1_qwen3vl": {k: v["answer_clean"] for k, v in load(R1_OUT, {}).items()}}
    for tag in ("qwen3vl", "qwen25vl"):
        rows = load(image_paths(tag)["B2"], {})
        sets[f"B2_{tag}"] = {k: ("" if v["not_found"] else v["answer"]) for k, v in rows.items()}
    sets["B2_v1_qwen25vl"] = None                 # graded earlier (image_L1_grades.json)
    for name, answers in sets.items():
        if not answers:
            continue
        g = grades.setdefault(name, {})
        for qid, ans in answers.items():
            if qid not in g:
                q = questions[qid]
                g[qid] = judge(c, q["question"], q["ground_truth"], ans)
        save(OUT / "grades.json", grades)
        print(f"  {name}: {len(g)} graded")
    for tag in ("qwen3vl", "qwen3"):
        rows = load(OUT / f"voice_{tag}.json", {})
        g = grades.setdefault(f"voice_{tag}", {})
        for cid, r in rows.items():
            if r["kind"] == "manual" and cid not in g:
                g[cid] = "WRONG" if r["not_found"] else judge(c, r["question"], r["expected"],
                                                              r["answer"])
        save(OUT / "grades.json", grades)


# ─── Step 5: report ──────────────────────────────────────────────────────────
def _score(g):
    c, p, w = (sum(x == k for x in g) for k in ("CORRECT", "PARTIAL", "WRONG"))
    return {"correct": c, "partial": p, "wrong": w, "n": len(g),
            "score_pct": round(100 * (c + 0.5 * p) / len(g), 1) if g else None}


def _med(xs):
    xs = [x for x in xs if x is not None]
    return {"median": round(statistics.median(xs), 2), "mean": round(statistics.mean(xs), 2)} \
        if xs else None


def step_report(a):
    grades = load(OUT / "grades.json", {})
    phases = load(OUT / "phases.json", {})
    honesty_cases = {c["case_id"]: c for c in L1.load_honesty()}
    rep = {"grading": UNOFFICIAL}
    r1b, r1v = load(R1_BASE, {}), load(R1_OUT, {})
    for name, rows in (("R1", r1b), ("R1_qwen3vl", r1v)):
        if rows:
            rep[name] = {"auto_grade": _score(list(grades.get(name, {}).values())),
                         "model_time_s": _med([r["latency_seconds"] for r in rows.values()]),
                         "not_found": sum("NOT FOUND" in r["answer_clean"].upper()
                                          for r in rows.values()),
                         "truncated": sum(r["finish_reason"] == "length" for r in rows.values())}
    if r1b and r1v:
        rep["R1_same_answer"] = sum(r1b[k]["answer_clean"].strip() == r1v[k]["answer_clean"].strip()
                                    for k in r1v)
        rep["R1_grade_diff"] = {k: [grades["R1"].get(k), grades["R1_qwen3vl"].get(k)]
                                for k in r1v if grades.get("R1", {}).get(k)
                                != grades.get("R1_qwen3vl", {}).get(k)}
    for tag in ("qwen3vl", "qwen25vl"):
        P = image_paths(tag)
        rows, hon = load(P["B2"], {}), load(P["HB2"], {})
        if not rows:
            continue
        answered = [r for r in rows.values() if not r["not_found"]]
        hv = {cid: L1.honesty_verdict(r, honesty_cases[cid]) for cid, r in sorted(hon.items())}
        rep[f"B2_{tag}"] = {
            "auto_grade": _score(list(grades.get(f"B2_{tag}", {}).values())),
            "gold_page_named": sum(r["page_hit_cited"] for r in rows.values()),
            "no_page_named": [r["question_id"] for r in answered if not r["mentioned_pages"]],
            "wrong_page_named": [r["question_id"] for r in rows.values()
                                 if r["mentioned_pages"] and not r["page_hit_cited"]],
            "not_found": [r["question_id"] for r in rows.values() if r["not_found"]],
            "list_or_markdown": [r["question_id"] for r in rows.values()
                                 if L1.LIST_RE.search(r["answer"])],
            "truncated": sum(r["finish_reason"] == "length" for r in rows.values()),
            "model_time_s": _med([r["timings"]["model_s"] for r in rows.values()]),
            "honesty": {"pass": sum(v[0] for v in hv.values()), "n": len(hv),
                        "cases": {k: {"pass": v[0], "reason": v[1], "answer": hon[k]["answer"]}
                                  for k, v in hv.items()}},
        }
    for tag in ("qwen3vl", "qwen3"):
        rows = load(OUT / f"voice_{tag}.json", {})
        if not rows:
            continue
        g = grades.get(f"voice_{tag}", {})
        man = [r for r in rows.values() if r["kind"] == "manual"]
        rep[f"voice_{tag}"] = {
            "manual_ok": sum(g.get(r["id"]) in ("CORRECT", "PARTIAL") and r["source"] == "rag"
                             for r in man),
            "manual_grades": {r["id"]: g.get(r["id"]) for r in man},
            "manual_page_named": sum(r["page"] in [int(p) for p in
                                     re.findall(r"\bpages?\s+(\d+)", r["answer"], re.I)]
                                     for r in man),
            "trick_pass": sum(r.get("pass_rule_result", False) for r in rows.values()
                              if r["kind"] == "trick"),
            "offline_pass_rule": sum(r.get("pass_rule_result", False) for r in rows.values()
                                     if r["kind"] == "offline_honesty"),
            "latency_s": _med([r["latency_s"] for r in rows.values()]),
            "answers": {k: [r["source"], r["answer"]] for k, r in rows.items()},
        }
    rep["phases"] = {k: {"ctx": v["ctx"], "peak": v["peak"],
                         "split": [(m["model"], m["gpu_pct"], m["context"])
                                   for m in v["ps_after"]["models"]]} for k, v in phases.items()}
    rep["vram"] = {k: {"after_load": [(m["model"], m["size_mb"], m["vram_mb"], m["gpu_pct"],
                                       m["context"]) for m in v["after_load"]["models"]],
                       "after_requests": [(m["model"], m["gpu_pct"]) for m in
                                          v["after_requests"]["models"]],
                       "gpu_used_mb_after_load": v["after_load"]["gpu_used_mb"],
                       "peak": v["peak"], "load_s": v["load_s"],
                       "probe_prompt_tokens": [v["probe"]["text"]["usage"]["prompt_tokens"],
                                               v["probe"]["image"]["usage"]["prompt_tokens"]]}
                   for k, v in load(OUT / "vram.json", {}).items()}
    save(OUT / "report.json", rep)
    print(json.dumps(rep, indent=1, ensure_ascii=False)[:20000])


STEPS = {"vram": step_vram, "text": step_text, "image": step_image, "image-base": step_image_base,
         "voice": step_voice, "voice-base": step_voice_base, "grade": step_grade,
         "report": step_report}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=STEPS)
    ap.add_argument("--ctx", type=int, default=4096, help="num_ctx for qwen3-vl")
    a = ap.parse_args()
    CTX["value"] = a.ctx
    if a.step != "report":
        L1.preflight()                  # refuses if the bridge or a benchmark run is active
    patch_client()
    STEPS[a.step](a)
