"""Image Level 1: compare image pipeline Mode A (swap) vs Mode B (vision only) on the 60
benchmark questions with their processed page images. The model is NOT told the page.

    python image_level1.py                 # full run: answers for A and B, swap timing, grading
    python image_level1.py --sample 3      # smoke test on 3 questions (writes *_sample3 files)
    python image_level1.py --grade-only    # re-grade + re-summarise existing outputs

Run order (keeps model swaps to a minimum):
  A1  qwen2.5vl loaded once -> 60 descriptions
  A2  qwen3 loaded once     -> 60 retrievals + answers   (temp 0, so same answers as the bridge)
  As  5 questions the bridge way (swap every time) -> measured swap_s for Mode A
  B1  qwen2.5vl loaded once -> 60 retrievals + vision answers
  Bs  5 questions the bridge way (incl. qwen3 reload) -> measured swap_s for Mode B
  G   qwen3 loaded once     -> UNOFFICIAL auto-grade of all answers (separate file)

Inputs (read-only): ~/benchmark_v2 manifest + benchmark_images/processed/.
Outputs (rag/results/, git-ignored): image_L1_A.json, image_L1_B.json (rows like local_S1.json
plus pages/timings), image_L1_swaps.json, image_L1_grades.json, image_L1_run_meta.json,
image_L1_summary.json. Re-running resumes; finished rows are never recomputed or edited.
Do not run this while the bridge or a benchmark run is using Ollama.
"""
import argparse
import json
import platform
import socket
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import httpx

import image_pipeline as ip
from nova_prompt import (DESCRIBE_SYSTEM, MODE_A_QUESTION, NOVA_PROMPT, NOVA_VISION_PROMPT,
                         mentioned_pages)
from rag_benchmark60 import judge
from rag_common import (COLLECTION, MANIFEST_PATH, MAX_TOKENS, RESULTS_DIR, TEMPERATURE, TOP_K,
                        load_benchmark_questions)
from rag_ingest import CHUNKING_VERSION

BENCH_DIR = Path.home() / "benchmark_v2"
SWAP_N = 5
BRIDGE_PORT = 5001
UNOFFICIAL = "UNOFFICIAL auto-grade (qwen3:8b judge, temp 0), not the official scoring"


# ─── Files ───────────────────────────────────────────────────────────────────
def paths(suffix):
    p = lambda name: RESULTS_DIR / f"image_L1_{name}{suffix}.json"
    return {"A": p("A"), "B": p("B"), "swaps": p("swaps"), "grades": p("grades"),
            "meta": p("run_meta"), "summary": p("summary")}


def load(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save(path, data):
    RESULTS_DIR.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# ─── Safety checks ───────────────────────────────────────────────────────────
def preflight():
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", BRIDGE_PORT)) == 0:
            raise SystemExit(f"The bridge is running on port {BRIDGE_PORT}. Stop it first.")
    busy = subprocess.run(["pgrep", "-af", "benchmark_qa_local|rag_benchmark60|rag_eval"],
                          capture_output=True, text=True).stdout.strip()
    if busy:
        raise SystemExit(f"A benchmark run seems to be in progress:\n{busy}")


# ─── Data ────────────────────────────────────────────────────────────────────
def expected_page(q):
    return int(q["pdf_page"] or q["page"])


_img_cache = {}


def image_for(q):
    """Processed benchmark image -> same in-memory preprocessing as the bridge (read-only)."""
    if q["processed_path"] not in _img_cache:
        _img_cache[q["processed_path"]] = ip.preprocess((BENCH_DIR / q["processed_path"]).read_bytes())
    return _img_cache[q["processed_path"]]


def base_row(q, mode):
    b64, info = image_for(q)
    return {"question_id": q["question_id"], "source": q["source"], "page": expected_page(q),
            "image_id": q["image_id"], "condition": f"IMG_L1_{mode}",
            "prompt_id": "NOVA_IMG_A" if mode == "A" else "NOVA_IMG_B",
            "model": f"{ip.VISION_MODEL} + {ip.TEXT_MODEL}" if mode == "A" else ip.VISION_MODEL,
            "question": q["question"], "image": info}


def complete_row(row):
    """Fill the S1-style fields from a finished pipeline result."""
    ip.finish(row, row["condition"][-1])
    row["answer_clean"] = ip._strip_think(row["answer_raw"])
    row["latency_seconds"] = row["timings"]["model_s"]       # model time, no swaps
    row["finish_reason"] = row.pop("done_reason")
    row["page_hit_cited"] = row["page"] in row["mentioned_pages"]
    row["page_hit_retrieved"] = row["page"] in row["retrieved_pages"]
    row.update(attempts=1, system_merged=False,
               timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return row


def log(row):
    t = row["timings"]
    hit = "c" if row["page_hit_cited"] else ("r" if row["page_hit_retrieved"] else "-")
    print(f"  {row['question_id']} p{row['page']:<3} [{hit}] {t['model_s']:6.2f}s  "
          f"{'NF ' if row['not_found'] else ''}{row['answer'][:80]}")


# ─── Runs ────────────────────────────────────────────────────────────────────
def run_mode_a(questions, P, max_tokens):
    rows = load(P["A"], {})
    todo = [q for q in questions if "description" not in rows.get(q["question_id"], {})]
    if todo:
        print(f"\n[A1] {len(todo)} descriptions with {ip.VISION_MODEL}  "
              f"(load {ip.use_only(ip.VISION_MODEL)}s)")
        for q in todo:
            row = base_row(q, "A")
            row.update(ip.describe(image_for(q)[0], q["question"], max_tokens))
            rows[q["question_id"]] = row
            save(P["A"], rows)
            print(f"  {q['question_id']} {row['vision_s']:6.2f}s  "
                  f"{row['description'][:90].replace(chr(10), ' | ')}")
    todo = [q for q in questions if "timings" not in rows[q["question_id"]]]
    if todo:
        print(f"\n[A2] {len(todo)} answers with {ip.TEXT_MODEL}  "
              f"(load {ip.use_only(ip.TEXT_MODEL)}s)")
        for q in todo:
            row = rows[q["question_id"]]
            row.update(ip.answer_from_description(q["question"], row["description"], max_tokens))
            rows[q["question_id"]] = complete_row(row)
            save(P["A"], rows)
            log(row)
    return rows


def run_mode_b(questions, P, max_tokens):
    rows = load(P["B"], {})
    todo = [q for q in questions if "timings" not in rows.get(q["question_id"], {})]
    if todo:
        print(f"\n[B1] {len(todo)} vision answers with {ip.VISION_MODEL}  "
              f"(load {ip.use_only(ip.VISION_MODEL)}s)")
        for q in todo:
            row = base_row(q, "B")
            row.update(ip.answer_with_vision(image_for(q)[0], q["question"], max_tokens))
            rows[q["question_id"]] = complete_row(row)
            save(P["B"], rows)
            log(row)
    return rows


def measure_swaps(mode, questions, rows, P, max_tokens):
    """Run SWAP_N questions exactly like the bridge (qwen3 resident before each request)."""
    swaps = load(P["swaps"], {})
    if len(swaps.get(mode, [])) >= min(SWAP_N, len(questions)):
        return swaps
    print(f"\n[{mode}s] swap timing, {SWAP_N} questions the bridge way")
    ip.use_only(ip.TEXT_MODEL)                 # bridge idle state; not timed
    runs = []
    for q in questions[:SWAP_N]:
        r = ip.run(image_for(q)[0], q["question"], mode, max_tokens)
        same = r["answer"] == rows[q["question_id"]]["answer"]
        runs.append({"question_id": q["question_id"], "timings": r["timings"],
                     "swap_steps": r["swap_steps"],
                     "same_answer_as_batch": same, "answer": r["answer"]})
        t = r["timings"]
        print(f"  {q['question_id']} swap {t['swap_s']:5.2f}s {r['swap_steps']}  model {t['model_s']:6.2f}s  "
              f"wall {t['wall_s']:6.2f}s  same={same}")
    swaps[mode] = runs
    save(P["swaps"], swaps)
    return swaps


def grade(questions, P):
    """Separate pass, qwen3 loaded once. Raw output files are not touched."""
    grades = load(P["grades"], {"note": UNOFFICIAL, "A": {}, "B": {}})
    client = ip._client()
    loaded = False
    for mode in ("A", "B"):
        rows = load(P[mode], {})
        todo = [q for q in questions if q["question_id"] in rows
                and "timings" in rows[q["question_id"]] and q["question_id"] not in grades[mode]]
        if not todo:
            continue
        if not loaded:
            print(f"\n[G] grading with {ip.TEXT_MODEL}  (load {ip.use_only(ip.TEXT_MODEL)}s)")
            loaded = True
        for q in todo:
            row = rows[q["question_id"]]
            g = "WRONG" if row["not_found"] else judge(client, q["question"], q["ground_truth"],
                                                         row["answer"])
            grades[mode][q["question_id"]] = g
            save(P["grades"], grades)
        print(f"  {mode}: graded {len(todo)}")
    return grades


# ─── Meta + summary ──────────────────────────────────────────────────────────
def run_meta(max_tokens, n, sample):
    client = ip._client()
    listed = {m.model: m for m in client.list().models}
    models = {}
    for m in ip.BIG_MODELS + ("nomic-embed-text",):
        name = m if m in listed else f"{m}:latest"
        d = client.show(name).details
        models[m] = {"quantization": d.quantization_level, "parameter_size": d.parameter_size,
                     "family": d.family, "digest": getattr(listed.get(name), "digest", None)}
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
                            capture_output=True, text=True).stdout.strip()
    return {
        "test": "image_level1", "modes": {
            "A": "qwen2.5vl describes image -> hybrid RAG on question+description -> qwen3 answers",
            "B": "hybrid RAG on question -> qwen2.5vl answers from image + question + top-5"},
        "models": models,
        "ollama_version": httpx.get("http://127.0.0.1:11434/api/version").json().get("version"),
        "hardware": {"gpu": gpu, "platform": platform.platform(), "python": platform.python_version()},
        "settings": {"temperature": TEMPERATURE, "max_tokens": max_tokens, "top_k": TOP_K,
                     "retrieval": "hybrid (BM25 + vector, RRF)", "collection": COLLECTION,
                     "chunking": CHUNKING_VERSION, "image_max_side": ip.MAX_SIDE,
                     "image_jpeg_quality": ip.JPEG_QUALITY, "think": False,
                     "page_given_to_model": False, "swap_measure_n": SWAP_N},
        "inputs": {"manifest": str(MANIFEST_PATH), "images": str(BENCH_DIR / "benchmark_images/processed"),
                   "n_questions": n, "sample": sample},
        "prompts": {"describe_system": DESCRIBE_SYSTEM, "mode_a_question": MODE_A_QUESTION,
                    "mode_a_system": NOVA_PROMPT["system_prompt"],
                    "mode_b_system": NOVA_VISION_PROMPT},
        "code_commit": commit,
        "batching_note": ("Mode A descriptions and answers were run in two batches (one model "
                          "load each); temp 0 gives the same answers as the per-request bridge "
                          "flow. Swap cost is measured separately on the first "
                          f"{SWAP_N} questions run exactly like the bridge."),
    }


def _stats(xs):
    xs = [x for x in xs if x is not None]
    return {"median": round(statistics.median(xs), 2), "mean": round(statistics.mean(xs), 2)} \
        if xs else {"median": None, "mean": None}


def summarize(questions, P):
    grades = load(P["grades"], {"A": {}, "B": {}})
    swaps = load(P["swaps"], {})
    out = {"grading": UNOFFICIAL}
    for mode in ("A", "B"):
        rows = load(P[mode], {})
        rows = [rows[q["question_id"]] for q in questions
                if "timings" in rows.get(q["question_id"], {})]
        if not rows:
            continue
        n = len(rows)
        g = [grades.get(mode, {}).get(r["question_id"]) for r in rows]
        c, p, w = g.count("CORRECT"), g.count("PARTIAL"), g.count("WRONG")
        sw = [s["timings"]["swap_s"] for s in swaps.get(mode, [])]
        swap_mean = statistics.mean(sw) if sw else None
        model = [r["timings"]["model_s"] for r in rows]
        out[mode] = {
            "n": n,
            "auto_grade": {"correct": c, "partial": p, "wrong": w, "ungraded": g.count(None),
                           "score_pct": round(100 * (c + 0.5 * p) / n, 1)},
            "page_hit_cited": sum(r["page_hit_cited"] for r in rows),
            "page_hit_retrieved": sum(r["page_hit_retrieved"] for r in rows),
            "not_found": sum(r["not_found"] for r in rows),
            "truncated": sum(r["finish_reason"] == "length" for r in rows),
            "model_time_s": _stats(model),
            "swap_s": {"mean": round(swap_mean, 2) if sw else None, "n": len(sw),
                       "runs": [round(x, 2) for x in sw]},
            "bridge_time_s": _stats([m + swap_mean for m in model]) if sw else None,
            "bridge_wall_measured_s": _stats([s["timings"]["wall_s"] for s in swaps.get(mode, [])]),
            "swap_runs_same_answer": f"{sum(s['same_answer_as_batch'] for s in swaps.get(mode, []))}"
                                     f"/{len(sw)}",
            "stage_medians_s": {k: _stats([r["timings"][k] for r in rows])["median"]
                                for k in ("vision_s", "rag_s", "answer_s")},
        }
    save(P["summary"], out)

    print(f"\n{'=' * 78}\nIMAGE LEVEL 1 SUMMARY   (correctness = {UNOFFICIAL})\n{'=' * 78}")
    hdr = f"{'':26}" + "".join(f"{'Mode ' + m:>24}" for m in ("A", "B") if m in out)
    print(hdr)
    def line(label, fn):
        print(f"{label:26}" + "".join(f"{fn(out[m]):>24}" for m in ("A", "B") if m in out))
    line("questions", lambda s: str(s["n"]))
    line("auto-grade C/P/W", lambda s: "{correct}/{partial}/{wrong}".format(**s["auto_grade"]))
    line("auto-grade score", lambda s: f"{s['auto_grade']['score_pct']}%")
    line("page hit (cited)", lambda s: f"{s['page_hit_cited']}/{s['n']}")
    line("page hit (retrieved@5)", lambda s: f"{s['page_hit_retrieved']}/{s['n']}")
    line("NOT FOUND", lambda s: str(s["not_found"]))
    line("truncated (1024 tok)", lambda s: str(s["truncated"]))
    line("model time med/mean s", lambda s: "{median} / {mean}".format(**s["model_time_s"]))
    line("swap mean s (measured)", lambda s: f"{s['swap_s']['mean']} (n={s['swap_s']['n']})")
    line("bridge time med/mean s", lambda s: "{median} / {mean}".format(**s["bridge_time_s"])
         if s["bridge_time_s"] else "n/a")
    line("bridge wall measured med", lambda s: str(s["bridge_wall_measured_s"]["median"]))
    line("swap-run answer identical", lambda s: s["swap_runs_same_answer"])
    line("median vision/rag/answer", lambda s: "{vision_s}/{rag_s}/{answer_s}".format(
        **s["stage_medians_s"]))
    print("model time = no swaps; bridge time = model time + mean measured swap "
          "(A: qwen3->vl->qwen3, B: qwen3->vl + qwen3 reload).")
    print(f"Saved {P['summary']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, help="only the first N questions (separate files)")
    ap.add_argument("--grade-only", action="store_true", help="re-grade + summarise, no answers")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    a = ap.parse_args()

    questions = load_benchmark_questions()
    assert len(questions) == 60, f"expected 60 benchmark questions, got {len(questions)}"
    if a.sample:
        questions = questions[:a.sample]
    P = paths(f"_sample{a.sample}" if a.sample else "")
    preflight()

    if not a.grade_only:
        meta = load(P["meta"], None) or run_meta(a.max_tokens, len(questions), a.sample)
        meta.setdefault("started", datetime.now().isoformat(timespec="seconds"))
        save(P["meta"], meta)
        t0 = time.perf_counter()
        rows_a = run_mode_a(questions, P, a.max_tokens)
        measure_swaps("A", questions, rows_a, P, a.max_tokens)
        rows_b = run_mode_b(questions, P, a.max_tokens)
        measure_swaps("B", questions, rows_b, P, a.max_tokens)
        meta["finished"] = datetime.now().isoformat(timespec="seconds")
        meta["last_run_wall_s"] = round(time.perf_counter() - t0, 1)
        save(P["meta"], meta)
    else:
        P_grades = P["grades"]
        if P_grades.exists():
            P_grades.unlink()                 # fresh grading pass over existing outputs
    grade(questions, P)
    summarize(questions, P)
    ip.use_only(ip.TEXT_MODEL)                # leave qwen3 resident, like the bridge expects


if __name__ == "__main__":
    main()
