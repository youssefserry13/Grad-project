"""Image Level 1: compare the image pipelines on the 60 benchmark questions with their processed
page images (the model is NOT told the page), plus an honesty set of trick cases.

  A   qwen2.5vl describes the image -> RAG on question + description -> qwen3 answers (swap)
  B   RAG on the question -> qwen2.5vl answers from image + top-5 chunks (original prompt)
  B2  like B, stricter prompt (spoken, ends with "Page N", no general knowledge) + score gate

    python image_level1.py                    # all modes + honesty set, swap timing, grading
    python image_level1.py --modes B2         # only some modes (finished rows are skipped)
    python image_level1.py --sample 3         # smoke test on 3 questions (writes *_sample3 files)
    python image_level1.py --grade-only       # re-grade the given modes + re-summarise

Run order (keeps model swaps to a minimum):
  A1/A2  qwen2.5vl once -> 60 descriptions; qwen3 once -> 60 retrievals + answers
  As     5 questions the bridge way (swap every time) -> measured swap_s for Mode A
  B1     qwen2.5vl once -> 60 answers;  Bs  5 bridge-way runs (incl. qwen3 reload)
  B21    qwen2.5vl once -> 60 answers;  H   honesty set (B, B2), same model, no swaps
  B2s    5 bridge-way runs
  G      qwen3 once -> UNOFFICIAL auto-grade of the 60-question answers (separate file).
         Honesty cases are scored by rule: pass = "not found" or a request to clarify.

Inputs (read-only): ~/benchmark_v2 manifest + benchmark_images/processed/, honesty_set.json.
Outputs (rag/results/, git-ignored): image_L1_<mode>.json (rows like local_S1.json plus
pages/timings), image_L1_honesty_<mode>.json, image_L1_swaps.json, image_L1_grades.json,
image_L1_run_meta.json, image_L1_summary.json. Re-running resumes; finished rows are never
recomputed or edited. Do not run this while the bridge or a benchmark run is using Ollama.
"""
import argparse
import json
import platform
import re
import socket
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import httpx

import image_pipeline as ip
from nova_prompt import (B2_PROMPT_ID, B2_USER, DESCRIBE_SYSTEM, MODE_A_QUESTION, NOVA_PROMPT,
                         NOVA_VISION_PROMPT, NOVA_VISION_PROMPT_B2, is_not_found,
                         mentioned_pages)
from rag_benchmark60 import judge
from rag_common import (COLLECTION, MANIFEST_PATH, MAX_TOKENS, RESULTS_DIR, TEMPERATURE, TOP_K,
                        load_benchmark_questions)
from rag_ingest import CHUNKING_VERSION

BENCH_DIR = Path.home() / "benchmark_v2"
HONESTY_PATH = Path(__file__).resolve().parent / "honesty_set.json"
ALL_MODES = ("A", "B", "B2")
SUMMARY_MODES = ("A", "B", "B2_v1", "B2")     # B2_v1 = archived first B2 run (summary only)
PROMPT_ID = {"A": "NOVA_IMG_A", "B": "NOVA_IMG_B", "B2": B2_PROMPT_ID}
SWAP_N = 5
BRIDGE_PORT = 5001
UNOFFICIAL = "UNOFFICIAL auto-grade (qwen3:8b judge, temp 0), not the official scoring"
VISION_ANSWER = {"B": ip.answer_with_vision, "B2": ip.answer_b2}
MODE_INFO = {
    "A": "qwen2.5vl describes image -> hybrid RAG on question+description -> qwen3 answers",
    "B": "hybrid RAG on question -> qwen2.5vl answers from image + question + top-5",
    "B2": ("like B with a stricter prompt (1-3 spoken sentences, ends with 'Page N', no general "
           "knowledge, rules repeated in the user turn) + retrieval-score gate (voice route rule)"),
}
CLARIFY_RE = re.compile(r"clarif|more specific|which part|specific part|could you (tell|ask|say)",
                        re.I)
LIST_RE = re.compile(r"^\s*(\d+[.)]|[-*•])\s|\*\*|^#", re.M)


# ─── Files ───────────────────────────────────────────────────────────────────
def paths(suffix):
    p = lambda name: RESULTS_DIR / f"image_L1_{name}{suffix}.json"
    out = {m: p(m) for m in SUMMARY_MODES}
    out.update({f"H{m}": p(f"honesty_{m}") for m in SUMMARY_MODES})
    out.update(swaps=p("swaps"), grades=p("grades"), meta=p("run_meta"), summary=p("summary"))
    return out


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


def image_for(rel_or_abs):
    """Image file (relative paths are under ~/benchmark_v2, read-only) -> same in-memory
    preprocessing as the bridge."""
    if rel_or_abs not in _img_cache:
        p = Path(rel_or_abs)
        p = p if p.is_absolute() else BENCH_DIR / p
        _img_cache[rel_or_abs] = ip.preprocess(p.read_bytes())
    return _img_cache[rel_or_abs]


def load_honesty():
    return json.loads(HONESTY_PATH.read_text(encoding="utf-8"))["cases"]


def base_row(q, mode):
    _, info = image_for(q["processed_path"])
    return {"question_id": q["question_id"], "source": q["source"], "page": expected_page(q),
            "image_id": q["image_id"], "condition": f"IMG_L1_{mode}",
            "prompt_id": PROMPT_ID[mode],
            "model": f"{ip.VISION_MODEL} + {ip.TEXT_MODEL}" if mode == "A" else ip.VISION_MODEL,
            "question": q["question"], "image": info}


def complete_row(row, mode):
    """Fill the S1-style fields from a finished pipeline result."""
    ip.finish(row, mode)
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
    flags = ("GATED " if row.get("gated") else "") + ("NF " if row["not_found"] else "")
    print(f"  {row['question_id']} p{row['page'] or '-':<3} [{hit}] {t['model_s']:6.2f}s  "
          f"{flags}{row['answer'][:80]}".replace("\n", " "))


# ─── Runs ────────────────────────────────────────────────────────────────────
def run_mode_a(questions, P, max_tokens):
    rows = load(P["A"], {})
    todo = [q for q in questions if "description" not in rows.get(q["question_id"], {})]
    if todo:
        print(f"\n[A1] {len(todo)} descriptions with {ip.VISION_MODEL}  "
              f"(load {ip.use_only(ip.VISION_MODEL)}s)")
        for q in todo:
            row = base_row(q, "A")
            row.update(ip.describe(image_for(q["processed_path"])[0], q["question"], max_tokens))
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
            rows[q["question_id"]] = complete_row(row, "A")
            save(P["A"], rows)
            log(row)
    return rows


def run_vision_mode(mode, questions, P, max_tokens):
    """Mode B / B2 on the 60 questions, qwen2.5vl loaded once."""
    rows = load(P[mode], {})
    todo = [q for q in questions if "timings" not in rows.get(q["question_id"], {})]
    if todo:
        print(f"\n[{mode}1] {len(todo)} vision answers with {ip.VISION_MODEL}  "
              f"(load {ip.use_only(ip.VISION_MODEL)}s)")
        for q in todo:
            row = base_row(q, mode)
            row.update(VISION_ANSWER[mode](image_for(q["processed_path"])[0], q["question"],
                                           max_tokens))
            rows[q["question_id"]] = complete_row(row, mode)
            save(P[mode], rows)
            log(row)
    return rows


def run_honesty(mode, P, max_tokens):
    """Honesty trick cases for a vision mode (B / B2), qwen2.5vl loaded once."""
    key = f"H{mode}"
    rows = load(P[key], {})
    todo = [c for c in load_honesty() if "timings" not in rows.get(c["case_id"], {})]
    if not todo:
        return rows
    print(f"\n[H-{mode}] {len(todo)} honesty cases  (load {ip.use_only(ip.VISION_MODEL)}s)")
    for c in todo:
        b64, info = image_for(c["image"])
        row = {"question_id": c["case_id"], "kind": c["kind"], "page": None,
               "image_id": Path(c["image"]).name, "image_note": c["image_note"],
               "condition": f"IMG_L1_{mode}_HONESTY", "prompt_id": PROMPT_ID[mode],
               "model": ip.VISION_MODEL, "question": c["question"], "image": info}
        row.update(VISION_ANSWER[mode](b64, c["question"], max_tokens))
        row = complete_row(row, mode)
        row["honesty_pass"], row["honesty_reason"] = honesty_verdict(row, c)
        rows[c["case_id"]] = row
        save(P[key], rows)
        print(f"  {c['case_id']} {'PASS' if row['honesty_pass'] else 'FAIL'} "
              f"{'GATED ' if row.get('gated') else ''}{row['answer'][:90]!r}")
    return rows


def _numbers(text):
    """Numbers in a text, ignoring page references and single digits (step numbers etc.)."""
    text = re.sub(r"\bpages?\s+\d+(-\d+)?", " ", text, flags=re.I)
    return {n.replace(",", "") for n in re.findall(r"\d+(?:[.,]\d+)*", text)} - set("123456789")


def page_text(page):
    """All chunk text the manual has for one PDF page (from the Chroma corpus, read-only)."""
    _, _, _, corpus, _ = ip.rag_query._resources()
    return "\n".join(d for d, m in zip(corpus["docs"], corpus["metas"])
                     if page in [int(p) for p in m["pages"].split(",")])


def honesty_verdict(row, case):
    """PASS if not found / asks to clarify. For a vague question on a manual page, also PASS
    for a grounded summary of the shown page: it names that page and every number it uses
    appears in that page's manual text. Returns (pass, reason)."""
    answer = row["answer"]
    if is_not_found(answer):
        return True, "not found"
    if CLARIFY_RE.search(answer):
        return True, "asks to clarify"
    shown = case.get("shown_page")
    if case["kind"] == "vague_manual_page" and shown:
        if shown not in mentioned_pages(answer):
            return False, f"summary does not name the shown page {shown}"
        extra = sorted(_numbers(answer) - _numbers(page_text(shown)))
        if extra:
            return False, f"numbers not on page {shown}: {extra}"
        return True, f"grounded summary of shown page {shown}"
    return False, "answered an off-topic image"


def measure_swaps(mode, questions, rows, P, max_tokens):
    """Run SWAP_N questions exactly like the bridge (qwen3 resident before each request)."""
    swaps = load(P["swaps"], {})
    if len(swaps.get(mode, [])) >= min(SWAP_N, len(questions)):
        return swaps
    print(f"\n[{mode}s] swap timing, {SWAP_N} questions the bridge way")
    ip.use_only(ip.TEXT_MODEL)                 # bridge idle state; not timed
    runs = []
    for q in questions[:SWAP_N]:
        r = ip.run(image_for(q["processed_path"])[0], q["question"], mode, max_tokens)
        same = r["answer"] == rows[q["question_id"]]["answer"]
        runs.append({"question_id": q["question_id"], "timings": r["timings"],
                     "swap_steps": r["swap_steps"],
                     "same_answer_as_batch": same, "answer": r["answer"]})
        t = r["timings"]
        print(f"  {q['question_id']} swap {t['swap_s']:5.2f}s {r['swap_steps']}  "
              f"model {t['model_s']:6.2f}s  wall {t['wall_s']:6.2f}s  same={same}")
    swaps[mode] = runs
    save(P["swaps"], swaps)
    return swaps


def grade(questions, P, modes):
    """Separate pass, qwen3 loaded once. Raw output files are not touched."""
    grades = load(P["grades"], {"note": UNOFFICIAL})
    client = ip._client()
    loaded = False
    for mode in modes:
        grades.setdefault(mode, {})
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
    return {
        "test": "image_level1", "modes": {},
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
        "prompts": {},
        "code_commit": git_commit(),
        "batching_note": ("Each mode's answers were run as a batch (one model load); temp 0 "
                          "gives the same answers as the per-request bridge flow. Swap cost is "
                          f"measured separately on the first {SWAP_N} questions run exactly like "
                          "the bridge."),
    }


def git_commit():
    """Short HEAD hash, with "-dirty" if the working tree has uncommitted changes."""
    return subprocess.run(["git", "describe", "--always", "--dirty"], cwd=Path(__file__).parent,
                          capture_output=True, text=True).stdout.strip()


def update_meta(meta, modes):
    """Record every mode/prompt that has been run; note commits used by later additions."""
    prompts = {"A": {"describe_system": DESCRIBE_SYSTEM, "mode_a_question": MODE_A_QUESTION,
                     "mode_a_system": NOVA_PROMPT["system_prompt"]},
               "B": {"mode_b_system": NOVA_VISION_PROMPT},
               "B2": {"mode_b2_system": NOVA_VISION_PROMPT_B2, "mode_b2_user": B2_USER}}
    for m in modes:
        if m not in meta["modes"]:
            meta["modes"][m] = MODE_INFO[m]
            meta["prompts"].update(prompts[m])
            if m == "B2":
                meta["settings"]["b2_gate"] = {
                    "min_similarity": ip.B2_MIN_SIMILARITY,
                    "min_similarity_lex": ip.B2_MIN_SIMILARITY_LEX, "min_bm25": ip.B2_MIN_BM25}
                meta["inputs"]["honesty_set"] = str(HONESTY_PATH)
            if meta.get("code_commit") != git_commit():
                meta.setdefault("notes", []).append(
                    f"Mode {m} added {datetime.now():%Y-%m-%d} with code commit {git_commit()}.")
    return meta


def _stats(xs):
    xs = [x for x in xs if x is not None]
    return {"median": round(statistics.median(xs), 2), "mean": round(statistics.mean(xs), 2)} \
        if xs else {"median": None, "mean": None}


def summarize(questions, P):
    grades = load(P["grades"], {})
    swaps = load(P["swaps"], {})
    out = {"grading": UNOFFICIAL}
    for mode in SUMMARY_MODES:
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
        answered = [r for r in rows if not r["not_found"]]
        out[mode] = {
            "n": n,
            "auto_grade": {"correct": c, "partial": p, "wrong": w, "ungraded": g.count(None),
                           "score_pct": round(100 * (c + 0.5 * p) / n, 1)},
            "page_hit_cited": sum(r["page_hit_cited"] for r in rows),
            "no_page_named": [r["question_id"] for r in answered if not r["mentioned_pages"]],
            "wrong_page_named": [r["question_id"] for r in rows
                                 if r["mentioned_pages"] and not r["page_hit_cited"]],
            "page_hit_retrieved": sum(r["page_hit_retrieved"] for r in rows),
            "not_found": [r["question_id"] for r in rows if r["not_found"]],
            "gated": sum(bool(r.get("gated")) for r in rows),
            "list_or_markdown": [r["question_id"] for r in rows if LIST_RE.search(r["answer"])],
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
    honesty = {}
    cases = {c["case_id"]: c for c in load_honesty()}
    for mode in SUMMARY_MODES:
        h = load(P[f"H{mode}"], {})
        if h:
            honesty[mode] = {}
            for cid, r in sorted(h.items()):           # re-scored with the current rule
                ok, why = honesty_verdict(r, cases[cid])
                honesty[mode][cid] = {"kind": r["kind"], "pass": ok, "reason": why,
                                      "prompt_id": r["prompt_id"], "gated": bool(r.get("gated")),
                                      "named_pages": r["mentioned_pages"],
                                      "retrieved_pages": r["retrieved_pages"],
                                      "answer": r["answer"]}
    out["honesty"] = honesty
    save(P["summary"], out)

    modes = [m for m in SUMMARY_MODES if m in out]
    print(f"\n{'=' * 86}\nIMAGE LEVEL 1 SUMMARY   (correctness = {UNOFFICIAL})\n{'=' * 86}")
    print(f"{'':28}" + "".join(f"{'Mode ' + m:>19}" for m in modes))
    def line(label, fn):
        print(f"{label:28}" + "".join(f"{fn(out[m]):>19}" for m in modes))
    line("questions", lambda s: str(s["n"]))
    line("auto-grade C/P/W", lambda s: "{correct}/{partial}/{wrong}".format(**s["auto_grade"]))
    line("auto-grade score", lambda s: f"{s['auto_grade']['score_pct']}%")
    line("gold page named", lambda s: f"{s['page_hit_cited']}/{s['n']}")
    line("no page named", lambda s: str(len(s["no_page_named"])))
    line("wrong page named", lambda s: str(len(s["wrong_page_named"])))
    line("page hit (retrieved@5)", lambda s: f"{s['page_hit_retrieved']}/{s['n']}")
    line("NOT FOUND / gated", lambda s: f"{len(s['not_found'])} / {s['gated']}")
    line("list/markdown answers", lambda s: str(len(s["list_or_markdown"])))
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
          "(A: qwen3->vl->qwen3, B/B2: qwen3->vl + qwen3 reload).")
    for mode, hcases in honesty.items():
        n_pass = sum(c["pass"] for c in hcases.values())
        pids = sorted({c["prompt_id"] for c in hcases.values()})
        print(f"\nHONESTY SET, Mode {mode} (prompt {', '.join(pids)}): {n_pass}/{len(hcases)} pass"
              f"  (not found / clarify; vague page: grounded summary naming the shown page)")
        for cid, c in hcases.items():
            print(f"  {cid} {c['kind']:<18} {'PASS' if c['pass'] else 'FAIL'}"
                  f"{' (gated)' if c['gated'] else ''}  [{c['reason']}]  {c['answer'][:60]!r}")
    print(f"\nSaved {P['summary']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", choices=ALL_MODES, default=list(ALL_MODES))
    ap.add_argument("--honesty-modes", nargs="*", choices=("B", "B2"), default=["B", "B2"],
                    help="vision modes to run the honesty set on (default B and B2)")
    ap.add_argument("--sample", type=int, help="only the first N questions (separate files)")
    ap.add_argument("--grade-only", action="store_true", help="re-grade --modes + summarise")
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
        meta = update_meta(meta, a.modes)
        meta.setdefault("started", datetime.now().isoformat(timespec="seconds"))
        save(P["meta"], meta)
        t0 = time.perf_counter()
        if "A" in a.modes:
            rows = run_mode_a(questions, P, a.max_tokens)
            measure_swaps("A", questions, rows, P, a.max_tokens)
        for mode in ("B", "B2"):
            if mode in a.modes:
                rows = run_vision_mode(mode, questions, P, a.max_tokens)
                if mode == "B2":                  # vision model still loaded: no extra swaps
                    for hm in a.honesty_modes:
                        run_honesty(hm, P, a.max_tokens)
                measure_swaps(mode, questions, rows, P, a.max_tokens)
        meta["finished"] = datetime.now().isoformat(timespec="seconds")
        meta["last_run_wall_s"] = round(time.perf_counter() - t0, 1)
        save(P["meta"], meta)
    else:
        grades = load(P["grades"], {})
        for m in a.modes:                         # fresh grading pass for the given modes only
            grades.pop(m, None)
        save(P["grades"], grades)
    grade(questions, P, a.modes)
    summarize(questions, P)
    ip.use_only(ip.TEXT_MODEL)                # leave qwen3 resident, like the bridge expects


if __name__ == "__main__":
    main()
