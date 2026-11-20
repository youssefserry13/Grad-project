"""Run ask_rag() on the 60 frozen benchmark questions and roughly score it against S1/S2/C1.

    python rag_benchmark60.py                    # run RAG + judge all four systems
    python rag_benchmark60.py --rejudge FILE     # re-score an existing rag_benchmark60 JSON

Inputs (read-only): ~/benchmark_v2/benchmark_manifest.xlsx, ~/benchmark_v2/results/raw_outputs/
local_S1.json + local_S2.json, /mnt/c/Grad/dataset/v2_groq_cloud_raw.json (condition C1).

NOTE on comparability: S1/S2/C1 were given the gold page image (P01); RAG must find the page
itself in the whole manual (P03). The correct/partial/wrong score is a ROUGH automatic grade
by a local LLM judge (qwen3:8b, temp 0), the same judge for all four systems. It is not the
official scoring (Fatma's evaluate.py).
"""
import argparse
import csv
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

import ollama

from rag_common import (COLLECTION, LLM_MODEL, MANIFEST_PATH, MANUAL_PATH, RESULT_SUFFIX,
                        RESULTS_DIR, TOP_K, load_benchmark_questions)
from rag_ingest import CHUNKING_VERSION
from rag_query import ask_rag

S_DIR = Path.home() / "benchmark_v2" / "results" / "raw_outputs"
CLOUD_PATH = Path("/mnt/c/Grad/dataset/v2_groq_cloud_raw.json")
SYSTEMS = ["RAG", "S1", "S2", "C1"]

JUDGE_SYSTEM = """You grade answers to questions about a milling machine manual.
Compare the CANDIDATE answer with the REFERENCE answer and reply with exactly one word:
CORRECT - contains the key facts of the reference (wording may differ; extra harmless detail is fine).
PARTIAL - some key facts are right but others are missing, vague, or slightly wrong.
WRONG - incorrect, contradicts the reference, answers something else, or says NOT FOUND / I don't know."""
JUDGE_USER = "QUESTION: {q}\nREFERENCE: {ref}\nCANDIDATE: {cand}\n\nGrade (CORRECT, PARTIAL or WRONG):"


def load_other_systems():
    other = {}
    for s in ("S1", "S2"):
        d = json.loads((S_DIR / f"local_{s}.json").read_text(encoding="utf-8"))
        other[s] = {qid: (r.get("answer_clean") or r.get("answer_raw") or "") for qid, r in d.items()}
    cloud = json.loads(CLOUD_PATH.read_text(encoding="utf-8"))
    other["C1"] = {r["question_id"]: r.get("answer") or "" for r in cloud if r["condition"] == "C1"}
    return other


def judge(client, q, ref, cand):
    if not cand.strip() or "NOT FOUND" in cand.upper():
        return "WRONG"
    resp = client.chat(model=LLM_MODEL, think=False, keep_alive=-1,
                       options={"temperature": 0, "num_predict": 8},
                       messages=[{"role": "system", "content": JUDGE_SYSTEM},
                                 {"role": "user", "content": JUDGE_USER.format(q=q, ref=ref, cand=cand)}])
    m = re.search(r"CORRECT|PARTIAL|WRONG", resp["message"]["content"].upper())
    return m.group(0) if m else "WRONG"


def run_rag(questions):
    ask_rag("warm-up")
    rows = []
    for q in questions:
        page = int(q["pdf_page"] or q["page"])
        r = ask_rag(q["question"], k=TOP_K)
        rank = r["retrieved_pages"].index(page) + 1 if page in r["retrieved_pages"] else None
        rows.append({
            "question_id": q["question_id"],
            "question": q["question"],
            "question_type": q.get("question_type", ""),
            "difficulty": q.get("difficulty", ""),
            "expected_answer": q["ground_truth"],
            "expected_page": page,
            "rag_answer": r["answer"],
            "retrieved_pages": r["retrieved_pages"],
            "page_hit": "y" if rank else "n",
            "hit_rank": rank,
            "not_found": "y" if r["not_found"] else "n",
            "latency_s": r["latency_s"],
        })
        print(f"{q['question_id']} {rows[-1]['page_hit']} p{page:<3} got {r['retrieved_pages']} "
              f"{r['latency_s']:.2f}s  {r['answer'][:70]}")
    return rows


def score_all(rows, other):
    client = ollama.Client()
    for i, r in enumerate(rows, 1):
        r["grade_RAG"] = judge(client, r["question"], r["expected_answer"], r["rag_answer"])
        for s in ("S1", "S2", "C1"):
            r[f"answer_{s}"] = other[s].get(r["question_id"], "")
            r[f"grade_{s}"] = judge(client, r["question"], r["expected_answer"], r[f"answer_{s}"])
        if i % 10 == 0:
            print(f"  judged {i}/{len(rows)}")


def summarize(rows):
    n = len(rows)
    s = {
        "n_questions": n,
        "page_hit_rate@1": round(sum(r["hit_rank"] == 1 for r in rows) / n, 3),
        f"page_hit_rate@{TOP_K}": round(sum(r["page_hit"] == "y" for r in rows) / n, 3),
        "mrr": round(sum(1 / r["hit_rank"] for r in rows if r["hit_rank"]) / n, 3),
        "not_found_count": sum(r["not_found"] == "y" for r in rows),
        "avg_latency_s": round(statistics.mean(r["latency_s"] for r in rows), 3),
        "grades": {},
    }
    for sys_ in SYSTEMS:
        g = [r[f"grade_{sys_}"] for r in rows]
        c, p, w = g.count("CORRECT"), g.count("PARTIAL"), g.count("WRONG")
        s["grades"][sys_] = {"correct": c, "partial": p, "wrong": w,
                             "score_pct": round(100 * (c + 0.5 * p) / n, 1)}
    return s


def save(rows, summary, stamp):
    RESULTS_DIR.mkdir(exist_ok=True)
    jpath = RESULTS_DIR / f"rag_benchmark60_{stamp}{RESULT_SUFFIX}.json"
    cpath = RESULTS_DIR / f"rag_benchmark60_{stamp}{RESULT_SUFFIX}.csv"
    config = {"manifest": str(MANIFEST_PATH), "manual": str(MANUAL_PATH), "collection": COLLECTION,
              "chunking": CHUNKING_VERSION, "llm": LLM_MODEL, "retrieval": "hybrid",
              "top_k": TOP_K, "prompt": "P03", "judge": f"{LLM_MODEL} temp 0 (rough, not official)",
              "compared": {"S1": str(S_DIR / "local_S1.json"), "S2": str(S_DIR / "local_S2.json"),
                           "C1": f"{CLOUD_PATH} (condition C1)"},
              "caveat": "S1/S2/C1 were given the gold page image; RAG retrieves from the whole manual."}
    jpath.write_text(json.dumps({"timestamp": stamp, "config": config, "summary": summary,
                                 "results": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(cpath, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({**r, "retrieved_pages": " ".join(map(str, r["retrieved_pages"]))})
    return jpath, cpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rejudge", help="re-score an existing rag_benchmark60 JSON (no RAG rerun)")
    args = ap.parse_args()

    other = load_other_systems()
    if args.rejudge:
        rows = json.loads(Path(args.rejudge).read_text(encoding="utf-8"))["results"]
    else:
        questions = load_benchmark_questions()
        assert len(questions) == 60, len(questions)
        rows = run_rag(questions)
    t = time.perf_counter()
    score_all(rows, other)
    print(f"Judging took {time.perf_counter() - t:.0f}s")

    summary = summarize(rows)
    jpath, cpath = save(rows, summary, datetime.now().strftime("%Y%m%d_%H%M%S"))
    print(json.dumps(summary, indent=2))
    print(f"Saved {jpath}\n      {cpath}")


if __name__ == "__main__":
    main()
