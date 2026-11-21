"""Export RAG answers on the 60 frozen benchmark questions in the SAME format as local_S1.json.

    python rag_export_r1.py [--out /mnt/c/Grad/to_send]

Writes <out>/rag_R1.json (dict keyed by question_id, same record fields as local_S1.json,
condition "R1", prompt P03, image_id null) and <out>/rag_R1_run_meta.json (models, retrieval,
chunking, settings, manifest + manual hashes, hardware, Ollama version).
Uses the default collection (20/11 manual); refuses to run on a RAG_COLLECTION override.
"""
import argparse
import csv
import hashlib
import json
import platform
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from rag_common import (COLLECTION, DEFAULT_COLLECTION, EMBED_MODEL, LLM_MODEL, MANIFEST_PATH,
                        MANUAL_PATH, MAX_TOKENS, QUESTIONS_PATH, TEMPERATURE, TOP_K,
                        load_benchmark_questions, load_prompt)
from rag_ingest import CHUNKING_VERSION, chunk_pages, parse_pages
from rag_query import CANDIDATES, RRF_K, _resources, ask_rag

CONDITION = "R1"


def clean_answer(raw):
    """Same rule as benchmark_qa_local.py: strip <think> blocks; empty if reasoning was cut off."""
    if "<think>" in raw and "</think>" not in raw:
        return ""
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # meta info is best-effort
        return f"unavailable ({e})"


def ollama_model_info(name):
    out = sh(["ollama", "show", name])
    get = lambda k: (re.search(rf"^\s*{k}\s+(.+?)\s*$", out, re.M) or [None, None])[1]
    digest = next((l.split()[1] for l in sh(["ollama", "list"]).splitlines()[1:]
                   if l.split() and l.split()[0] in (name, f"{name}:latest")), None)
    return {"name": name, "digest": digest, "parameters": get("parameters"),
            "quantization": get("quantization"), "context_length": get("context length")}


def hardware():
    gpu = sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    cpu = re.search(r"Model name:\s*(.+)", sh(["lscpu"]))
    mem_kb = int(re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1))
    return {"gpu": gpu, "cpu": cpu.group(1).strip() if cpu else platform.processor(),
            "ram_visible_to_wsl_gb": round(mem_kb / 1024 / 1024, 1),
            "os": f"{platform.system()} {platform.release()} (WSL2 Ubuntu)",
            "processor_for_llm": "GPU (Ollama, 100% GPU per `ollama ps`)"}


RAG_PACKAGE_MD = Path("/mnt/c/Grad/dataset/rag_package/manuals/universal_milling_machine_en.md")


def load_ragpackage_questions():
    """rag_package questions.csv -> R1-style items; ids RP01..RP30 in CSV row order."""
    with open(QUESTIONS_PATH, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return [{"question_id": f"RP{i:02d}", "source": "Milling Machine",
             "page": int(r["page"]) if r["page"].strip().isdigit() else None,
             "question": r["question"], "category": r["category"],
             "answerable": r["answerable"].strip().lower()} for i, r in enumerate(rows, 1)]


def index_check():
    """Re-chunk the manual in memory and compare with the stored chunks (no writes)."""
    corpus = _resources()[3]
    stored = dict(zip(corpus["ids"], corpus["docs"]))
    chunks = chunk_pages(parse_pages(MANUAL_PATH.read_text(encoding="utf-8")))
    same = sum(stored.get(c["id"]) == c["text"] for c in chunks)
    return {"stored_chunks": len(stored), "rechunked_from_manual": len(chunks),
            "identical": same, "index_matches_manual": same == len(stored) == len(chunks)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/c/Grad/to_send")
    ap.add_argument("--set", default="benchmark", choices=("benchmark", "ragpackage"),
                    help="benchmark = 60 manifest questions (rag_R1.json); "
                         "ragpackage = 30 rag_package questions.csv (rag_R1_ragpackage.json)")
    args = ap.parse_args()
    if COLLECTION != DEFAULT_COLLECTION:
        raise SystemExit(f"R1 export must use the default collection ({DEFAULT_COLLECTION}), "
                         f"not {COLLECTION}. Unset RAG_COLLECTION / RAG_MANUAL.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.set == "benchmark":
        questions = load_benchmark_questions()
        assert len(questions) == 60, len(questions)
        stem = "rag_R1"
    else:
        questions = load_ragpackage_questions()
        assert len(questions) == 30, len(questions)
        stem = "rag_R1_ragpackage"
    ask_rag("warm-up")

    records, started = {}, datetime.now(timezone.utc)
    for q in questions:
        r = ask_rag(q["question"], k=TOP_K)
        records[q["question_id"]] = {
            "question_id": q["question_id"],
            "source": q["source"],
            "page": int(q["page"]) if q["page"] not in (None, "") else None,
            "image_id": None,
            "condition": CONDITION,
            "prompt_id": "P03",
            "model": LLM_MODEL,
            "answer_raw": r["content"],
            "answer_clean": clean_answer(r["content"]),
            "latency_seconds": round(r["latency_s"], 2),   # retrieval + generation
            "finish_reason": r["done_reason"],
            "usage": r["usage"],
            "attempts": 1,
            "system_merged": False,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        print(f"{q['question_id']} {r['latency_s']:.2f}s  {records[q['question_id']]['answer_clean'][:70]}")

    (out / f"{stem}.json").write_text(json.dumps(records, ensure_ascii=False, indent=2),
                                       encoding="utf-8")

    lat = [x["latency_seconds"] for x in records.values()]
    p03 = load_prompt("P03")
    meta = {
        CONDITION: {
            "description": "Local RAG pipeline (text retrieval over the manual, no images)",
            "model": LLM_MODEL,
            "llm": ollama_model_info(LLM_MODEL),
            "embedding_model": ollama_model_info(EMBED_MODEL),
            "think": False,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "keep_alive": -1,
            "prompt_id": "P03",
            "prompt_system": p03["system_prompt"],
            "prompt_user_template": p03["user_template"],
            "retrieval": {"type": "hybrid BM25 (rank_bm25 BM25Okapi) + vector (ChromaDB cosine), "
                                  "reciprocal rank fusion",
                          "top_k": TOP_K, "rrf_k": RRF_K, "candidates_per_retriever": CANDIDATES,
                          "embedding_prefixes": "search_document: / search_query:"},
            "chunking": {"version": CHUNKING_VERSION,
                         "rules": "page/section chunks (~600 token max, never mixing sections); "
                                  "table rows as 'Header: value.' sentences (>3-row tables split "
                                  "2 rows/chunk); spec page 11 split by <Subheading>, lines as "
                                  "'Label: value.'; blank/TOC pages dropped",
                         "n_chunks": len(_resources()[3]["ids"]), "collection": COLLECTION},
            "manual": str(MANUAL_PATH),
            "manual_sha256": sha256(MANUAL_PATH),
            "manifest_sha256": sha256(MANIFEST_PATH),
            "latency_note": "latency_seconds = retrieval + generation (model already loaded; "
                            "one warm-up call excluded)",
            "n_questions": len(records),
            "avg_latency_seconds": round(statistics.mean(lat), 3),
            "not_found_count": sum("NOT FOUND" in x["answer_clean"].upper() for x in records.values()),
            "hardware": hardware(),
            "ollama_version": sh(["ollama", "--version"]).replace("ollama version is ", ""),
            "python_packages": {p: version(p) for p in ("ollama", "chromadb", "rank-bm25")},
            "first_run_utc": started.isoformat(),
            "last_run_utc": datetime.now(timezone.utc).isoformat(),
        }
    }
    if args.set == "ragpackage":
        md_pkg = sha256(RAG_PACKAGE_MD)
        meta[CONDITION].update({
            "question_set": "rag_package questions.csv (30: 25 answerable + 5 trick)",
            "questions_file": str(QUESTIONS_PATH),
            "questions_sha256": sha256(QUESTIONS_PATH),
            "question_ids": {qid: {"question": q["question"], "category": q["category"],
                                   "answerable": q["answerable"]}
                             for qid, q in zip(records, questions)},
            "manual_hash_check": {
                "indexed_manual": str(MANUAL_PATH), "indexed_manual_sha256": sha256(MANUAL_PATH),
                "rag_package_manual": str(RAG_PACKAGE_MD), "rag_package_manual_sha256": md_pkg,
                "match": md_pkg == sha256(MANUAL_PATH),
                "note": "The index is the frozen R1 index built from the final 20/11 .md. The "
                        "rag_package .md is the earlier 19/11 version, so the hashes differ.",
                "index_vs_indexed_manual": index_check(),
            },
        })
    (out / f"{stem}_run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    print(f"\nSaved {out / f'{stem}.json'}\n      {out / f'{stem}_run_meta.json'}")


if __name__ == "__main__":
    main()
