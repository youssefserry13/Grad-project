"""Evaluate ask_rag on rag_package/questions.csv (25 answerable + 5 trick).

Per question: answer, retrieved pages, expected page, page hit (y/n), refused, latency.
Summary: page-retrieval hit rate @k (answerable), trick refusal rate, false refusals, avg latency.
Answer correctness is NOT auto-scored here (that's Fatma's evaluate.py); the expected answer
is logged next to ours for review.

    python rag_eval.py [--k 5] [--mode hybrid|vector|bm25]
"""
import argparse
import csv
import json
import statistics
import time
from datetime import datetime

from rag_common import (COLLECTION, EMBED_MODEL, LLM_MODEL, MANUAL_PATH, MAX_TOKENS,
                        QUESTIONS_PATH, RESULT_SUFFIX, RESULTS_DIR, TEMPERATURE, TOP_K)
from rag_ingest import CHUNKING_VERSION
from rag_query import _resources, MODES, ask_rag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--mode", default="hybrid", choices=MODES)
    args = ap.parse_args()

    with open(QUESTIONS_PATH, encoding="utf-8-sig", newline="") as f:
        questions = list(csv.DictReader(f))

    n_chunks = len(_resources()[3]["ids"])

    # warm-up so model load time doesn't land on question 1
    t = time.perf_counter()
    ask_rag("warm-up", k=args.k, mode=args.mode)
    warmup_s = round(time.perf_counter() - t, 2)
    print(f"Warm-up (model load) {warmup_s}s")

    rows = []
    for i, q in enumerate(questions, 1):
        answerable = q["answerable"].strip().lower() == "yes"
        expected = int(q["page"]) if answerable else None
        r = ask_rag(q["question"], k=args.k, mode=args.mode)
        hit = (expected in r["retrieved_pages"]) if answerable else None
        rank = (r["retrieved_pages"].index(expected) + 1) if hit else None
        rows.append({
            "id": i,
            "category": q["category"],
            "answerable": "yes" if answerable else "no",
            "question": q["question"],
            "answer": r["answer"],
            "expected_answer": q["correct answer"],
            "retrieved_pages": r["retrieved_pages"],
            "expected_page": expected if answerable else "N/A",
            "page_hit": ("y" if hit else "n") if answerable else "N/A",
            "hit_rank": rank,
            "refused": "y" if r["not_found"] else "n",
            "top1_distance": r["retrieved"][0]["distance"],
            "top1_rrf": r["retrieved"][0]["rrf"],
            "retrieval_s": r["retrieval_s"],
            "generation_s": r["generation_s"],
            "latency_s": r["latency_s"],
        })
        print(f"[{i:02d}] {rows[-1]['page_hit']:>3} p{rows[-1]['expected_page']!s:>3} "
              f"got {r['retrieved_pages']}  {r['latency_s']:.2f}s  {r['answer'][:70]}")

    ans = [r for r in rows if r["answerable"] == "yes"]
    trick = [r for r in rows if r["answerable"] == "no"]
    lat = [r["latency_s"] for r in rows]
    summary = {
        "n_questions": len(rows),
        "n_answerable": len(ans),
        "n_trick": len(trick),
        f"page_hit_rate@{args.k}": round(sum(r["page_hit"] == "y" for r in ans) / len(ans), 3),
        "page_hit_rate@1": round(sum(r["hit_rank"] == 1 for r in ans) / len(ans), 3),
        "mrr": round(sum(1 / r["hit_rank"] for r in ans if r["hit_rank"]) / len(ans), 3),
        "trick_refusal_rate": round(sum(r["refused"] == "y" for r in trick) / len(trick), 3),
        "false_refusal_rate_answerable": round(sum(r["refused"] == "y" for r in ans) / len(ans), 3),
        "avg_latency_s": round(statistics.mean(lat), 3),
        "median_latency_s": round(statistics.median(lat), 3),
        "avg_retrieval_s": round(statistics.mean(r["retrieval_s"] for r in rows), 3),
        "avg_generation_s": round(statistics.mean(r["generation_s"] for r in rows), 3),
        "warmup_s_excluded": warmup_s,
    }
    config = {"manual": str(MANUAL_PATH), "questions": str(QUESTIONS_PATH),
              "collection": COLLECTION, "embed_model": EMBED_MODEL, "llm": LLM_MODEL,
              "think": False, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
              "top_k": args.k, "retrieval": args.mode, "chunking": CHUNKING_VERSION,
              "n_chunks": n_chunks, "prompt": "P03 (benchmark_manifest.xlsx)"}

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jpath = RESULTS_DIR / f"rag_eval_{stamp}_{args.mode}{RESULT_SUFFIX}.json"
    cpath = RESULTS_DIR / f"rag_eval_{stamp}_{args.mode}{RESULT_SUFFIX}.csv"
    jpath.write_text(json.dumps({"timestamp": stamp, "config": config, "summary": summary,
                                 "results": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(cpath, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({**r, "retrieved_pages": " ".join(map(str, r["retrieved_pages"]))})

    print("\nSummary:")
    for k_, v in summary.items():
        print(f"  {k_}: {v}")
    print(f"\nSaved {jpath}\n      {cpath}")


if __name__ == "__main__":
    main()
