"""Compare rag_eval result files and list the misses of the last one.

    python rag_compare.py results/rag_eval_A.json results/rag_eval_B.json [...]
"""
import json
import sys
from pathlib import Path

METRICS = ["page_hit_rate@1", "page_hit_rate@5", "mrr", "trick_refusal_rate",
           "false_refusal_rate_answerable", "avg_latency_s"]


def label(d):
    c = d["config"]
    return f"{c.get('chunking', 'v1')}/{c.get('retrieval', 'vector')}"


def main(paths):
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    names = [label(r) for r in runs]
    w = max(12, *(len(n) for n in names))
    print(f"{'metric':<32}" + "".join(f"{n:>{w + 2}}" for n in names))
    for m in METRICS:
        print(f"{m:<32}" + "".join(f"{r['summary'].get(m, float('nan')):>{w + 2}}" for r in runs))

    base, new = runs[0], runs[-1]
    b = {r["id"]: r for r in base["results"]}
    print(f"\nPer-question page-hit changes ({names[0]} -> {names[-1]}):")
    for r in new["results"]:
        o = b.get(r["id"])
        if o and r["answerable"] == "yes" and o["page_hit"] != r["page_hit"]:
            print(f"  Q{r['id']:02d} p{r['expected_page']}: {o['page_hit']} -> {r['page_hit']}  {r['question']}")

    print(f"\nMisses in {names[-1]} (answerable: page miss; trick: not refused):")
    for r in new["results"]:
        miss = (r["answerable"] == "yes" and r["page_hit"] == "n") or \
               (r["answerable"] == "no" and r["refused"] == "n")
        if miss:
            print(f"  Q{r['id']:02d} [{r['category']}] {r['question']}\n"
                  f"      expected p{r['expected_page']}, got {r['retrieved_pages']}\n"
                  f"      answer:   {r['answer']}\n"
                  f"      expected: {r['expected_answer']}")


if __name__ == "__main__":
    main(sys.argv[1:])
