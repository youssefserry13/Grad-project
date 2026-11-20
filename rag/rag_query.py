"""Answer questions about the milling-machine manual with hybrid retrieval + qwen3:8b.

    from rag_query import ask_rag
    r = ask_rag("What is the coolant tank capacity?")
    r["answer"], r["cited_pages"]

Retrieval modes: "hybrid" (default: BM25 + vector, reciprocal rank fusion), "vector", "bm25".
CLI:  python rag_query.py [--mode hybrid|vector|bm25] "What is the coolant tank capacity?"
"""
import argparse
import re
import time
from functools import lru_cache

import chromadb
import ollama
from rank_bm25 import BM25Okapi

from rag_common import (CHROMA_DIR, COLLECTION, LLM_MODEL, MAX_TOKENS, QUERY_PREFIX,
                        TEMPERATURE, TOP_K, embed, load_prompt)

NOT_FOUND = "NOT FOUND"
MODES = ("hybrid", "vector", "bm25")
RRF_K = 60            # standard RRF constant
CANDIDATES = 20       # per-retriever candidate list fed into RRF

_STOP = set("""a an and are as at be by can do does for from how i if in into is it its may
must of on or should the their there these this to was what when where which who why will
with machine""".split())
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def tokenize(text: str):
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


@lru_cache(maxsize=1)
def _resources():
    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = db.get_collection(COLLECTION)  # raises if rag_ingest.py hasn't been run
    allc = col.get(include=["documents", "metadatas"])
    corpus = {"ids": allc["ids"], "docs": allc["documents"], "metas": allc["metadatas"]}
    bm25 = BM25Okapi([tokenize(d) for d in corpus["docs"]])
    return ollama.Client(), col, bm25, corpus, load_prompt("P03")


def _vector_ranked(question, n):
    client, col, _, _, _ = _resources()
    res = col.query(query_embeddings=embed(client, [question], QUERY_PREFIX), n_results=n,
                    include=["distances"])
    return list(zip(res["ids"][0], res["distances"][0]))


def _bm25_ranked(question, n):
    _, _, bm25, corpus, _ = _resources()
    scores = bm25.get_scores(tokenize(question))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
    return [(corpus["ids"][i], float(scores[i])) for i in order if scores[i] > 0]


def retrieve(question: str, k: int = TOP_K, mode: str = "hybrid"):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    _, _, _, corpus, _ = _resources()
    vec = _vector_ranked(question, max(k, CANDIDATES)) if mode in ("hybrid", "vector") else []
    lex = _bm25_ranked(question, max(k, CANDIDATES)) if mode in ("hybrid", "bm25") else []
    vec_rank = {cid: r for r, (cid, _) in enumerate(vec, 1)}
    lex_rank = {cid: r for r, (cid, _) in enumerate(lex, 1)}
    dist = dict(vec)

    fused = {}
    for ranks in (vec_rank, lex_rank):
        for cid, r in ranks.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + r)
    top = sorted(fused, key=lambda c: -fused[c])[:k]

    idx = {cid: i for i, cid in enumerate(corpus["ids"])}
    hits = []
    for cid in top:
        m = corpus["metas"][idx[cid]]
        hits.append({"id": cid, "text": corpus["docs"][idx[cid]],
                     "pages": [int(p) for p in m["pages"].split(",")],
                     "section": m["section"], "kind": m.get("kind", "text"),
                     "rrf": round(fused[cid], 5), "vec_rank": vec_rank.get(cid),
                     "bm25_rank": lex_rank.get(cid),
                     "distance": round(dist[cid], 4) if cid in dist else None})
    return hits


def _ordered_pages(hits):
    seen = []
    for h in hits:
        for p in h["pages"]:
            if p not in seen:
                seen.append(p)
    return seen


def ask_rag(question: str, k: int = TOP_K, mode: str = "hybrid") -> dict:
    """Retrieve top-k chunks and answer with prompt P03.

    Returns {answer, not_found, cited_pages, retrieved, latency_s, retrieval_s, generation_s}.
    cited_pages = pages of the retrieved context in rank order ([] when the answer is NOT FOUND,
    because P03 asks for a bare answer, the model does not cite pages itself).
    """
    client, _, _, _, p03 = _resources()
    t0 = time.perf_counter()
    hits = retrieve(question, k, mode)
    t1 = time.perf_counter()

    context = "\n\n---\n\n".join(h["text"] for h in hits)
    resp = client.chat(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": p03["system_prompt"].replace("{context}", context)},
                  {"role": "user", "content": p03["user_template"].replace("{question}", question)}],
        think=False,
        keep_alive=-1,
        options={"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
    )
    t2 = time.perf_counter()

    content = resp["message"]["content"]          # untouched model output
    answer = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    not_found = NOT_FOUND in answer.upper()
    retrieved_pages = _ordered_pages(hits)
    return {
        "question": question,
        "mode": mode,
        "answer": NOT_FOUND if not_found else answer,
        "raw_answer": answer,
        "content": content,
        "not_found": not_found,
        "cited_pages": [] if not_found else retrieved_pages,
        "retrieved_pages": retrieved_pages,
        "retrieved": [{k_: h[k_] for k_ in ("id", "pages", "section", "kind", "rrf",
                                            "vec_rank", "bm25_rank", "distance")} for h in hits],
        "done_reason": resp.get("done_reason"),
        "usage": {"prompt_tokens": resp.get("prompt_eval_count"),
                  "completion_tokens": resp.get("eval_count"),
                  "total_duration_ns": resp.get("total_duration")},
        "retrieval_s": round(t1 - t0, 3),
        "generation_s": round(t2 - t1, 3),
        "latency_s": round(t2 - t0, 3),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="hybrid", choices=MODES)
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("question", nargs="*")
    a = ap.parse_args()
    q = " ".join(a.question) or "What is the coolant tank capacity?"
    r = ask_rag(q, a.k, a.mode)
    print(f"Q: {q}\nA: {r['answer']}\nPages: {r['cited_pages']}  "
          f"(retrieved {r['retrieved_pages']}, {a.mode})  {r['latency_s']}s")
