"""Image questions about the milling machine: two pipelines, shared by the bridge and tests.

Mode A (swap):        qwen2.5vl describes the image -> retrieve with question + description
                      -> qwen3:8b answers with the Nova prompt (ask_rag).
Mode B (vision only): retrieve with the question -> qwen2.5vl answers from image + question
                      + top-5 chunks with a Nova-style prompt.

Only one big model fits in 8 GB VRAM, so models are swapped explicitly (keep_alive=0 unload,
then load) and the swap time is measured on its own. qwen3 is put back with keep_alive=-1;
qwen2.5vl only ever gets a short keep_alive (KEEP_ALIVE).

Building blocks (describe / answer_from_description / answer_with_vision) do NOT swap, so a
batch test can load a model once and run many questions. run_mode_a / run_mode_b do the full
bridge sequence including swaps.
"""
import base64
import io
import os
import re
import time

from PIL import Image, ImageOps

import rag_query
from nova_prompt import (DESCRIBE_SYSTEM, DESCRIBE_USER, MODE_A_QUESTION, NOVA_PROMPT,
                         NOVA_VISION_PROMPT, NOT_FOUND_REPLY, answer_pages, is_not_found,
                         label_pages, mentioned_pages)
from rag_common import LLM_MODEL, TEMPERATURE, TOP_K

TEXT_MODEL = LLM_MODEL                 # qwen3:8b
VISION_MODEL = "qwen2.5vl:7b"
BIG_MODELS = (TEXT_MODEL, VISION_MODEL)
MODES = ("A", "B")

MAX_SIDE = int(os.environ.get("IMAGE_MAX_SIDE", 1024))
JPEG_QUALITY = 85
DEFAULT_QUESTION = "What is this and what should I know about it from the manual?"
UNLOAD_TIMEOUT_S = 30
# qwen3 stays pinned (voice answers). qwen2.5vl only gets a short lease: the bridge unloads it
# explicitly after each request, and this timeout frees it anyway if a request dies mid-way.
# (Not 0: batch tests make many vision calls in a row and would reload it every call.)
KEEP_ALIVE = {TEXT_MODEL: -1, VISION_MODEL: os.environ.get("IMAGE_VISION_KEEP_ALIVE", "2m")}


def _client():
    return rag_query._resources()[0]


def _strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _usage(resp):
    return {"prompt_tokens": resp.get("prompt_eval_count"),
            "completion_tokens": resp.get("eval_count"),
            "total_duration_ns": resp.get("total_duration")}


# ─── Preprocessing ───────────────────────────────────────────────────────────
class BadImage(ValueError):
    pass


def preprocess(image_bytes, max_side=MAX_SIDE, quality=JPEG_QUALITY):
    """Longest side -> max_side px (never upscaled), re-encode JPEG at `quality`, in memory.
    Returns (base64 JPEG, info dict). Raises BadImage for anything that isn't JPEG/PNG."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        fmt = img.format
        img.load()
    except Exception as e:
        raise BadImage(f"cannot read image: {e}") from e
    if fmt not in ("JPEG", "PNG"):
        raise BadImage(f"expected JPEG or PNG, got {fmt}")
    img = ImageOps.exif_transpose(img).convert("RGB")
    orig = img.size
    if max(orig) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    info = {"format": fmt, "orig_size": list(orig), "proc_size": list(img.size),
            "proc_kb": round(buf.tell() / 1024, 1)}
    return base64.b64encode(buf.getvalue()).decode(), info


# ─── VRAM swapping ───────────────────────────────────────────────────────────
def loaded_models():
    return {getattr(m, "model", None) or m.get("model") for m in _client().ps().models}


def use_only(model, keep_alive=None):
    """Make `model` the only big model in VRAM. Returns seconds spent (0 if nothing to do)."""
    keep_alive = KEEP_ALIVE[model] if keep_alive is None else keep_alive
    client = _client()
    t0 = time.perf_counter()
    loaded = loaded_models()
    for other in BIG_MODELS:
        if other != model and other in loaded:
            client.generate(model=other, prompt="", keep_alive=0)
            deadline = time.monotonic() + UNLOAD_TIMEOUT_S
            while other in loaded_models():           # wait until the VRAM is really free
                if time.monotonic() > deadline:
                    raise RuntimeError(f"{other} did not unload within {UNLOAD_TIMEOUT_S}s")
                time.sleep(0.1)
    if model not in loaded:
        client.generate(model=model, prompt="", keep_alive=keep_alive)  # empty prompt = load only
    return round(time.perf_counter() - t0, 3)


# ─── Building blocks (no swapping) ───────────────────────────────────────────
def describe(image_b64, question, max_tokens):
    """Mode A step 1: qwen2.5vl describes the image."""
    t0 = time.perf_counter()
    resp = _client().chat(
        model=VISION_MODEL, keep_alive=KEEP_ALIVE[VISION_MODEL],
        messages=[{"role": "system", "content": DESCRIBE_SYSTEM},
                  {"role": "user", "content": DESCRIBE_USER.format(question=question),
                   "images": [image_b64]}],
        options={"temperature": TEMPERATURE, "num_predict": max_tokens})
    return {"description": _strip_think(resp["message"]["content"]),
            "description_raw": resp["message"]["content"],
            "description_usage": _usage(resp), "description_done_reason": resp.get("done_reason"),
            "vision_s": round(time.perf_counter() - t0, 3)}


def answer_from_description(question, description, max_tokens):
    """Mode A step 2: retrieve with question + description, answer with qwen3 (Nova prompt)."""
    t0 = time.perf_counter()
    hits, scores = rag_query.retrieve_scored(f"{question}\n{description}", TOP_K)
    rag_s = round(time.perf_counter() - t0, 3)
    r = rag_query.ask_rag(MODE_A_QUESTION.format(question=question, description=description),
                          prompt=NOVA_PROMPT, hits=label_pages(hits), max_tokens=max_tokens)
    return {"answer_raw": r["content"], "answer": r["answer"], "not_found": r["not_found"],
            "retrieved_pages": r["retrieved_pages"], "retrieved": r["retrieved"],
            "scores": scores, "usage": r["usage"], "done_reason": r["done_reason"],
            "rag_s": rag_s, "answer_s": r["generation_s"]}


def answer_with_vision(image_b64, question, max_tokens):
    """Mode B: retrieve with the question only, qwen2.5vl answers from image + chunks."""
    t0 = time.perf_counter()
    hits, scores = rag_query.retrieve_scored(question, TOP_K)
    t1 = time.perf_counter()
    hits = label_pages(hits)
    context = "\n\n---\n\n".join(h["text"] for h in hits)
    resp = _client().chat(
        model=VISION_MODEL, keep_alive=KEEP_ALIVE[VISION_MODEL],
        messages=[{"role": "system", "content": NOVA_VISION_PROMPT.replace("{context}", context)},
                  {"role": "user", "content": question, "images": [image_b64]}],
        options={"temperature": TEMPERATURE, "num_predict": max_tokens})
    t2 = time.perf_counter()
    content = resp["message"]["content"]
    answer = _strip_think(content)
    not_found = is_not_found(answer)
    retrieved_pages = rag_query._ordered_pages(hits)
    return {"answer_raw": content, "answer": NOT_FOUND_REPLY if not_found else answer,
            "not_found": not_found, "retrieved_pages": retrieved_pages,
            "retrieved": [{k: h[k] for k in ("id", "pages", "section", "rrf")} for h in hits],
            "scores": scores, "usage": _usage(resp), "done_reason": resp.get("done_reason"),
            "rag_s": round(t1 - t0, 3), "vision_s": round(t2 - t1, 3)}


def finish(out, mode):
    """Add pages + a uniform timings block. vision_s = qwen2.5vl call (A: description,
    B: the answer); answer_s = qwen3 answer (A only). model_s = everything except swaps."""
    out["mode"] = mode
    out["mentioned_pages"] = mentioned_pages(out["answer"])
    out["pages"] = answer_pages(out["answer"], out["retrieved_pages"], out["not_found"])
    t = {k: out.pop(k, 0.0) for k in ("vision_s", "rag_s", "answer_s", "swap_s")}
    t["model_s"] = round(t["vision_s"] + t["rag_s"] + t["answer_s"], 3)
    t["total_s"] = round(t["model_s"] + t["swap_s"], 3)
    out["timings"] = t
    return out


# ─── Full bridge sequences (with swaps) ──────────────────────────────────────
def run_mode_a(image_b64, question, max_tokens):
    steps = {"to_vision_s": use_only(VISION_MODEL)}
    d = describe(image_b64, question, max_tokens)
    steps["to_text_s"] = use_only(TEXT_MODEL)   # qwen3 back, pinned
    a = answer_from_description(question, d["description"], max_tokens)
    return finish({**d, **a, "swap_s": round(sum(steps.values()), 3), "swap_steps": steps}, "A")


def run_mode_b(image_b64, question, max_tokens, restore_text=True):
    steps = {"to_vision_s": use_only(VISION_MODEL)}
    b = answer_with_vision(image_b64, question, max_tokens)
    if restore_text:                                     # keep voice answers fast afterwards
        steps["to_text_s"] = use_only(TEXT_MODEL)
    return finish({**b, "swap_s": round(sum(steps.values()), 3), "swap_steps": steps}, "B")


def run(image_b64, question, mode, max_tokens):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    t0 = time.perf_counter()
    out = (run_mode_a if mode == "A" else run_mode_b)(image_b64, question, max_tokens)
    out["timings"]["wall_s"] = round(time.perf_counter() - t0, 3)
    return out
