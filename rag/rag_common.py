"""Shared config + helpers for the Nova RAG pipeline (SV-CH / SV-F milling machine manual)."""
import html
import os
import re
import zipfile
from pathlib import Path

RAG_DIR = Path(__file__).resolve().parent
CHROMA_DIR = RAG_DIR / "chroma_db"
RESULTS_DIR = RAG_DIR / "results"

# FINAL 20/11 clean manual (not the 19/11 copy in rag_package, not rag_chunks.jsonl)
DEFAULT_MANUAL = Path("/mnt/c/Grad/dataset/universal_milling_machine_en.md")
DEFAULT_COLLECTION = "milling_manual_v20"
# Overrides for comparison runs, e.g. the 19/11 manual into its own collection:
#   RAG_MANUAL=/mnt/c/Grad/dataset/rag_package/manuals/universal_milling_machine_en.md \
#   RAG_COLLECTION=milling_manual_v19 python rag_ingest.py
MANUAL_PATH = Path(os.environ.get("RAG_MANUAL", DEFAULT_MANUAL))
# Read-only: P03 system prompt + user template come from the frozen manifest
MANIFEST_PATH = Path.home() / "benchmark_v2" / "benchmark_manifest.xlsx"
QUESTIONS_PATH = Path("/mnt/c/Grad/dataset/rag_package/questions.csv")

COLLECTION = os.environ.get("RAG_COLLECTION", DEFAULT_COLLECTION)
# results of non-default collections get a filename suffix so runs never overwrite/mix
RESULT_SUFFIX = "" if COLLECTION == DEFAULT_COLLECTION else f"_{COLLECTION}"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:8b"
TEMPERATURE = 0
MAX_TOKENS = 1024
TOP_K = 5

# nomic-embed-text is trained with task prefixes; using them matters for retrieval quality
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token for English technical text)."""
    return max(1, round(len(text) / 4))


def read_sheet(sheet_name: str) -> list[dict]:
    """Read a manifest sheet as a list of {header: value} dicts (stdlib only, read-only)."""
    with zipfile.ZipFile(MANIFEST_PATH) as z:
        wb = z.read("xl/workbook.xml").decode()
        rels = z.read("xl/_rels/workbook.xml.rels").decode()
        rid = re.search(rf'<sheet [^>]*name="{re.escape(sheet_name)}"[^>]*r:id="([^"]+)"', wb).group(1)
        target = re.search(rf'<Relationship [^>]*Id="{rid}"[^>]*Target="([^"]+)"', rels)
        if target is None:  # attribute order may differ
            target = re.search(rf'<Relationship [^>]*Target="([^"]+)"[^>]*Id="{rid}"', rels)
        sheet = z.read("xl/" + target.group(1).lstrip("/").removeprefix("xl/")).decode()

    rows = []
    for row in re.findall(r"<row[^>]*>(.*?)</row>", sheet, re.S):
        cells = {}
        for col, body in re.findall(r'<c r="([A-Z]+)\d+"[^>]*>(.*?)</c>', row, re.S):
            text = "".join(re.findall(r"<t[^>]*>(.*?)</t>", body, re.S))
            if not text:  # numeric / shared-formula cells: <v>7</v>
                text = "".join(re.findall(r"<v>(.*?)</v>", body, re.S))
            cells[col] = html.unescape(text)
        rows.append(cells)
    header = rows[0]
    return [{header[c]: r.get(c, "") for c in header} for r in rows[1:]]


def load_benchmark_questions() -> list[dict]:
    """The 60 frozen benchmark questions (Manifest sheet)."""
    return [r for r in read_sheet("Manifest") if r.get("question_id")]


def load_prompt(prompt_id: str = "P03") -> dict:
    """Read a prompt row from the manifest's Frozen_Prompts sheet."""
    for rec in read_sheet("Frozen_Prompts"):
        if rec.get("prompt_id") == prompt_id:
            assert "{context}" in rec["system_prompt"] or prompt_id != "P03"
            return rec
    raise KeyError(f"{prompt_id} not found in {MANIFEST_PATH}")


def embed(client, texts, prefix):
    resp = client.embed(model=EMBED_MODEL, input=[prefix + t for t in texts])
    return resp["embeddings"]
