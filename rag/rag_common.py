"""Shared config + helpers for the Nova RAG pipeline (SV-CH / SV-F milling machine manual)."""
import html
import re
import zipfile
from pathlib import Path

RAG_DIR = Path(__file__).resolve().parent
CHROMA_DIR = RAG_DIR / "chroma_db"
RESULTS_DIR = RAG_DIR / "results"

# FINAL 20/11 clean manual (not the 19/11 copy in rag_package, not rag_chunks.jsonl)
MANUAL_PATH = Path("/mnt/c/Grad/dataset/universal_milling_machine_en.md")
# Read-only: P03 system prompt + user template come from the frozen manifest
MANIFEST_PATH = Path.home() / "benchmark_v2" / "benchmark_manifest.xlsx"
QUESTIONS_PATH = Path("/mnt/c/Grad/dataset/rag_package/questions.csv")

COLLECTION = "milling_manual_v20"
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


def load_prompt(prompt_id: str = "P03") -> dict:
    """Read a prompt row from the manifest's Frozen_Prompts sheet (stdlib only, read-only)."""
    with zipfile.ZipFile(MANIFEST_PATH) as z:
        wb = z.read("xl/workbook.xml").decode()
        rels = z.read("xl/_rels/workbook.xml.rels").decode()
        rid = re.search(r'<sheet [^>]*name="Frozen_Prompts"[^>]*r:id="([^"]+)"', wb).group(1)
        target = re.search(rf'<Relationship [^>]*Id="{rid}"[^>]*Target="([^"]+)"', rels)
        if target is None:  # attribute order may differ
            target = re.search(rf'<Relationship [^>]*Target="([^"]+)"[^>]*Id="{rid}"', rels)
        sheet = z.read("xl/" + target.group(1).lstrip("/").removeprefix("xl/")).decode()

    rows = []
    for row in re.findall(r"<row[^>]*>(.*?)</row>", sheet, re.S):
        cells = {}
        for col, body in re.findall(r'<c r="([A-Z]+)\d+"[^>]*>(.*?)</c>', row, re.S):
            text = "".join(re.findall(r"<t[^>]*>(.*?)</t>", body, re.S))
            cells[col] = html.unescape(text)
        rows.append(cells)
    header = rows[0]
    for r in rows[1:]:
        rec = {header[c]: r.get(c, "") for c in header}
        if rec.get("prompt_id") == prompt_id:
            assert "{context}" in rec["system_prompt"] or prompt_id != "P03"
            return rec
    raise KeyError(f"{prompt_id} not found in {MANIFEST_PATH}")


def embed(client, texts, prefix):
    resp = client.embed(model=EMBED_MODEL, input=[prefix + t for t in texts])
    return resp["embeddings"]
