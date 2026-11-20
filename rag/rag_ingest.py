"""Chunk the final milling-machine manual and store it in ChromaDB.

Usage:
    python rag_ingest.py --stats-only   # chunk + print stats, no embedding
    python rag_ingest.py                # chunk, embed (nomic-embed-text), rebuild collection

Chunking rules (v3):
  * Split on the [Page N - Section] markers. A chunk never mixes sections.
  * Spec page(s) in SUBHEADING_SPLIT_PAGES (p.11) are split into one chunk per <Subheading>
    group, each prefixed "[Page 11 - 3-1 Machine specifications] <Subheading>", and every
    spec line is written "Label: value." ("(60Hz) ..." continuation lines get their label back).
  * Every markdown table row is converted to a self-contained sentence
    "Header: value. Header: value." (see table_to_rows), with the table caption
    (preceding <Subheading>, e.g. "<(OPTION)_NT No.40>") kept. Tables with more than
    TABLE_MAX_ROWS rows are split into groups of TABLE_ROWS_PER_CHUNK rows, each prefixed
    with page + section (+ caption); smaller tables stay whole inside their text chunk.
  * Remaining text: consecutive pages with the SAME section heading may be merged up to
    ~MAX tokens; oversized pages are split on paragraph/line boundaries with a small overlap.
  * Blank pages and table-of-contents pages are dropped (they only add page-number noise).
"""
import argparse
import json
import re
import statistics

from rag_common import (CHROMA_DIR, COLLECTION, DEFAULT_COLLECTION, DEFAULT_MANUAL,
                        DOC_PREFIX, MANUAL_PATH, approx_tokens, embed)

CHUNKING_VERSION = "v3"
MIN_TOKENS = 400
MAX_TOKENS = 600
OVERLAP_MAX_TOKENS = 80
SUBHEADING_SPLIT_PAGES = {11}
TABLE_MAX_ROWS = 3          # tables with more data rows than this get split
TABLE_ROWS_PER_CHUNK = 2
PAGE_RE = re.compile(r"^\[Page (\d+) - (.*)\]\s*$")
SUBHEADING_RE = re.compile(r"^<[^<>]+>\s*$")


def parse_pages(text):
    pages, cur = [], None
    for line in text.splitlines():
        m = PAGE_RE.match(line)
        if m:
            cur = {"page": int(m.group(1)), "section": m.group(2).strip(), "lines": []}
            pages.append(cur)
        elif cur is not None:
            cur["lines"].append(line)
    for p in pages:
        p["body"] = "\n".join(p.pop("lines")).strip()
    return pages


def is_toc(page):
    lines = [l for l in page["body"].splitlines() if l.strip()]
    dotted = sum(bool(re.search(r"-{5,}\s*\d+\s*$", l)) for l in lines)
    return page["section"] == "Contents" and (not lines or dotted >= len(lines) / 3)


def blocks_of(body):
    """Split a page body into blocks: whole tables, or paragraphs (split by lines if too big)."""
    blocks, buf, in_table = [], [], False

    def flush():
        if buf:
            blocks.append({"text": "\n".join(buf).strip(), "table": in_table})
            buf.clear()

    for line in body.splitlines():
        is_tbl = line.lstrip().startswith("|")
        if is_tbl != in_table:
            flush()
            in_table = is_tbl
        if not is_tbl and not line.strip():
            flush()
            continue
        buf.append(line)
    flush()

    out = []
    for b in blocks:
        if b["text"] and not b["table"] and approx_tokens(b["text"]) > MAX_TOKENS:
            piece = []
            for line in b["text"].splitlines():
                if piece and approx_tokens("\n".join(piece + [line])) > MAX_TOKENS // 2:
                    out.append({"text": "\n".join(piece), "table": False})
                    piece = []
                piece.append(line)
            if piece:
                out.append({"text": "\n".join(piece), "table": False})
        elif b["text"]:
            out.append(b)
    return out


# ---------- tables -> "Header: value." row text ----------

def _cells(row):
    return [c.strip() for c in row.strip().strip("|").split("|")]


def _sentence(pairs, label=""):
    body = " ".join(f"{k}: {v}" + ("" if v.endswith(".") else ".") for k, v in pairs)
    return f"{label} — {body}" if label else body


def table_to_rows(table_text):
    """Convert a markdown table into one self-contained sentence per data row.

    * Header names = last non-empty header cell per column (a mostly-empty first header row,
      e.g. p.70, is merged with the next row).
    * Columns with no header name are row-label columns (filled down when empty);
      a numeric label becomes "No. N" (pp.82, 87).
    * Bare-number headers become "<first header> N" (p.51 "Step 6") or "No. N" (p.15).
    Returns None if the text is not a markdown table.
    """
    lines = [l for l in table_text.splitlines() if l.strip()]
    sep = next((i for i, l in enumerate(lines) if re.fullmatch(r"\|?[\s|:-]+\|?", l.strip())), None)
    if sep is None:
        return None
    header_rows, rows = [_cells(l) for l in lines[:sep]], [_cells(l) for l in lines[sep + 1:]]
    if rows and sum(not c for c in header_rows[0]) >= len(header_rows[0]) / 2:
        header_rows, rows = header_rows + rows[:1], rows[1:]
    ncol = max(len(r) for r in header_rows + rows)
    names = []
    for c in range(ncol):
        vals = [h[c] for h in header_rows if c < len(h) and h[c]]
        names.append(vals[-1] if vals else "")

    def is_num(s):
        return re.fullmatch(r"-?\s*\d+", s or "") is not None

    # p.51 style: "Step | 1 | 2 ..." -> first column is the row label, others "Step N"
    if names[0] and len(names) > 1 and all(is_num(n) for n in names[1:]):
        prefix, names = names[0], [""] + names[1:]
    else:
        prefix = "No."
    names = [f"{prefix} {re.sub(r'[^0-9]', '', n)}" if is_num(n) else n for n in names]
    label_cols = [c for c in range(ncol) if not names[c]]

    out, last = [], {}
    for r in rows:
        r = r + [""] * (ncol - len(r))
        labels = []
        for c in label_cols:
            v = r[c] or (last.get(c, "") if c == label_cols[0] else "")
            last[c] = v
            if v:
                labels.append(f"No. {v}" if v.isdigit() else v)
        pairs = [(names[c], r[c]) for c in range(ncol) if c not in label_cols and r[c]]
        if pairs:
            out.append(_sentence(pairs, " — ".join(labels)))
    return out


SPEC_SPLIT_RE = re.compile(r"^(.*?[A-Za-z)\]:])\s+([\d~\[+-].*)$")  # value starts at first number
FREQ_RE = re.compile(r"^\((\d+\s*Hz)\)\s*(.*)$")


def spec_lines_to_rows(lines):
    """p.11 spec lines -> "Label: value." Continuation lines get their label back:
    "(60Hz) 3000 mm/min" -> "Longitudinal rapid feed (60Hz): 3000 mm/min";
    "(0.0014, ... inch/rev.)" is appended to the previous value."""
    rows = []  # [label, value]
    for line in (l.strip() for l in lines):
        if not line:
            continue
        m = FREQ_RE.match(line)
        if m and rows:
            base = re.sub(r"\s*\(\d+\s*Hz\)\s*$", "", rows[-1][0])
            rows.append([f"{base} ({m.group(1)})", m.group(2)])
        elif line.startswith("(") and rows:
            rows[-1][1] += " " + line
        else:
            m = SPEC_SPLIT_RE.match(line)
            rows.append([m.group(1).rstrip(":"), m.group(2)] if m else [line, ""])
    return [f"{k}: {v}." if v else k for k, v in rows]


def table_chunks(rows, page, section, caption):
    """Big converted table -> chunks of TABLE_ROWS_PER_CHUNK row sentences, title repeated."""
    out = []
    for i in range(0, len(rows), TABLE_ROWS_PER_CHUNK):
        grp = rows[i:i + TABLE_ROWS_PER_CHUNK]
        label = (f"row {i + 1}" if len(grp) == 1 else f"rows {i + 1}-{i + len(grp)}") + f" of {len(rows)}"
        cap = f" {caption}" if caption else ""
        text = f"[Page {page} - {section}]{cap} (table, {label})\n" + "\n".join(grp)
        out.append(_meta_chunk(text, [page], section, "table_rows", has_table=True, subheading=caption))
    return out


def split_subheadings(page):
    """One chunk per <Subheading> group (text before the first subheading is the page title)."""
    groups, cur = [], None
    for line in page["body"].splitlines():
        if SUBHEADING_RE.match(line):
            cur = {"sub": line.strip(), "lines": []}
            groups.append(cur)
        elif cur is not None and line.strip():
            cur["lines"].append(line)
    return [_meta_chunk(f"[Page {page['page']} - {page['section']}] {g['sub']}\n"
                        + "\n".join(spec_lines_to_rows(g["lines"])),
                        [page["page"]], page["section"], "spec_group", subheading=g["sub"])
            for g in groups if g["lines"]]


# ---------- main chunker ----------

def chunk_pages(pages):
    pages = [p for p in pages if p["body"] and not is_toc(p)]
    structured, text_pages = [], []
    for p in pages:
        if p["page"] in SUBHEADING_SPLIT_PAGES:
            structured += split_subheadings(p)
            continue
        blocks, caption = [], ""
        for b in blocks_of(p["body"]):
            if not b["table"]:
                subs = [l.strip() for l in b["text"].splitlines() if SUBHEADING_RE.match(l)]
                caption = subs[-1] if subs else caption
                blocks.append(b)
                continue
            rows = table_to_rows(b["text"])
            if rows is None:                       # not a real table: keep as text
                blocks.append({**b, "table": False})
            elif len(rows) > TABLE_MAX_ROWS:
                structured += table_chunks(rows, p["page"], p["section"], caption)
            else:                                  # small table: stays whole, as row text
                cap = caption if caption and not (blocks and caption in blocks[-1]["text"]) else ""
                blocks.append({"text": (cap + "\n" if cap else "") + "\n".join(rows), "table": True})
        if blocks:
            text_pages.append({**p, "blocks": blocks})

    # group consecutive pages that share a section heading, then pack blocks
    groups = []
    for p in text_pages:
        if groups and groups[-1][-1]["section"] == p["section"] and groups[-1][-1]["page"] + 1 == p["page"]:
            groups[-1].append(p)
        else:
            groups.append([p])

    chunks = []
    for group in groups:
        section = group[0]["section"]
        items = [(p["page"], b) for p in group for b in p["blocks"]]
        cur, cur_tok = [], 0
        for page, b in items:
            t = approx_tokens(b["text"])
            if cur and cur_tok + t > MAX_TOKENS:
                chunks.append(_make_chunk(cur, section))
                # small overlap: carry the last short non-table block, if from the same page
                last_page, last = cur[-1]
                carry = [] if (last["table"] or approx_tokens(last["text"]) > OVERLAP_MAX_TOKENS
                               or b["table"] or last_page != page) else [cur[-1]]
                cur = carry
                cur_tok = sum(approx_tokens(x["text"]) for _, x in cur)
            cur.append((page, b))
            cur_tok += t
        if cur:
            chunks.append(_make_chunk(cur, section))

    chunks += structured
    chunks.sort(key=lambda c: c["metadata"]["page"])  # stable: keeps order within a page
    for i, c in enumerate(chunks):
        c["id"] = f"chunk-{i:03d}"
    return chunks


def _meta_chunk(text, pages, section, kind, has_table=False, subheading=""):
    return {
        "text": text,
        "metadata": {
            "page": pages[0],                      # first page (int, filterable)
            "pages": ",".join(map(str, pages)),    # all pages covered
            "section": section,
            "kind": kind,                          # text | table_rows | spec_group
            "subheading": subheading,
            "has_table": has_table,
            "approx_tokens": approx_tokens(text),
        },
    }


def _make_chunk(items, section):
    pages = sorted({p for p, _ in items})
    page_label = str(pages[0]) if len(pages) == 1 else f"{pages[0]}-{pages[-1]}"
    body = "\n\n".join(b["text"] for _, b in items)
    return _meta_chunk(f"[Page {page_label} - {section}]\n{body}", pages, section, "text",
                       has_table=any(b["table"] for _, b in items))


def print_stats(pages, chunks):
    toks = [c["metadata"]["approx_tokens"] for c in chunks]
    kept_pages = sorted({int(p) for c in chunks for p in c["metadata"]["pages"].split(",")})
    kinds = {}
    for c in chunks:
        kinds.setdefault(c["metadata"]["kind"], []).append(c["metadata"]["approx_tokens"])
    print(f"Source: {MANUAL_PATH}  (chunking {CHUNKING_VERSION})")
    print(f"Pages in file: {len(pages)}   pages kept in chunks: {len(kept_pages)}")
    print(f"Dropped pages (blank/TOC): {sorted(set(p['page'] for p in pages) - set(kept_pages))}")
    print(f"Chunks: {len(chunks)}")
    print(f"Tokens/chunk (~chars/4): avg {statistics.mean(toks):.0f}, median {statistics.median(toks):.0f}, "
          f"min {min(toks)}, max {max(toks)}, total {sum(toks)}")
    for k, v in kinds.items():
        print(f"  {k:<11} {len(v):3d} chunks, avg {statistics.mean(v):.0f} tokens")
    print(f"Multi-page chunks: {sum(',' in c['metadata']['pages'] for c in chunks)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-only", action="store_true", help="chunk and print stats, no embedding")
    ap.add_argument("--dump", help="write chunks to this JSONL path for inspection")
    args = ap.parse_args()

    pages = parse_pages(MANUAL_PATH.read_text(encoding="utf-8"))
    chunks = chunk_pages(pages)
    print_stats(pages, chunks)
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"Dumped chunks to {args.dump}")
    if args.stats_only:
        return

    if COLLECTION == DEFAULT_COLLECTION and MANUAL_PATH.resolve() != DEFAULT_MANUAL.resolve():
        raise SystemExit(f"Refusing to rebuild '{COLLECTION}' from {MANUAL_PATH}: "
                         "set RAG_COLLECTION to a new name for a different manual.")

    import chromadb
    import ollama

    client = ollama.Client()
    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        db.delete_collection(COLLECTION)
    except Exception:
        pass
    col = db.create_collection(COLLECTION, metadata={"hnsw:space": "cosine",
                                                      "source": MANUAL_PATH.name,
                                                      "chunking": CHUNKING_VERSION})
    B = 16
    for i in range(0, len(chunks), B):
        batch = chunks[i:i + B]
        col.add(ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
                embeddings=embed(client, [c["text"] for c in batch], DOC_PREFIX))
        print(f"  embedded {min(i + B, len(chunks))}/{len(chunks)}")
    print(f"Stored {col.count()} chunks in {CHROMA_DIR} (collection '{COLLECTION}')")


if __name__ == "__main__":
    main()
