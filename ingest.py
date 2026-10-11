"""Reads PDF / DOCX / TXT / MD / CSV into (page, text) sections and builds section-aligned chunks.

Section text uses "# Heading" lines followed by paragraphs separated by blank lines.
Every chunk starts with its heading, so searches and citations know which section it came from.
"""
import re
from collections import Counter
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

import config

SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".csv"}


# ------------------------------------------------------------------ PDF
def _join_lines(lines):
    out = ""
    for ln in lines:
        if out.endswith("\u00ad"):            # soft hyphen at line end: glue the word back together
            out = out[:-1] + ln
        else:
            out += ("\n" if out else "") + ln
    return re.sub("[\u00ad\ufeff\u200b]", "", out)


def _pdf_page_text(page):
    d = page.get_text("dict")
    blocks, size_chars = [], Counter()
    for b in d["blocks"]:
        if b.get("type") != 0:
            continue
        lines, sizes, bold, chars = [], [], 0, 0
        for l in b["lines"]:
            txt = "".join(s["text"] for s in l["spans"]).strip()
            if not txt:
                continue
            lines.append(txt)
            for s in l["spans"]:
                n = len(s["text"].strip())
                if n == 0:
                    continue
                sizes.append(s["size"]); chars += n
                size_chars[round(s["size"] * 2) / 2] += n
                if (s["flags"] & 16) or "bold" in s["font"].lower():
                    bold += n
        if lines:
            blocks.append(dict(lines=lines, size=max(sizes) if sizes else 0,
                               bold=bold / max(chars, 1)))
    if not blocks:
        return []
    body = size_chars.most_common(1)[0][0] if size_chars else 10
    out, h1, h2, last_h1 = [], "", "", False
    for b in blocks:
        joined = _join_lines([" ".join(b["lines"])] if len(b["lines"]) == 1 else b["lines"]).replace("\n", " ")
        if len(joined) <= 3 and joined.isdigit():          # page number
            continue
        is_h = (len(joined) <= 110 and len(b["lines"]) <= 3
                and not joined.endswith((".", ",", ";", ":"))
                and (b["bold"] >= 0.8 or b["size"] >= body * 1.25))
        if is_h and b["size"] >= body * 1.7:               # page title
            if last_h1:
                h1 = f"{h1} {joined}"; out[-1] = f"# {h1}"
            else:
                h1, h2 = joined, ""; out.append(f"# {h1}")
            last_h1 = True
        elif is_h:
            h2 = joined
            out.append(f"# {h1} > {h2}" if h1 else f"# {h2}")
            last_h1 = False
        else:
            out.append(_join_lines(b["lines"]))
            last_h1 = False
    return out


# ------------------------------------------------------------------ other formats
def _group(text: str, size=3000):
    """Split non-paged text into numbered sections of roughly `size` chars."""
    parts, cur, n = [], "", 1
    for para in re.split(r"\n\s*\n", text):
        if len(cur) + len(para) > size and cur:
            parts.append((n, cur)); n += 1; cur = ""
        cur += para + "\n\n"
    if cur.strip():
        parts.append((n, cur))
    return parts


def extract_sections(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        pages = []
        with fitz.open(path) as doc:
            for i, page in enumerate(doc, start=1):
                pages.append((i, _pdf_page_text(page)))
        # drop running headers/footers: short non-heading blocks repeated on many pages
        norm = lambda u: re.sub(r"\d+", "#", u.lower()).strip()
        counts = Counter(norm(u) for _, units in pages for u in set(units)
                         if not u.startswith("# ") and len(u) < 100)
        limit = max(4, int(0.35 * len(pages))) if len(pages) >= 6 else 10**9
        out = []
        for i, units in pages:
            kept = [u for u in units if u.startswith("# ") or len(u) >= 100 or counts[norm(u)] < limit]
            text = "\n\n".join(kept).strip()
            if text:
                out.append((i, text))
        return out
    if ext == ".docx":
        from docx import Document
        d = Document(str(path))
        lines = []
        for p in d.paragraphs:
            if not p.text.strip():
                continue
            style = (p.style.name or "") if p.style is not None else ""
            lines.append(f"# {p.text.strip()}" if style.startswith("Heading") else p.text)
        for tbl in d.tables:
            for row in tbl.rows:
                lines.append(" | ".join(c.text.strip() for c in row.cells))
        return _group("\n\n".join(lines))
    text = path.read_text(encoding="utf-8", errors="ignore")
    return _group(text)


# ------------------------------------------------------------------ chunking
def chunk_text(text: str, size=None, overlap=None):
    """Plain sliding-window split for a single very long paragraph."""
    size = size or config.CHUNK_CHARS
    overlap = overlap or config.CHUNK_OVERLAP
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) <= size:
        return [text] if len(text) > 30 else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind("\n"), window.rfind(". "))
            if cut > size * 0.5:
                end = start + cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if len(c) > 30]


def _page_chunks(text, size):
    units = [u.strip() for u in re.split(r"\n\s*\n", text) if u.strip()]
    chunks, heading, cur, n = [], "", [], 0

    def wrap(body):
        return f"# {heading}\n\n{body}" if heading else body

    def flush():
        nonlocal cur, n
        if cur:
            chunks.append(wrap("\n\n".join(cur)))
        cur, n = [], 0

    for u in units:
        if u.startswith("# "):
            flush(); heading = u[2:].strip(); continue
        if len(u) > size:
            flush()
            chunks.extend(wrap(p) for p in chunk_text(u, size))
            continue
        if n + len(u) > size:
            flush()
        cur.append(u); n += len(u)
    flush()
    return [c for c in chunks if len(c) > 40]


def make_chunks(sections):
    """-> list of (page, idx, text). Chunks never cross pages, so citations are exact."""
    out, idx = [], 0
    for page, text in sections:
        for c in _page_chunks(text, config.CHUNK_CHARS):
            out.append((page, idx, c)); idx += 1
    return out


def embed_text(doc_name: str, page: int, chunk: str) -> str:
    return f"{doc_name} (page {page})\n{chunk}"