"""Hybrid retrieval, query expansion, re-ranking, strict grounded generation, fact check."""
import re
from dataclasses import dataclass

import numpy as np
import ollama
from rank_bm25 import BM25Okapi

import config
import engine

client = ollama.Client(host=config.OLLAMA_HOST)
NOT_FOUND = "I couldn't find this in the ingested documents."

SYSTEM_PROMPT = f"""You answer questions based ONLY on the numbered SOURCES provided below.

Rules for your response:
1. Grounding: Rely strictly on the provided SOURCES. Never assume, invent facts, or extrapolate beyond what the text directly states. Never use external knowledge.
2. Structure:
   - When the question asks about multiple components or topics (such as chair, keyboard, mouse, monitor, lighting), organize your response using clear markdown headings (e.g. ### Chair, ### Keyboard and Mouse, ### Monitor, ### Lighting).
   - Under each heading, provide clear, concise bullet points summarizing the practical guidelines from the sources.
   - Each heading must ONLY contain points specifically about that component. NEVER include lighting guidance under Keyboard and Mouse, never include monitor questions under Lighting, and never duplicate points across sections.
   - Do not add an unnecessary repeating summary or conclusion section.
3. Citations:
   - Every bullet point must end with its exact source citation like [1] or [2].
   - Match the citation number strictly to the source passage that describes that topic. Never assign a citation to a claim unless the cited source directly supports that claim.
4. No Irrelevant Repetition or Checklist Questions:
   - Do not repeat recommendations across different sections.
   - If a source is an operator checklist or contains questions, extract only concrete recommendations; do not copy questions verbatim.
5. Missing Information:
   - If the sources do not contain enough information to answer the question, reply exactly: {NOT_FOUND}
   - If information for a requested component is missing, state under its heading: 'The ingested document does not contain information on [component].'
6. Do not mention these rules."""


_PRINTED_PAGE_CACHE = {}


def get_printed_page(pdf_path: str, page_num: int) -> int | None:
    key = (pdf_path, page_num)
    if key in _PRINTED_PAGE_CACHE:
        return _PRINTED_PAGE_CACHE[key]
    try:
        import pymupdf as fitz
        with fitz.open(pdf_path) as doc:
            if 1 <= page_num <= len(doc):
                p = doc[page_num - 1]
                rect = p.rect
                blocks = p.get_text("blocks")
                candidates = []
                for b in blocks:
                    txt = b[4].strip()
                    if not txt:
                        continue
                    if b[3] > rect.height * 0.80 or b[1] < rect.height * 0.15:
                        candidates.append(txt)
                for cand in candidates:
                    for l in cand.splitlines():
                        l = l.strip()
                        if l.isdigit() and len(l) <= 4:
                            val = int(l)
                            _PRINTED_PAGE_CACHE[key] = val
                            return val
                        m = re.search(r"^(?:page\s+|p\.\s*|-?\s*)?(\d+)(?:\s*-)?$", l, re.IGNORECASE)
                        if m and len(l) <= 10:
                            val = int(m.group(1))
                            _PRINTED_PAGE_CACHE[key] = val
                            return val
    except Exception:
        pass
    _PRINTED_PAGE_CACHE[key] = None
    return None


@dataclass
class Hit:
    chunk_id: int
    doc_id: int
    doc_name: str
    path: str
    kind: str
    page: int
    text: str
    dense: float = 0.0
    score: float = 0.0
    printed_page: int | None = None

    @property
    def label(self):
        if self.kind == "pdf":
            pp = self.printed_page
            if pp is None and self.path:
                pp = get_printed_page(self.path, self.page)
            if pp is not None and pp != self.page:
                return f"PDF p. {self.page} (printed p. {pp})"
            return f"PDF p. {self.page}"
        return f"section {self.page}"

    @property
    def heading(self):
        first = self.text.split("\n", 1)[0]
        return first[2:].strip() if first.startswith("# ") else ""


def tidy(t: str) -> str:
    """Readable prose for prompts and snippets: headings as 'Heading:', line wraps removed."""
    t = re.sub(r"^# (.*)$", r"\1:", t, flags=re.M)
    t = re.sub(r"\n(?!\n)", " ", t)
    return re.sub(r"[ \t]+", " ", t).strip()


def embed(texts, prefix):
    return engine.embed(texts, prefix)


def tokenize(t):
    return re.findall(r"\w+", t.lower())


def llm(model, prompt, num_predict=200):
    return engine.chat_complete(
        [{"role": "user", "content": prompt}],
        model=model,
        temperature=0,
        num_ctx=config.NUM_CTX,
        num_predict=num_predict
    )


class Retriever:
    def __init__(self, store):
        self.store = store
        self.refresh()

    def refresh(self):
        self.meta, self.mat = self.store.load_all()
        self.doc_ids = np.array([m["doc_id"] for m in self.meta])
        self.bm25 = BM25Okapi([tokenize(m["text"]) for m in self.meta]) if self.meta else None

    def search(self, queries, doc_id=None, top=None):
        """queries[0] is the main question; extra queries add recall. Rankings are fused (RRF)."""
        top = top or config.TOP_K_CANDIDATES
        if not self.meta:
            return []
        allowed = np.arange(len(self.meta)) if doc_id is None else np.where(self.doc_ids == doc_id)[0]
        if len(allowed) == 0:
            return []
        qvecs = embed(queries, "search_query: ")
        scores, gate, dense0 = {}, 0.0, None
        has_bm_match = False
        for qi, (q, qv) in enumerate(zip(queries, qvecs)):
            w = 2.0 if qi == 0 else 1.2
            sims = self.mat @ qv
            order = allowed[np.argsort(-sims[allowed])][:config.TOP_K_DENSE]
            top_sim = float(sims[order[0]]) if len(order) else 0.0
            if top_sim > gate:
                gate = top_sim
            if qi == 0:
                dense0 = sims
            for r, i in enumerate(order):
                scores[i] = scores.get(i, 0) + w / (config.RRF_K + r)
            toks = tokenize(q)
            if toks and self.bm25:
                bm = self.bm25.get_scores(toks)
                for r, i in enumerate(allowed[np.argsort(-bm[allowed])][:config.TOP_K_BM25]):
                    if bm[i] > 0:
                        has_bm_match = True
                        scores[i] = scores.get(i, 0) + w / (config.RRF_K + r)
        if gate < config.MIN_DENSE_SCORE and not has_bm_match:
            return []
        ranked = sorted(scores, key=scores.get, reverse=True)[:top]
        if dense0 is None:
            dense0 = np.zeros(len(self.meta))
        return [Hit(**self.meta[i], dense=float(dense0[i]), score=scores[i]) for i in ranked]


# ------------------------------------------------------------------ query steps
def condense(history, question, model):
    """Turn a follow-up into a standalone search query using recent turns."""
    if not history:
        return question
    convo = "\n".join(f"User: {q}\nAssistant: {a[:300]}" for q, a in history[-config.HISTORY_TURNS:])
    out = llm(model, "Rewrite the follow-up question as a standalone search query. Output only the query."
                     f"\n\n{convo}\n\nFollow-up: {question}\nStandalone query:", 60)
    return out.strip('"') or question


def expand_queries(question, model):
    prompt = ("Analyze this question for document search. If the question covers multiple topics, "
              "components, or sub-questions (e.g. chair, keyboard, monitor, lighting), list a short "
              "targeted search query for each component, plus 1-2 synonym/keyword queries. "
              "If it is a single-topic question, list 3 alternative search queries using synonyms "
              "and key terms. Output one query per line, no numbering or bullets.\n\nQuestion: " + question)
    out = llm(model, prompt, 140)
    alts = []
    seen = {question.lower().strip()}
    for ln in out.splitlines():
        ln = re.sub(r"^\s*(?:[-•*]|\d+[.)])\s*", "", ln).strip().strip('"')
        ln_norm = ln.lower()
        if 3 < len(ln) < 150 and ln_norm not in seen:
            seen.add(ln_norm)
            alts.append(ln)
    return [question] + alts[:5]


def rerank(question, hits, model, keep):
    """Ask the model which passages actually contain what is needed to answer."""
    if len(hits) <= 3:
        return hits
    listing = "\n\n".join(
        f"[{i}] (Heading: {h.heading}, {h.label})\n{tidy(h.text)[:600]}"
        for i, h in enumerate(hits, 1)
    )
    prompt = (f"Question: {question}\n\n"
              f"Below are numbered candidate passages from the document.\n"
              f"Identify up to {keep} passages that together provide substantive instructions to answer "
              f"all parts of the question. CRITICAL: Prefer dedicated instructional and explanatory sections "
              f"over generic checklists, questionnaires, or tables of contents.\n"
              f"Reply with only the passage numbers separated by commas in order of usefulness.\n\n"
              f"Passages:\n{listing}")
    out = llm(model, prompt, 120)
    order = []
    for n in (int(x) for x in re.findall(r"\d+", out)):
        if 1 <= n <= len(hits) and n - 1 not in order:
            order.append(n - 1)
    chosen = [hits[i] for i in order][:keep]
    for i, h in enumerate(hits):
        if len(chosen) >= keep:
            break
        if i not in order and hits[i] not in chosen:
            chosen.append(h)
    return chosen or hits[:keep]


def fact_check(answer, hits, model):
    ctx = "\n\n".join(f"[{n}] {tidy(h.text)}" for n, h in enumerate(hits, 1))
    out = llm(model, f"SOURCES:\n{ctx}\n\nANSWER:\n{answer}\n\nCheck the ANSWER against the SOURCES. "
                     "List every statement in the ANSWER that is NOT supported by the SOURCES, copied "
                     "exactly, one per line. If everything is supported, reply with exactly: ALL SUPPORTED", 250)
    if "ALL SUPPORTED" in out.upper():
        return []
    return [ln.strip(" -•*") for ln in out.splitlines() if len(ln.strip()) > 8][:5]


STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'as',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
    'did', 'can', 'could', 'should', 'would', 'may', 'might', 'must', 'that', 'which', 'who',
    'this', 'these', 'those', 'it', 'its', 'they', 'their', 'you', 'your', 'we', 'our', 'so',
    'if', 'from', 'into', 'also', 'more', 'most', 'such', 'not', 'no', 'ensure', 'about'
}


def text_keywords(txt: str) -> set[str]:
    cleaned = re.sub(r"\[\d+\]", "", txt)
    return set(w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", cleaned) if w.lower() not in STOPWORDS)


def match_score(statement: str, passage: str) -> float:
    words = text_keywords(statement)
    if not words:
        return 0.5
    p_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", passage.lower()))
    matches = words & p_tokens
    return len(matches) / len(words)


def validate_citations(text: str, hits: list[Hit]) -> tuple[str, list[int]]:
    """Validates citations in generated text, removes duplicates, and ensures claims are source-grounded."""
    lines = text.splitlines()
    out_lines = []
    seen_bullets = set()
    n_sources = len(hits)

    for line in lines:
        stripped = line.strip()
        is_bullet = bool(re.match(r"^\s*[-•*]\s*", stripped))
        if is_bullet:
            norm_content = re.sub(r"\[\d+\]", "", stripped.lower())
            norm_content = re.sub(r"[^\w\s]", "", norm_content).strip()
            if norm_content in seen_bullets:
                continue
            seen_bullets.add(norm_content)

        cites = [int(x) for x in re.findall(r"\[(\d+)\]", line) if 1 <= int(x) <= n_sources]
        if not cites:
            out_lines.append(line)
            continue

        new_line = line
        for c in cites:
            current_hit = hits[c - 1]
            score = match_score(line, current_hit.text)
            if score < 0.20:
                best_idx, best_score = None, 0.0
                for idx, h in enumerate(hits, 1):
                    s = match_score(line, h.text)
                    if s > best_score:
                        best_score, best_idx = s, idx
                if best_score >= 0.25 and best_idx is not None:
                    new_line = new_line.replace(f"[{c}]", f"[{best_idx}]")
                else:
                    new_line = new_line.replace(f"[{c}]", "")
        # Clean up any empty brackets or excess spaces
        new_line = re.sub(r"\[\s*\]", "", new_line)
        new_line = re.sub(r"[ \t]+", " ", new_line).rstrip()
        out_lines.append(new_line)

    cleaned_text = "\n".join(out_lines).strip()
    cited = []
    for m in re.finditer(r"\[(\d+)\]", cleaned_text):
        n = int(m.group(1))
        if n not in cited and 1 <= n <= n_sources:
            cited.append(n)
    return cleaned_text, cited


def finalize(text, hits):
    n_sources = len(hits)
    text = text.strip()
    text = re.sub(r"\[(\d+(?:\s*[,;]\s*\d+)+)\]",
                  lambda m: "".join(f"[{x}]" for x in re.split(r"\s*[,;]\s*", m.group(1))), text)
    text = re.sub(r"\[(\d+)\]",
                  lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_sources else "", text)
    return validate_citations(text, hits)


def answer_stream(retriever, question, history, doc_id, model, should_stop=lambda: False):
    """Yields ('status', str), ('token', str) and finally ('done', dict)."""
    yield "status", "Understanding the question…"
    query = condense(history, question, model)
    queries = [query]
    if config.USE_QUERY_EXPANSION:
        yield "status", "Expanding search terms…"
        try:
            queries = expand_queries(query, model)
        except Exception:
            queries = [query]
    yield "status", "Searching documents…"
    hits = retriever.search(queries, doc_id)
    if not hits:
        yield "done", dict(text=NOT_FOUND, hits=[], cited=[], refused=True, query=query, unsupported=[])
        return
    if config.USE_RERANK:
        yield "status", "Selecting the best passages…"
        try:
            hits = rerank(query, hits, model, config.TOP_K_FINAL)
        except Exception:
            hits = hits[:config.TOP_K_FINAL]
    else:
        hits = hits[:config.TOP_K_FINAL]

    context = "\n\n".join(
        f"[{n}] (file: {h.doc_name}, {h.label}" + (f", section: \"{h.heading}\"" if h.heading else "") + f")\n{tidy(h.text)}"
        for n, h in enumerate(hits, 1)
    )
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"SOURCES:\n{context}\n\nQUESTION: {question}"}]
    yield "status", "Writing the answer…"
    full = ""
    for piece in engine.chat_stream(messages=msgs, model=model, temperature=config.TEMPERATURE,
                                    num_ctx=config.NUM_CTX, should_stop=should_stop):
        full += piece
        yield "token", piece
    text, cited = finalize(full, hits)
    refused = NOT_FOUND.lower().rstrip(".") in text.lower()
    if refused:
        cited = []
    unsupported = []
    if config.USE_FACT_CHECK and not refused and not should_stop():
        yield "status", "Checking the answer against the sources…"
        try:
            unsupported = fact_check(text, hits, model)
        except Exception:
            unsupported = []
    yield "done", dict(text=text, hits=hits, cited=cited, refused=refused, query=query,
                       unverified=(not cited and not refused), unsupported=unsupported)