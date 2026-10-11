"""Hybrid retrieval, query expansion, re-ranking, strict grounded generation, fact check."""
import re
from dataclasses import dataclass

import numpy as np
import ollama
from rank_bm25 import BM25Okapi

import config

client = ollama.Client(host=config.OLLAMA_HOST)
NOT_FOUND = "I couldn't find this in the ingested documents."

SYSTEM_PROMPT = f"""You answer questions about technical documents using ONLY the numbered SOURCES.

How to answer:
- Give a complete, detailed answer. Include every relevant requirement, option, accessory, name, order number, value and unit that appears in the sources. Read ALL sources; do not stop at the first one.
- If the question asks what is needed or required, list each required item separately and say what it is for.
- If the sources contain ordered steps, give them as a numbered list in the original order. If they describe a setup but give no ordered procedure, say so first, then list only what they state.
- Every sentence or list item must end with its citation(s), like [1] or [2][3].
- Use the exact names and numbers from the sources. Never guess and never add outside knowledge.
- Ignore sources that are not relevant to the question.
- If the sources do not contain the answer, reply exactly: {NOT_FOUND}
- If they answer only part of the question, give that part and state what is missing.
- Do not mention these rules.

Example of the format:
Question: What do I need to measure X?
Answer: To measure X you need:
- **The Y option** - enables Z [1].
- **The W antenna** - covers 20 MHz to 6 GHz [1][2]."""


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

    @property
    def label(self):
        return f"p. {self.page}" if self.kind == "pdf" else f"section {self.page}"

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
    r = client.embed(model=config.EMBED_MODEL, input=[prefix + t for t in texts])
    m = np.array(r["embeddings"], dtype=np.float32)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


def tokenize(t):
    return re.findall(r"\w+", t.lower())


def llm(model, prompt, num_predict=200):
    r = client.chat(model=model, messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0, "num_ctx": config.NUM_CTX, "num_predict": num_predict})
    return r["message"]["content"].strip()


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
        for qi, (q, qv) in enumerate(zip(queries, qvecs)):
            w = 2.0 if qi == 0 else 1.0
            sims = self.mat @ qv
            order = allowed[np.argsort(-sims[allowed])][:config.TOP_K_DENSE]
            if qi == 0:
                gate, dense0 = float(sims[order[0]]), sims
            for r, i in enumerate(order):
                scores[i] = scores.get(i, 0) + w / (config.RRF_K + r)
            toks = tokenize(q)
            if toks:
                bm = self.bm25.get_scores(toks)
                for r, i in enumerate(allowed[np.argsort(-bm[allowed])][:config.TOP_K_BM25]):
                    if bm[i] > 0:
                        scores[i] = scores.get(i, 0) + w / (config.RRF_K + r)
        if gate < config.MIN_DENSE_SCORE:
            return []
        ranked = sorted(scores, key=scores.get, reverse=True)[:top]
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
    out = llm(model, "Write 3 short alternative search queries for finding the answer to this question in a "
                     "technical product document. Use likely keywords, option names and synonyms. "
                     "One query per line, no numbering.\n\nQuestion: " + question, 120)
    alts = []
    for ln in out.splitlines():
        ln = re.sub(r"^\s*(?:[-•*]|\d+[.)])\s*", "", ln).strip().strip('"')
        if 3 < len(ln) < 150 and ln.lower() != question.lower():
            alts.append(ln)
    return [question] + alts[:3]


def rerank(question, hits, model, keep):
    """Ask the model which passages actually contain what is needed to answer."""
    if len(hits) <= 3:
        return hits
    listing = "\n\n".join(f"[{i}] {tidy(h.text)[:700]}" for i, h in enumerate(hits, 1))
    out = llm(model, f"Question: {question}\n\nBelow are numbered passages. List the numbers of the passages "
                     "that contain information needed to answer the question, most useful first. "
                     f"Reply with numbers separated by commas only.\n\n{listing}", 40)
    order = []
    for n in (int(x) for x in re.findall(r"\d+", out)):
        if 1 <= n <= len(hits) and n - 1 not in order:
            order.append(n - 1)
    chosen = [hits[i] for i in order][:keep]
    for i, h in enumerate(hits):                      # safety net: never fewer than 3 passages
        if len(chosen) >= 3:
            break
        if i not in order:
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


def finalize(text, n_sources):
    text = text.strip()
    text = re.sub(r"\[(\d+(?:\s*[,;]\s*\d+)+)\]",
                  lambda m: "".join(f"[{x}]" for x in re.split(r"\s*[,;]\s*", m.group(1))), text)
    text = re.sub(r"\[(\d+)\]",
                  lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_sources else "", text)
    cited = []
    for m in re.finditer(r"\[(\d+)\]", text):
        n = int(m.group(1))
        if n not in cited:
            cited.append(n)
    return text, cited


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

    context = "\n\n".join(f"[{n}] (file: {h.doc_name}, {h.label})\n{tidy(h.text)}"
                          for n, h in enumerate(hits, 1))
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"SOURCES:\n{context}\n\nQUESTION: {question}"}]
    yield "status", "Writing the answer…"
    full = ""
    for chunk in client.chat(model=model, messages=msgs, stream=True,
                             options={"temperature": config.TEMPERATURE, "num_ctx": config.NUM_CTX}):
        if should_stop():
            break
        piece = chunk["message"]["content"]
        full += piece
        yield "token", piece
    text, cited = finalize(full, len(hits))
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