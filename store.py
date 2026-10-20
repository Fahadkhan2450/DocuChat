"""SQLite storage for documents, chunks and embeddings."""
import hashlib
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from config import DB_PATH


@contextmanager
def db():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.execute("PRAGMA foreign_keys=ON")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init():
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS docs(
            id INTEGER PRIMARY KEY, name TEXT, path TEXT, sha TEXT UNIQUE,
            kind TEXT, added REAL);
        CREATE TABLE IF NOT EXISTS chunks(
            id INTEGER PRIMARY KEY,
            doc_id INTEGER REFERENCES docs(id) ON DELETE CASCADE,
            page INTEGER, idx INTEGER, text TEXT, emb BLOB,
            printed_page INTEGER);
        CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(doc_id);
        """)
        try:
            c.execute("ALTER TABLE chunks ADD COLUMN printed_page INTEGER")
        except sqlite3.OperationalError:
            pass


def sha_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def doc_exists(sha: str) -> bool:
    with db() as c:
        return c.execute("SELECT 1 FROM docs WHERE sha=?", (sha,)).fetchone() is not None


def add_document(name, path, sha, kind, chunks, embs: np.ndarray):
    """chunks: list of (page, printed_page, idx, text) or (page, idx, text); embs: normalized float32 matrix."""
    with db() as c:
        cur = c.execute("INSERT INTO docs(name,path,sha,kind,added) VALUES(?,?,?,?,?)",
                        (name, str(path), sha, kind, time.time()))
        doc_id = cur.lastrowid
        records = []
        for k, ch in enumerate(chunks):
            if len(ch) == 4:
                p, pp, i, t = ch
            else:
                p, i, t = ch
                pp = None
            records.append((doc_id, p, i, t, embs[k].astype(np.float32).tobytes(), pp))
        c.executemany(
            "INSERT INTO chunks(doc_id,page,idx,text,emb,printed_page) VALUES(?,?,?,?,?,?)",
            records)
    return doc_id


def delete_document(doc_id: int):
    with db() as c:
        row = c.execute("SELECT path FROM docs WHERE id=?", (doc_id,)).fetchone()
        c.execute("DELETE FROM docs WHERE id=?", (doc_id,))
    if row:
        Path(row[0]).unlink(missing_ok=True)


def list_documents():
    with db() as c:
        return c.execute("""SELECT d.id, d.name, d.kind, COUNT(k.id)
                            FROM docs d LEFT JOIN chunks k ON k.doc_id=d.id
                            GROUP BY d.id ORDER BY d.added""").fetchall()


def load_all():
    """Returns (meta list, embedding matrix)."""
    with db() as c:
        has_pp = any(col[1] == "printed_page" for col in c.execute("PRAGMA table_info(chunks)").fetchall())
        query = ("SELECT k.id, k.doc_id, d.name, d.path, d.kind, k.page, k.text, k.emb, k.printed_page "
                 "FROM chunks k JOIN docs d ON d.id=k.doc_id ORDER BY k.id") if has_pp else (
                 "SELECT k.id, k.doc_id, d.name, d.path, d.kind, k.page, k.text, k.emb "
                 "FROM chunks k JOIN docs d ON d.id=k.doc_id ORDER BY k.id")
        rows = c.execute(query).fetchall()
    meta, vecs = [], []
    for r in rows:
        cid, did, name, path, kind, page, text, emb = r[:8]
        printed_page = r[8] if len(r) > 8 else None
        meta.append(dict(chunk_id=cid, doc_id=did, doc_name=name, path=path,
                         kind=kind, page=page, text=text, printed_page=printed_page))
        vecs.append(np.frombuffer(emb, dtype=np.float32))
    mat = np.vstack(vecs) if vecs else np.zeros((0, 1), dtype=np.float32)
    return meta, mat