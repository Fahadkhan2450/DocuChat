"""Local RAG desktop app. Run: python app.py"""
import html
import re
import shutil
import sys
from pathlib import Path

import pymupdf as fitz  # PyMuPDF
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QScrollArea, QSplitter, QStackedWidget,
                               QTextBrowser, QVBoxLayout, QWidget)

import config
import ingest
import rag
import store


# ----------------------------------------------------------------- workers
class IngestWorker(QThread):
    progress = Signal(str)
    finished_ok = Signal(str)

    def __init__(self, paths):
        super().__init__()
        self.paths = paths

    def run(self):
        summary = []
        for p in self.paths:
            try:
                self.progress.emit(f"Reading {p.name}…")
                sha = store.sha_of_file(p)
                if store.doc_exists(sha):
                    summary.append(f"• {p.name}: already indexed")
                    continue
                chunks = ingest.make_chunks(ingest.extract_sections(p))
                if not chunks:
                    summary.append(f"• {p.name}: no extractable text (scanned PDF? it needs OCR first)")
                    continue
                embs = []
                for i in range(0, len(chunks), 32):
                    self.progress.emit(f"Embedding {p.name}: {i}/{len(chunks)}")
                    batch = chunks[i:i + 32]
                    embs.append(rag.embed([ingest.embed_text(p.name, pg, t) for pg, _, t in batch],
                                          "search_document: "))
                dest = config.DOCS_DIR / f"{sha[:10]}_{p.name}"
                shutil.copy2(p, dest)
                store.add_document(p.name, dest, sha, p.suffix.lower().lstrip("."), chunks, np.vstack(embs))
                summary.append(f"• {p.name}: {len(chunks)} chunks indexed")
            except Exception as e:
                summary.append(f"• {p.name}: failed ({e})")
        self.finished_ok.emit("\n".join(summary))


class ChatWorker(QThread):
    status = Signal(str)
    token = Signal(str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, retriever, question, history, doc_id, model):
        super().__init__()
        self.args = (retriever, question, history, doc_id, model)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            for ev, val in rag.answer_stream(*self.args, should_stop=lambda: self._stop):
                if ev == "status":
                    self.status.emit(val)
                elif ev == "token":
                    self.token.emit(val)
                else:
                    self.done.emit(val)
        except Exception as e:
            self.failed.emit(str(e))


# ----------------------------------------------------------------- helpers
CITE = re.compile(r"\[(\d+)\]")


def render_answer(text, msg_idx, valid):
    t = html.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)

    def link(m):
        n = int(m.group(1))
        if n in valid:
            return (f'<a href="cite:{msg_idx}:{n}" style="color:#1a56db;'
                    f'text-decoration:none;font-weight:600">[{n}]</a>')
        return m.group(0)
    return CITE.sub(link, t).replace("\n", "<br>")


def render_pdf_page(path, page_no, passage, zoom=1.5):
    """Render a PDF page with the cited passage highlighted. Returns (QImage, y_of_first_highlight)."""
    with fitz.open(path) as doc:
        page = doc[page_no - 1]
        first_y = None
        lines = [l.strip().lstrip("# ").strip() for l in passage.splitlines() if len(l.strip()) >= 10][:80]
        for line in lines:
            rects = page.search_for(line) or page.search_for(line[:30])
            for r in rects:
                page.add_highlight_annot(r)
                if first_y is None or r.y0 < first_y:
                    first_y = r.y0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), annots=True, alpha=False)
        img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888).copy()
    return img, (first_y or 0) * zoom


# ----------------------------------------------------------------- main window
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local Document Assistant (100% offline)")
        self.resize(1500, 900)
        store.init()
        self.retriever = rag.Retriever(store)
        self.messages = []     # dicts: role, text, hits, cited, ...
        self.history = []      # (question, answer) pairs
        self.worker = None
        self.current_path = None
        self._build_ui()
        self.reload_docs()
        QTimer.singleShot(200, self.check_ollama)

    # ---- UI
    def _build_ui(self):
        left = QWidget()
        lv = QVBoxLayout(left)
        self.status = QLabel("Starting…")
        self.status.setWordWrap(True)
        btn_add = QPushButton("Add documents…")
        btn_add.clicked.connect(self.add_docs)
        self.doc_list = QListWidget()
        btn_del = QPushButton("Remove selected")
        btn_del.clicked.connect(self.remove_doc)
        self.scope = QComboBox()
        self.model_box = QComboBox()
        for w in (self.status, btn_add, self.doc_list, btn_del,
                  QLabel("Search in:"), self.scope, QLabel("Chat model:"), self.model_box):
            lv.addWidget(w)

        center = QWidget()
        cv = QVBoxLayout(center)
        self.chat = QTextBrowser()
        self.chat.setOpenLinks(False)
        self.chat.anchorClicked.connect(self.on_link)
        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask a question about your documents…")
        self.input.returnPressed.connect(self.send)
        self.btn_send = QPushButton("Send")
        self.btn_send.clicked.connect(self.send)
        btn_clear = QPushButton("New chat")
        btn_clear.clicked.connect(self.clear_chat)
        for w in (self.input, self.btn_send, btn_clear):
            row.addWidget(w)
        cv.addWidget(self.chat)
        cv.addLayout(row)

        right = QWidget()
        rv = QVBoxLayout(right)
        self.v_title = QLabel("Click a citation like [1] to view its source here")
        self.v_title.setWordWrap(True)
        self.page_label = QLabel()
        self.page_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.page_label)
        self.text_view = QTextBrowser()
        self.stack = QStackedWidget()
        self.stack.addWidget(self.scroll)
        self.stack.addWidget(self.text_view)
        btn_open = QPushButton("Open original file")
        btn_open.clicked.connect(
            lambda: self.current_path and QDesktopServices.openUrl(QUrl.fromLocalFile(self.current_path)))
        rv.addWidget(self.v_title)
        rv.addWidget(self.stack, 1)
        rv.addWidget(btn_open)

        split = QSplitter()
        for w in (left, center, right):
            split.addWidget(w)
        split.setSizes([260, 620, 620])
        lay = QHBoxLayout(self)
        lay.addWidget(split)

    # ---- Ollama / documents
    def check_ollama(self):
        try:
            names = [m["model"] for m in rag.client.list()["models"]]
        except Exception:
            self.status.setText("Ollama is NOT running.\nStart the Ollama app, then restart this program.")
            return
        base = {n.split(":")[0] for n in names}
        missing = [m for m in (config.EMBED_MODEL, config.CHAT_MODEL.split(":")[0]) if m not in base]
        chat_models = [n for n in names if "embed" not in n] or [config.CHAT_MODEL]
        self.model_box.clear()
        self.model_box.addItems(chat_models)
        if config.CHAT_MODEL in chat_models:
            self.model_box.setCurrentText(config.CHAT_MODEL)
        if missing:
            self.status.setText("Ollama connected, but models are missing.")
            QMessageBox.warning(self, "Models missing", "Run in a terminal:\n\n" +
                                "\n".join(f"ollama pull {m}" for m in
                                          (config.EMBED_MODEL, config.CHAT_MODEL) if m.split(":")[0] in missing))
        else:
            self.status.setText("Ollama connected ✓  (all data stays on this computer)")

    def reload_docs(self):
        self.retriever.refresh()
        self.doc_list.clear()
        self.scope.clear()
        self.scope.addItem("All documents", None)
        for did, name, kind, n in store.list_documents():
            it = QListWidgetItem(f"{name}  ({n} chunks)")
            it.setData(Qt.UserRole, did)
            self.doc_list.addItem(it)
            self.scope.addItem(name, did)

    def add_docs(self):
        exts = " ".join(f"*{e}" for e in sorted(ingest.SUPPORTED))
        files, _ = QFileDialog.getOpenFileNames(self, "Add documents", "", f"Documents ({exts})")
        if not files:
            return
        self.ingest_worker = IngestWorker([Path(f) for f in files])
        self.ingest_worker.progress.connect(self.status.setText)
        self.ingest_worker.finished_ok.connect(self.on_ingested)
        self.btn_send.setEnabled(False)
        self.ingest_worker.start()

    def on_ingested(self, summary):
        self.btn_send.setEnabled(True)
        self.reload_docs()
        self.status.setText("Ollama connected ✓")
        QMessageBox.information(self, "Ingestion finished", summary)

    def remove_doc(self):
        it = self.doc_list.currentItem()
        if it and QMessageBox.question(self, "Remove", "Remove this document and its index?") \
                == QMessageBox.Yes:
            store.delete_document(it.data(Qt.UserRole))
            self.reload_docs()

    # ---- chat
    def send(self):
        q = self.input.text().strip()
        if not q or (self.worker and self.worker.isRunning()):
            return
        self.input.clear()
        self.messages.append(dict(role="user", text=q))
        self.messages.append(dict(role="assistant", text="", hits=[], cited=[], final=False))
        self.render()
        self.btn_send.setEnabled(False)
        self.worker = ChatWorker(self.retriever, q, list(self.history),
                                 self.scope.currentData(), self.model_box.currentText())
        self.worker.status.connect(self.status.setText)
        self.worker.token.connect(self.on_token)
        self.worker.done.connect(lambda res, q=q: self.on_done(q, res))
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_token(self, piece):
        self.messages[-1]["text"] += piece
        self.render()

    def on_done(self, q, res):
        m = self.messages[-1]
        m.update(text=res["text"], hits=res["hits"], cited=res["cited"], final=True,
                 refused=res["refused"], unverified=res.get("unverified", False),
                 unsupported=res.get("unsupported", []))
        self.history.append((q, res["text"]))
        self.btn_send.setEnabled(True)
        self.status.setText("Ready")
        self.render()

    def on_failed(self, err):
        self.messages[-1].update(text=f"Error: {err}\n\nIs Ollama running and are the models pulled?",
                                 final=True)
        self.btn_send.setEnabled(True)
        self.render()

    def clear_chat(self):
        self.messages.clear()
        self.history.clear()
        self.render()

    def render(self):
        parts = []
        for i, m in enumerate(self.messages):
            if m["role"] == "user":
                parts.append('<div style="background:#e8f0fe;color:#111;padding:8px;margin:6px 0">'
                             f'<b>You</b><br>{html.escape(m["text"])}</div>')
                continue
            body = render_answer(m["text"] or "…", i, set(range(1, len(m["hits"]) + 1)))
            extra = ""
            if m.get("final") and m["cited"]:
                links = []
                for n in m["cited"]:
                    h = m["hits"][n - 1]
                    snip = html.escape(rag.tidy(h.text)[:200])
                    links.append(f'<a href="cite:{i}:{n}" style="color:#1a56db">[{n}] '
                                 f'{html.escape(h.doc_name)} — {h.label}'
                                 + (f' · {html.escape(h.heading)}' if h.heading else '') + '</a>'
                                 f'<br><span style="color:#666;font-size:small">“{snip}…”</span>')
                extra = '<hr><b>Sources</b> (click to open)<br>' + "<br>".join(links)
            if m.get("unsupported"):
                items = "".join(f"<li>{html.escape(u)}</li>" for u in m["unsupported"])
                extra += ('<br><span style="color:#b45309">⚠ The fact check could not confirm these statements '
                          f'in the sources (verify them in the cited pages):</span><ul>{items}</ul>')
            if m.get("unverified"):
                extra += ('<br><span style="color:#b45309">⚠ No citations were produced. '
                          'Treat this answer as unverified.</span>')
            parts.append('<div style="background:#f4f4f5;color:#111;padding:8px;margin:6px 0">'
                         f'<b>Assistant</b><br>{body}{extra}</div>')
        self.chat.setHtml("".join(parts))
        sb = self.chat.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ---- citation viewer
    def on_link(self, url: QUrl):
        s = url.toString()
        if not s.startswith("cite:"):
            return
        _, mi, n = s.split(":")
        hit = self.messages[int(mi)]["hits"][int(n) - 1]
        self.show_hit(hit)

    def show_hit(self, hit):
        self.current_path = hit.path
        self.v_title.setText(f"{hit.doc_name} — {hit.label}")
        if hit.kind == "pdf":
            try:
                img, y = render_pdf_page(hit.path, hit.page, hit.text)
                self.page_label.setPixmap(QPixmap.fromImage(img))
                self.page_label.adjustSize()
                self.stack.setCurrentIndex(0)
                QTimer.singleShot(60, lambda: self.scroll.verticalScrollBar().setValue(max(0, int(y) - 120)))
                return
            except Exception as e:
                self.text_view.setPlainText(f"(Could not render page: {e})\n\n{hit.text}")
        else:
            self.text_view.setHtml(
                f'<p style="color:#666">Passage from {html.escape(hit.doc_name)} — {hit.label}</p>'
                f'<div style="background:#fff3a3;color:#111;padding:8px">'
                f'{html.escape(hit.text).replace(chr(10), "<br>")}</div>')
        self.stack.setCurrentIndex(1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())