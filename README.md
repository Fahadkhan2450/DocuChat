# Local Document Assistant

Offline RAG chatbot with a desktop UI. Answers only from your ingested files and gives clickable
citations: clicking `[1]` opens the PDF page with the passage highlighted.

## Privacy
- Chat and embedding models run in Ollama on `127.0.0.1`. No cloud API, no telemetry.
- Documents are copied into `data/documents/`, the index is `data/index.sqlite`. Delete `data/` to wipe everything.
- Internet is needed only once, for `pip install` and `ollama pull`.

## Setup (macOS / Windows / Linux)
1. Install Ollama from https://ollama.com and start it.
2. Pull the models:
   ```
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```
3. Use Python 3.11 or 3.12, then:
   ```
   cd local_rag_desktop
   python3 -m venv venv
   source venv/bin/activate          # Windows: venv\Scripts\activate
   python -m pip install -r requirements.txt
   python app.py
   ```
   On macOS you can also double-click `run.command`.

## Use
1. **Add documents…** (PDF, DOCX, TXT, MD, CSV). Wait for the "indexed" message.
2. Ask a question. Use **Search in** to limit it to one document.
3. Click `[n]` in the answer or a line under **Sources** to see the exact page on the right.

## IMPORTANT after updating from an older version
Chunking changed. Delete the `data/` folder (or remove each document in the app) and add your documents again.

## How it keeps answers correct
- Section-aware chunks: each chunk is one section with its heading (for example "Automatic direction finding > Accurate AoA direction finding"), so the right passage is found and cited.
- Query expansion: the question is rewritten into extra keyword queries, which catches option names like "CS-DF".
- Re-ranking: the model picks which retrieved passages truly answer the question before it writes.
- Fact check: statements the sources do not support are listed under the answer with a warning.
- Hybrid search: embeddings (meaning) + BM25 (exact terms like key names), merged with rank fusion.
- Chunks never cross pages, so page citations are exact.
- Strict prompt: answer only from the numbered sources, cite every claim, otherwise say it isn't found.
- A relevance gate returns "not found" when nothing similar exists, instead of letting the model guess.
- Invalid citation numbers are removed; an answer with no citations is flagged "unverified".
- Temperature 0.1. Follow-up questions are rewritten into standalone search queries.

No local model is perfect: always click the citation for important facts.

## Tuning (config.py)
- Better answers: `CHAT_MODEL = "qwen2.5:14b"` (needs about 16 GB RAM).
- Multilingual documents: `ollama pull bge-m3` and set `EMBED_MODEL = "bge-m3"` (re-add documents after changing).
- Answers miss details: raise `TOP_K_FINAL` to 8 or `CHUNK_CHARS` to 1400.
- Too many "not found": lower `MIN_DENSE_SCORE` to 0.2.

## Limits
- Scanned PDFs (images) have no text; run OCR first (e.g. `ocrmypdf in.pdf out.pdf`).
- If you change `EMBED_MODEL` or chunk settings, remove and re-add the documents.
- Package as an app (optional): `pip install pyinstaller` then `pyinstaller --windowed app.py`.
# DocuChat
