# DocuChat

### Private, Offline Document Intelligence

DocuChat is a desktop-based document assistant that lets users ask questions about their own files and receive evidence-based answers with clickable citations. Built on local language models and retrieval-augmented generation (RAG), it retrieves relevant passages from ingested documents and links responses directly to their original sources, including the corresponding PDF page and highlighted passage.

All document processing, retrieval, and response generation run locally, keeping your files and conversations on your own machine.

## Key Features

* **Offline AI Chat:** Ask natural-language questions about your documents without relying on cloud-based AI services.
* **Document Support:** Ingest PDF, DOCX, TXT, Markdown, and CSV files.
* **Clickable Citations:** Navigate from an answer to the exact PDF page and highlighted supporting passage.
* **Hybrid Retrieval:** Combine semantic embeddings with BM25 keyword search to retrieve both conceptually relevant passages and exact technical terms.
* **Section-Aware Chunking:** Preserve document headings and section context to improve retrieval accuracy.
* **Query Expansion:** Generate additional search queries to improve retrieval for technical terminology, abbreviations, and alternative phrasing.
* **Passage Reranking:** Reorder retrieved passages according to their relevance to the question.
* **Evidence-Based Responses:** Restrict answers to retrieved sources, validate citation references, and flag unsupported statements.
* **Document Filtering:** Limit searches to a selected document when needed.
* **Follow-Up Questions:** Resolve follow-up questions into standalone queries while retaining conversational context.

## Privacy and Data Storage

DocuChat is designed for local operation and private document access.

* Chat and embedding models run through Ollama on `127.0.0.1`.
* No cloud AI API or application telemetry is required by the described architecture.
* Imported documents are stored in `data/documents/`.
* The retrieval index is stored in `data/index.sqlite`.
* Internet access is required initially to install Python dependencies and download the selected models.

To remove the locally stored documents and index, close the application and delete the `data/` directory. This permanently removes the files and index stored there, so back up anything you need first.

## Technology Stack

* **Language:** Python
* **Local Model Runtime:** Ollama
* **Chat Model:** Llama 3.1 8B
* **Embedding Model:** Nomic Embed Text
* **Retrieval:** Semantic search and BM25
* **Index Storage:** SQLite
* **Application:** Desktop user interface

## Requirements

* Python 3.11 or 3.12
* Ollama
* Git (optional, for cloning the repository)
* Sufficient memory and disk space for the selected models

## Installation

### 1. Install Ollama

Download and install Ollama from [ollama.com](https://ollama.com), then start the service.

### 2. Download the Models

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 3. Set Up the Python Environment

Clone or open the project directory, then run:

```bash
cd local_rag_desktop
python3 -m venv venv
```

Activate the environment.

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```powershell
venv\Scripts\activate
```

Install the dependencies and launch the application:

```bash
python -m pip install -r requirements.txt
python app.py
```

On macOS, you can also launch the application by double-clicking `run.command`, if provided.

## How to Use

1. **Import documents:** Select *Add documents…* and choose PDF, DOCX, TXT, Markdown, or CSV files.
2. **Wait for indexing:** Allow the application to process the documents and confirm that indexing has completed.
3. **Ask a question:** Enter a question about the imported material in the chat interface.
4. **Refine the search:** Use *Search in* to restrict retrieval to a particular document.
5. **Verify the answer:** Select a citation such as `[1]` or a source entry to open the relevant PDF page and inspect the highlighted passage.

## How Answer Reliability Is Improved

DocuChat uses several retrieval and response-generation safeguards to reduce unsupported answers.

1. **Hybrid retrieval:** Semantic search identifies conceptually related passages, while BM25 helps retrieve exact terms, model numbers, and technical identifiers.
2. **Section-aware indexing:** Chunks retain their section headings to preserve context during retrieval.
3. **Query expansion:** Additional keyword-oriented queries improve the chance of finding relevant information.
4. **Reranking:** Candidate passages are reordered before being supplied to the language model.
5. **Page-aware chunking:** Chunks do not cross page boundaries, allowing citations to refer to the relevant source page.
6. **Grounded generation:** The model is instructed to answer from the numbered retrieved sources and cite its claims.
7. **Relevance filtering:** A relevance threshold helps reject questions for which the indexed documents provide insufficient evidence.
8. **Citation validation:** Invalid citation references are removed, and answers without valid citations are marked as unverified.
9. **Evidence checking:** Unsupported statements are identified and presented with a warning when the verification stage detects them.
10. **Follow-up resolution:** Follow-up questions are rewritten into standalone search queries to improve retrieval.

These safeguards improve traceability but do not guarantee factual correctness. For important technical or operational decisions, verify the original passage and its surrounding context.

## Configuration

Model and retrieval settings can be adjusted in `config.py`.

### Improve Answer Quality

For systems with sufficient memory, use a larger chat model:

```python
CHAT_MODEL = "qwen2.5:14b"
```

This model may require approximately 16 GB of memory or more, depending on quantization and runtime overhead.

### Support Multilingual Documents

Download the multilingual embedding model:

```bash
ollama pull bge-m3
```

Then update the embedding configuration:

```python
EMBED_MODEL = "bge-m3"
```

Re-index your documents after changing the embedding model.

### Retrieve More Evidence

Increase the number of final passages supplied to the model:

```python
TOP_K_FINAL = 8
```

You can also increase the chunk size:

```python
CHUNK_CHARS = 1400
```

Tune these settings according to your documents and available system resources.

### Adjust Relevance Filtering

If relevant passages are frequently rejected, consider lowering the dense-retrieval threshold:

```python
MIN_DENSE_SCORE = 0.2
```

The appropriate threshold depends on the embedding model, retrieval implementation, and document collection. Lower thresholds may also introduce irrelevant passages.

## Important Notes

* **Re-index after configuration changes:** If you update chunking settings or the embedding model, remove and re-add the affected documents so the index matches the current configuration.

* **Update from older versions:** If chunking behavior changes between versions, rebuild the index by deleting the `data/` directory or removing and re-importing the documents through the application.

* **Scanned PDFs:** Image-only PDFs require OCR before their text can be retrieved. For example:

  ```bash
  ocrmypdf input.pdf output.pdf
  ```

* **Model limitations:** Local language models can misinterpret technical details or overlook relevant passages. Review citations before relying on answers.

## Optional Packaging

To experiment with packaging the application as a desktop executable, install PyInstaller:

```bash
python -m pip install pyinstaller
pyinstaller --windowed app.py
```

Additional packaging configuration may be required depending on the GUI framework, bundled assets, model dependencies, and target operating system.

## Project Goal

DocuChat aims to make private document collections easier to search, understand, and verify. By combining local language models with hybrid retrieval and source-level citations, it provides a practical way to explore technical manuals, reports, research papers, and other documents while keeping processing on the user's machine.

**DocuChat — Ask your documents. Verify every answer.**
