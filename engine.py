"""Unified in-process (llama-cpp-python / fastembed) and Ollama fallback engine.

When GGUF files are present in the models/ folder, the application runs 100% in-process
with zero external prerequisites, servers, or internet connection.
If local GGUF files are not present, it gracefully connects to the local Ollama service.
"""
import os
import re
from pathlib import Path

import numpy as np

import config

# Lazy singletons for local models
_LOCAL_CHAT_LLM = None
_LOCAL_EMBED_LLM = None
_FASTEMBED_MODEL = None


def is_in_process_available() -> bool:
    """Returns True if local GGUF models exist in models/ folder."""
    return config.LOCAL_CHAT_GGUF.exists()


def get_backend_name() -> str:
    if is_in_process_available():
        return f"In-Process GGUF ({config.LOCAL_CHAT_GGUF.name})"
    return f"Ollama ({config.CHAT_MODEL})"


# ------------------------------------------------------------------ Chat Model
def _get_local_chat_llm():
    global _LOCAL_CHAT_LLM
    if _LOCAL_CHAT_LLM is None:
        try:
            from llama_cpp import Llama
            _LOCAL_CHAT_LLM = Llama(
                model_path=str(config.LOCAL_CHAT_GGUF),
                n_ctx=config.NUM_CTX,
                n_threads=max(1, os.cpu_count() - 1) if os.cpu_count() else 4,
                verbose=False
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load in-process model {config.LOCAL_CHAT_GGUF.name}: {e}\n"
                "Ensure llama-cpp-python is installed."
            )
    return _LOCAL_CHAT_LLM


def chat_stream(messages, model=None, temperature=0.1, num_ctx=None, should_stop=lambda: False):
    """Streams response chunks from either in-process Llama or Ollama."""
    num_ctx = num_ctx or config.NUM_CTX
    if is_in_process_available():
        llm = _get_local_chat_llm()
        res = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            stream=True
        )
        for chunk in res:
            if should_stop():
                break
            delta = chunk["choices"][0].get("delta", {})
            piece = delta.get("content", "")
            if piece:
                yield piece
    else:
        import ollama
        client = ollama.Client(host=config.OLLAMA_HOST)
        target_model = model or config.CHAT_MODEL
        for chunk in client.chat(model=target_model, messages=messages, stream=True,
                                 options={"temperature": temperature, "num_ctx": num_ctx}):
            if should_stop():
                break
            yield chunk["message"]["content"]


def chat_complete(messages, model=None, temperature=0, num_ctx=None, num_predict=200):
    """Non-streaming chat completion."""
    num_ctx = num_ctx or config.NUM_CTX
    if is_in_process_available():
        llm = _get_local_chat_llm()
        res = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=num_predict,
            stream=False
        )
        return res["choices"][0]["message"]["content"].strip()
    else:
        import ollama
        client = ollama.Client(host=config.OLLAMA_HOST)
        target_model = model or config.CHAT_MODEL
        r = client.chat(model=target_model, messages=messages,
                        options={"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict})
        return r["message"]["content"].strip()


# ------------------------------------------------------------------ Embedding Model
def _get_local_embed_llm():
    global _LOCAL_EMBED_LLM
    if _LOCAL_EMBED_LLM is None and config.LOCAL_EMBED_GGUF.exists():
        try:
            from llama_cpp import Llama
            _LOCAL_EMBED_LLM = Llama(
                model_path=str(config.LOCAL_EMBED_GGUF),
                embedding=True,
                verbose=False
            )
        except Exception:
            _LOCAL_EMBED_LLM = None
    return _LOCAL_EMBED_LLM


def embed(texts: list[str], prefix: str = "") -> np.ndarray:
    """Computes normalized embeddings using in-process GGUF/fastembed, or Ollama fallback."""
    full_texts = [prefix + t for t in texts]

    # 1. Try local GGUF embedding via llama-cpp
    embed_llm = _get_local_embed_llm()
    if embed_llm is not None:
        raw_embs = []
        for text in full_texts:
            res = embed_llm.create_embedding(input=text)
            raw_embs.append(res["data"][0]["embedding"])
        m = np.array(raw_embs, dtype=np.float32)
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)

    # 2. Try fastembed if installed
    global _FASTEMBED_MODEL
    try:
        if _FASTEMBED_MODEL is None:
            from fastembed import TextEmbedding
            _FASTEMBED_MODEL = TextEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5")
        embs = list(_FASTEMBED_MODEL.embed(full_texts))
        m = np.array(embs, dtype=np.float32)
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    except Exception:
        pass

    # 3. Fallback to Ollama client
    import ollama
    client = ollama.Client(host=config.OLLAMA_HOST)
    r = client.embed(model=config.EMBED_MODEL, input=full_texts)
    m = np.array(r["embeddings"], dtype=np.float32)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
