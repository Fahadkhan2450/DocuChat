"""Downloads GGUF model files into the models/ directory for standalone offline execution."""
import os
import sys
import urllib.request
from pathlib import Path

import config

MODELS = {
    "qwen2.5-3b-instruct-q4_k_m.gguf": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        "size": 2104932768,
        "description": "Chat & Reranking Model (Qwen 2.5 3B Instruct Q4_K_M)",
    },
    "nomic-embed-text-v1.5.Q4_K_M.gguf": {
        "url": "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf",
        "size": 84106624,
        "description": "Dense Semantic Embedding Model (Nomic Embed Text v1.5 Q4_K_M)",
    },
}


def download_file(url: str, dest_path: Path, expected_size: int, description: str):
    print(f"\nTarget: {description}")
    print(f"File:   {dest_path.name}")
    print(f"Path:   {dest_path}")

    if dest_path.exists():
        current_size = dest_path.stat().st_size
        if current_size == expected_size:
            print("Status: Already fully downloaded ✓")
            return
        elif current_size < expected_size:
            print(f"Status: Partial download found ({current_size / (1024*1024):.1f} MB / {expected_size / (1024*1024):.1f} MB). Resuming...")
        else:
            print("Status: File size mismatch. Re-downloading from scratch...")
            dest_path.unlink()

    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)

    existing_bytes = dest_path.stat().st_size if dest_path.exists() else 0
    if existing_bytes > 0:
        req.add_header("Range", f"bytes={existing_bytes}-")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "ab" if existing_bytes > 0 else "wb"

    try:
        with urllib.request.urlopen(req) as resp, open(dest_path, mode) as f:
            total_size = expected_size
            downloaded = existing_bytes
            block_size = 1024 * 1024  # 1MB buffer

            while True:
                buf = resp.read(block_size)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
                percent = (downloaded / total_size) * 100 if total_size else 0
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\rProgress: [{mb_down:.1f} MB / {mb_total:.1f} MB] ({percent:.1f}%)")
                sys.stdout.flush()

        print(f"\nFinished: {dest_path.name} saved successfully ✓")
    except Exception as e:
        print(f"\nDownload error: {e}")
        print("Tip: You can manually download the file and place it in the models/ folder.")
        raise


def main():
    print("=" * 65)
    print("DocuChat Standalone Model Downloader")
    print("=" * 65)
    print(f"Models directory: {config.MODELS_DIR.resolve()}\n")

    for filename, info in MODELS.items():
        dest = config.MODELS_DIR / filename
        download_file(info["url"], dest, info["size"], info["description"])

    print("\nAll required standalone model weights are in models/ ✓")
    print("You can now build the standalone executable or run app.py without Ollama.")


if __name__ == "__main__":
    main()
