"""Automated PyInstaller build script for DocuChat."""
import os
import subprocess
import sys
from pathlib import Path

import config


def build():
    print("=" * 65)
    print("DocuChat Standalone Executable Builder")
    print("=" * 65)

    # 1. Check models
    chat_model = config.LOCAL_CHAT_GGUF
    embed_model = config.LOCAL_EMBED_GGUF

    print(f"\nChecking models directory ({config.MODELS_DIR})...")
    missing = []
    if not chat_model.exists():
        missing.append(chat_model.name)
    if not embed_model.exists():
        missing.append(embed_model.name)

    if missing:
        print(f"Warning: The following GGUF models are missing in models/:")
        for m in missing:
            print(f"  • {m}")
        print("\nTo build a 100% offline standalone package without Ollama, run first:")
        print("    python download_models.py")
        print("\nDo you want to proceed anyway with Ollama-only mode? [y/N]: ", end="")
        choice = input().strip().lower()
        if choice not in ("y", "yes"):
            sys.exit(1)

    # 2. Check PyInstaller
    try:
        import PyInstaller
    except ImportError:
        print("\nInstalling PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # 3. Run PyInstaller build
    print("\nStarting PyInstaller build with DocuChat.spec...")
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "DocuChat.spec"]
    subprocess.check_call(cmd)

    dist_dir = Path("dist") / "DocuChat"
    print("\n" + "=" * 65)
    print("Build Complete!")
    print("=" * 65)
    print(f"Standalone application folder created at:\n  {dist_dir.resolve()}")
    print("\nNext step for Windows installer:")
    print("  1. Open 'installer.iss' in Inno Setup Compiler (or run: iscc installer.iss).")
    print("  2. It will generate a single-file 'DocuChat-Setup-v2.0.exe' inside 'dist-installer/'.")
    print("  3. Give that setup file to any user — it runs without internet or Ollama!")


if __name__ == "__main__":
    build()
