# Nova – Sovereign AI Desktop Assistant (grad project)

## Setup
- WSL2 Ubuntu, RTX 5060 8GB, Ollama in WSL. Check GPU with `nvidia-smi` + `ollama ps`.
- If nvidia-smi fails in WSL, the fix is updating the NVIDIA Windows driver.
- venv: ~/nova/bridge/venv. LD_LIBRARY_PATH must include nvidia-cublas-cu12 / nvidia-cudnn-cu12 lib dirs.
- Pin av==18.1.0 (av 19 breaks faster-whisper 1.2.1).

## Models
- qwen3:8b (text, think off), qwen2.5vl:7b (vision), nomic-embed-text (embeddings), Whisper large-v3-turbo.
- Only 8GB VRAM: never run two big models at once.

## Bridge
- bridge/OllamaBridge.py on port 5001, wake word "Nova" (fuzzy 0.85).

## Benchmark v2.0 (FROZEN)
- Never edit benchmark_manifest.xlsx, frozen_prompts.json, benchmark images, or results/raw_outputs/*.
- Settings: temperature 0, max tokens 1024. RAG uses prompt P03.
- Source text: universal_milling_machine_en.md (final). Build our own chunks from it.

## Rules
- Ask before installing packages or changing driver/WSL setup.
- Don't start Ollama jobs while a benchmark run is in progress.
