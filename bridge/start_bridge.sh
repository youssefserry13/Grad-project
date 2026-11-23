#!/usr/bin/env bash
# Start the Nova bridge (port 5001) with the CUDA libraries Whisper needs.
#   ./start_bridge.sh                 # single model (qwen3-vl), the demo setup
#   MODEL_MODE=dual ./start_bridge.sh # old qwen3 + qwen2.5vl fallback
#   STREAM_SILENCE_S=1.5 ./start_bridge.sh   # longer pause before an ESP utterance is closed
cd "$(dirname "$0")" || exit 1
SITE=$(ls -d venv/lib/python3*/site-packages)
export LD_LIBRARY_PATH="$PWD/$SITE/nvidia/cublas/lib:$PWD/$SITE/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1
exec venv/bin/python OllamaBridge.py
