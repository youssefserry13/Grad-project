"""S3 = S1 with only the model changed (qwen3-vl:8b-instruct), plus num_ctx 8192.

Runs the frozen ~/benchmark_v2/scripts/benchmark_qa_local.py unmodified (imported as a module):
same manifest, frozen P01 prompt, processed images, temperature 0, max tokens 1024, think off,
same record format. The only overrides here:
  * condition label "S3" (same config as S1: P01 + processed_path)
  * Ollama option num_ctx=8192 (S1 sent no num_ctx)
  * output dir (never writes into ~/benchmark_v2/results/raw_outputs)

    ~/benchmark_v2/.venv/bin/python run_S3.py --out <dir>      # -> <dir>/local_S3.json
"""
import argparse
import sys
from pathlib import Path

import requests

SCRIPTS = Path.home() / "benchmark_v2" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import benchmark_qa_local as b  # noqa: E402

MODEL = "qwen3-vl:8b-instruct"
NUM_CTX = 8192


def call_api(api_key, model, messages, settings):
    payload = {
        "model": model, "messages": messages, "stream": False, "think": False,
        "options": {"temperature": settings["temperature"], "num_predict": settings["max_tokens"],
                    "num_ctx": NUM_CTX},
    }
    return requests.post(f"{b.OLLAMA_URL}/api/chat", json=payload, timeout=b.REQUEST_TIMEOUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args, rest = ap.parse_known_args()
    b.RESULTS_DIR = Path(args.out)
    b.MODES["S3"] = dict(b.MODES["S1"], label="Local + Processed images (qwen3-vl:8b-instruct)")
    b.call_api = call_api
    sys.argv = [sys.argv[0], "--model", MODEL, "--mode", "S3", *rest]
    b.main()
