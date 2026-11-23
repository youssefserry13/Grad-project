# qwen3-vl:8b-instruct as the single Nova model (test 2026-11-23)

Can qwen3-vl:8b-instruct replace both qwen3:8b (text/voice) and qwen2.5vl:7b (images), with no swaps?
Frozen benchmark files/prompts untouched. Accuracy = UNOFFICIAL auto-grade (qwen3:8b judge, temp 0);
R1 official score stays 95%. Script: `rag/qwen3vl_test.py`; full numbers: `qwen3vl_report.json`.

| | Text R1 qwen3:8b | Text R1 qwen3-vl | Image B2 qwen2.5vl | Image B2 qwen3-vl | Voice qwen3:8b | Voice qwen3-vl |
|---|---|---|---|---|---|---|
| Accuracy (C/P/W) | 91.7% (52/6/2) | 93.3% (53/6/1) | 94.2% (54/5/1) | 98.3% (58/2/0) | manual 5/5 | manual 5/5 |
| Gold page named | n/a (P03) | n/a | 58/60 | 56/60 | 4/5 | 3/5 |
| Honesty | - | - | 8/8 | 8/8 | trick 5/5, offline 5/7* | trick 5/5, offline 5/7* |
| Model time (median) | 1.41 s | 1.78 s | 3.41 s + ~28 s swap | 4.51 s, no swap | 1.12 s | 1.57 s |
| CPU/GPU split | 100% GPU | 14/86 (ctx 4096) | 14/86 (ctx 8192) | 14/86 (ctx 4096) | 100% GPU | 14/86 |
| Peak RAM used / swap | - | 11.1 GB / 3 MB | 11.1 GB / 488 MB | 11.7 GB / 488 MB | 4.5 GB / 30 MB | 11.1 GB / 3 MB |

\* "Set a timer" and "Send an email" were routed to the manual (reply "I couldn't find that in the manual"):
a bridge routing issue, same for both models.

VRAM fit (qwen3-vl alone and with Whisper large-v3-turbo on GPU + nomic-embed-text): ctx 8192 = 78% GPU,
ctx 4096 = 86% GPU, both with and without Whisper; never 100% GPU. Largest prompts: text ~2.1k tokens,
image ~3.4k tokens (1024 px), longest answer 82 tokens, so ctx 4096 is enough.
"RAM used" here is MemTotal - MemAvailable sampled during the runs.

Settings: temperature 0, max tokens 1024, hybrid retrieval top-5, image 1024 px, B2 prompt NOVA_IMG_B2_v2,
qwen2.5vl baseline rerun with the same v2 prompt (its 6 earlier v2 rows matched exactly). Ollama 0.40.16.
