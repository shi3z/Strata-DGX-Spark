# Benchmark: NVIDIA GB10 (DGX Spark, Arm64), IQ3_XXS with images on

Measured on 2026-10-04 on one ASUS GX10 (a GB10 machine, the same chip as the DGX Spark) by the owner of this fork. This is the first run of
Strata on an Arm64 machine: the Arm64 port is the commit that adds this folder's parent change (`324b0ef`,
"Arm64 / NVIDIA GB10 (DGX Spark) support").

Decode was **37-42 tok/s at every prompt length from 1.6K to 75K tokens**. Prompts of 6K tokens and more were read
at **1,196-1,338 tok/s**. Prompts under 2K tokens were read at only **56-61 tok/s**. Each row is one request
(no repeats), so differences of a few tok/s between rows are within what one run can show. The prompts are
synthetic (random words from a 20-word list, then "Write a short story about a robot learning to paint."), with
greedy decoding and a 128-token output cap. They do not establish answer quality or speed on other workloads.

## Hardware and software

- **GPU:** NVIDIA GB10 (compute capability 12.1), driver 580.159.03, CUDA 13.0. The GPU shares one memory pool
  with the CPU (121 GiB, nvidia-smi reports no VRAM size).
- **CPU:** 20 cores, Arm Cortex-X925 and Cortex-A725 (NEON, dotprod, i8mm, SVE2; Strata uses NEON through
  ggml-cpu). The engine used 19 expert-pool workers plus its host thread.
- **OS:** Linux 6.17.0-1021-nvidia (aarch64). NVMe SSD.
- **Other services:** the machine was in normal use (about 75 GB of RAM was in use by other programs before the
  model loaded, among them a SearXNG server). It was not isolated.
- **Engine:** 0.1.38 built on this machine from the Arm64 port with `-DCMAKE_CUDA_ARCHITECTURES=121`, llama.cpp
  at the pinned 3cf0325.

## Model and configuration

- Qwen3.8-Flash-Next IQ3_XXS (setup's `--family qwen --model IQ3_XXS --vision gpu`), 39.97 GiB of experts in RAM.
- Context 131,072 tokens, 8-bit KV cache; KV streaming on (32,768 of 131,072 cells per layer in VRAM, the rest
  in pinned RAM). Setup counts a quarter of the RAM as VRAM on this machine (32 GB at most), so the GPU's expert
  cache held 619 experts (1.02 GiB).
- Speculative decoding (the MTP draft layer) on; `reasoning_effort: none`.

## Results

| Prompt tokens | Prefill (tok/s) | Decode (tok/s) | Wall time (s) |
| ---: | ---: | ---: | ---: |
| 431 | 56.1 | 15.3 | 16.1 |
| 1,582 | 61.0 | 41.7 | 29.0 |
| 6,190 | 1,195.7 | 37.3 | 8.6 |
| 12,334 | 1,293.6 | 39.6 | 12.8 |
| 24,622 | 1,331.9 | 38.6 | 21.9 |
| 49,199 | 1,337.6 | 38.2 | 40.3 |
| 75,047 | 1,328.6 | 39.0 | 60.0 |

Every row generated 128 tokens. Raw numbers: `results.json`. The script: `bench.py` (needs a running server on
port 8081; run it as `python3 bench.py`).

## What the numbers do and do not show

- **The first row's decode (15.3 tok/s)** is the first request after a 16-token warm-up; it is not repeated. It is
  probably the engine still warming up, but this was not checked.
- **Short prompts are read about 20x slower** than long ones (56-61 tok/s under 2K tokens, over 1,190 tok/s from
  6K). The engine log of the 18- and 87-token requests shows the same rate (54 and 83 tok/s). Why was not
  investigated; the prompt path may only use the GPU for longer prompts, so short ones may go through the CPU
  experts, which on Arm64 run on ggml-cpu's NEON code. That is a guess.
- **Decode does not slow down with the prompt** up to 75K tokens. A request with a 100,000-token prompt was
  planned but the run stopped at 75K, so nothing is measured above that.
- **No comparison** with an x86 PC or with another Arm64 build was made, and the Arm64 build has no tuned Arm
  kernels yet (the AVX-512 / AVX-2 multi-token kernels are not built here).
