# Qwen3EmbeddingXDNA

Runs **Qwen3-Embedding-0.6B entirely on the Phoenix XDNA-1 NPU** (AIE-ML,
8 MAC tiles) through IRON / mlir-aie — every weight matmul happens on the
NPU; attention, RoPE and norms run host-side in float64. Ships an
**OpenAI-compatible `/v1/embeddings` server**, drop-in shape-compatible with
the ROCm vLLM container on `:8999`.

**Supported models — Q8_0, tested:**

| model | path | tested result |
|---|---|---|
| Qwen3-Embedding-0.6B | full model on NPU (W8A16 engine) | cos 0.9960–0.9967 vs fp64, live server `:8996` |
| Qwen3-Embedding-0.6B (Q8_0 GGUF) | generic GGUF engine, host f64; parity vs vLLM `:8993` | agree-cos ≥ 0.9994, identical NN retrieval |
| jina-embeddings-v5-text-small (0.6B Q8_0 GGUF) | Q8_0 forward + parity harness vs vLLM `:8999` | agree-cos ≥ 0.9998, identical NN retrieval |

All can run **at the same time** — one process per model, one port each (the
`npu` backend is the exception: it owns the NPU device). Both share the
identical 28-layer Qwen3 backbone, so the NPU design geometry transfers 1:1
(jina differs only in vocab 151936 and prompt/pooling prep; its NPU-engine
wiring would need a vocab-offset swap + device validation).
Chinese version: [README.zh-CN.md](README.zh-CN.md).

This is a dedicated single-model runtime ("Route B"), forked from
`~/projects/iron-moe`. The long-term plan and running status notes live in
ReMe: `digest/tasks/qwen3-embed-npu-dedicated-plan.md`.

## Quantization scheme — W8A16

Weights are **int8, per output channel** (`d[n] = max|w[n,:]| / 127`).
Activations are 16-bit-equivalent, carried as a **hi/lo int8 pair** (13-bit
effective, `QACT = 8191`) so the NPU's int8×int8→int32 MACs can be used:

```
acc = 256 * (a_hi @ W) + (a_lo @ W)          # int32, on-tile
fold (256*acc) * s[m] * d[n] once per row-group per K-chunk, drain-ADD to f32
```

Accuracy ladder (cos vs float64 reference embeddings):

| scheme | cos |
|---|---|
| Q8_0 (GGUF) | 0.9997 |
| **this W8A16, full model on NPU** | **0.9960 – 0.9967** |
| int8-only activations | 0.989 (rejected) |

The accepted bar is "Q8_0-equivalent band" (≥ 0.995).

## Layout

| path | role |
|---|---|
| `kernels/i8gemm_w8a16.cc` | **shipping kernel** — W8A16 GEMM, K-chunked, column-passed |
| `kernels/i8gemm.cc`, `kernels/null.cc` | P1 unit-GEMM kernel, health-probe null kernel |
| `embed.py` | host float64 reference forward (`EmbedEngine`), last-token + L2 pooling |
| `scripts/p4_npu_engine.py` | **device engine** `NpuEmbed`: 5 jitted designs, per-weight stream templates |
| `embed_server.py` | OpenAI-compat server; `EMBED_BACKEND=npu` routes to `NpuEmbed` |
| `scripts/w8a16_layer.py` | per-shape device gate (the only way new shapes reach hardware) |
| `scripts/w8a16_model.py` | device-free mirror model — **run before any device test** |
| `scripts/bench_null.py` | health probe (null stream, ~0.26 ms = healthy NPU) |
| `gguf_read.py` | GGUF reader (metadata + tensor infos + Q8_0 payloads) |
| `quant_ref.py` / `host_w.py` | Q8_0 oracle; int8 weight shredding for the stream |
| `scripts/gguf_embed.py` | generic Qwen3 Q8_0 GGUF engine (config from GGUF metadata; serves qwen3-embedding and jina-v5) |
| `scripts/safetensors_to_gguf_q80.py` | HF safetensors -> Q8_0 GGUF converter (llama.cpp tensor naming) |
| `scripts/jina_q80_eval.py`, `jina_fp64_ref.py` | jina-v5 Q8_0 vs vLLM parity harness |
| `scripts/run.sh` | activates `ironenv314` and runs a script with the AI toolchain |

## Running

First health-check the NPU (expect ~0.26 ms, no errors; skip for the
`host`/`jina` backends which don't touch the device):

```bash
bash scripts/run.sh scripts/bench_null.py 8 4 1
```

Launch the embedding server — pick a backend via `EMBED_BACKEND`
(`run.sh` activates the `ironenv314` toolchain itself):

| backend | model / engine | example |
|---|---|---|
| `npu` | Qwen3-Embedding-0.6B, **full forward on the NPU** (`p4_npu_engine.NpuEmbed`) | `EMBED_BACKEND=npu bash scripts/run.sh embed_server.py --port 8996` |
| `jina` | jina-v5-text-small Q8_0 GGUF, host f64 forward (`jina_q80_eval.Jina`) | `EMBED_BACKEND=jina bash scripts/run.sh embed_server.py --port 8995` |
| `gguf` | any Qwen3-architecture Q8_0 GGUF — **both** Qwen3-Embedding-0.6B and jina-v5 (host f64, `gguf_embed.GgufEmbed`; pass `--tokenizer` when the GGUF ships its own `tokenizer.json`, jina's must differ). EOS handling follows the reference served for each model: Qwen3-Embedding appends EOS (151645) before last-token pooling, jina-v5 raw strings do not | `EMBED_BACKEND=gguf bash scripts/run.sh embed_server.py --port 8994 --gguf <path.gguf> [--tokenizer <tokenizer.json>]` |
| `host` | Qwen3-Embedding-0.6B, host float64 reference (`embed.EmbedEngine`) | `EMBED_BACKEND=host bash scripts/run.sh embed_server.py --port 8990` |

`npu` jits 5 designs on first launch (~8 s), then serves. Default port is
8996 if `--port` is omitted. Background variant (keeps running after logout):

```bash
EMBED_BACKEND=npu nohup bash scripts/run.sh embed_server.py --port 8996 \
    > /tmp/srv8996.log 2>&1 &
```

Use it (OpenAI-compatible, model name must match `/v1/models`):

```bash
curl -s localhost:8996/health
curl -s localhost:8996/v1/embeddings -H 'content-type: application/json' \
     -d '{"input":["hello world"],"model":"Qwen3-Embedding-0.6B"}'
```

Kill the server before running any other `run.sh` device script — they
contend for the single device. Rebuild after kernel edits: purge
`~/.npu/cache` (JIT artifacts keyed by design, not source).

Weights: host reference reads `~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/*/`.

## Construction (building the Q8_0 model files)

Qwen3-Embedding GGUF — build it locally from the HF safetensors (llama.cpp
naming, Q8_0 = f16 scale + 32×int8 blocks; needs ~12 GB RAM transiently):

```bash
~/projects/mlir-aie/ironenv314/bin/python scripts/safetensors_to_gguf_q80.py
# -> ~/models/qwen3-embedding/qwen3-embedding-0.6b-Q8_0.gguf  (~633 MB)
```

jina-v5 GGUF — download instead of converting (the repo ships its own
tokenizer.json; the Qwen3 one appends a special token jina does not use):

```bash
# v5-small-retrieval-Q8_0.gguf -> ~/models/jina-v5/
```

Then run both simultaneously (example: 8994 = Qwen3, 8995 = jina):

```bash
EMBED_BACKEND=gguf nohup bash scripts/run.sh embed_server.py --port 8994 \
  --gguf ~/models/qwen3-embedding/qwen3-embedding-0.6b-Q8_0.gguf \
  > /tmp/srv8994.log 2>&1 &
EMBED_BACKEND=gguf nohup bash scripts/run.sh embed_server.py --port 8995 \
  --gguf ~/models/jina-v5/v5-small-retrieval-Q8_0.gguf \
  --tokenizer ~/models/jina-v5/tokenizer.json \
  > /tmp/srv8995.log 2>&1 &
```

`/v1/models` on each port reports the GGUF basename as the model id.

## Hardware safety rules (learned the expensive way)

Two ways to wedge the NPU, both recovered **only by a power cycle**:

1. **Per-tile L1 overflow** — resident pbuf + `g_outbuf` (M·RGW·4) + `g_d`
   must be ≤ 32768 B on every tile. Enforced by an assert in `w8a16_model.py`.
2. **Kernel hang** — any spinning kernel trips `ERT_CMD_STATE_TIMEOUT`, the
   teardown `aie2_suspend_fw` then fails (ret −32) and the SMU dies.

Therefore:

* **Only the allow-listed `(GW, GK, KC)` shapes in `scripts/w8a16_layer.py`
  go to the device** (q 2048×1024, kv 1024×1024, o 1024×2048, gu 1536×1024
  ×2 column passes, dn 1024×3072 K-chunked). The script asserts membership
  *before* touching hardware.
* **Mirror first**: every candidate shape must PASS `scripts/w8a16_model.py`
  (device-free) first — a mirror PASS with a device hang means a DMA edge
  case, not a math bug.
* **Small shapes only** on the device (stream ≤ ~4 MB, `MM=8`); no exotic
  or tiny probes (512×128-class shapes hang the shim).
* Static `.bss` per tile tops out ≈ 11.5 KB → `GW ≤ 2560` columns/tile,
  which is why gate/up run as 2 column passes.

Wedge symptom: `RuntimeError: DRM_IOCTL_AMDXDNA_GET_INFO IOCTL failed
(err=-22)` → reboot, then re-probe with `bench_null.py`.

## Design invariants

* Every drain ≤ `OB = 3456` B; activations stream as objects, never one big
  host tensor. Tile map `Tile(c%4, 2+c//8)`; `TPO = 52` weight bytes per obj.
* Stream layout per layer: obj0 meta (`i32 rg0 | f32 s[MM] | f32 d[GW]`),
  then per K-chunk: A objects (`ah`/`al` interleaved by row-block and
  K-tile) followed by weight objects. K-chunking keeps A ≤ 24 KB/tile
  (down_proj: `KC=1024, KCH=3`).
* Attention/RoPE use the **rotate-half** convention (`cat(-x2, x1)` over
  both halves). Interleaved-pair RoPE is silently wrong and compounds to
  ~0.96 agree-cos through 28 layers — check this first when debugging.

## Verification status

| stage | result |
|---|---|
| Null dispatch probe | 0.267 ms |
| Per-shape device gates (`w8a16_layer.py`) | cos 1.000000 all shapes: 1024² 0.75 ms, 2048×1024 1.10, 1536×1024 0.88, 1024×3072 1.70 ms |
| Full model on NPU (`p4_npu_engine.py`) | cos 0.9960–0.9967 vs fp64, 1.0–3.3 s/text (perf tuning pending) |
| Server `:8996` | `/health` ok, `/v1/embeddings` → (N, 1024), unit norms |
| Q8_0 GGUF parity vs vLLM `:8999` (jina-v5, 16 STS texts) | agree-cos ≥ 0.9998, identical NN retrieval, max sim-matrix diff 0.0028 |
| Q8_0 GGUF parity vs vLLM `:8993` (Qwen3-Embedding Q8_0, 16 STS texts) | agree-cos ≥ 0.9994 (mean 0.9996), identical NN retrieval, max sim-matrix diff 0.0074 |

## Toolchain

* Python: `~/projects/mlir-aie/ironenv314` (3.14); device runs via
  `bash scripts/run.sh <script>`.
* Kernel compile: `$SITE/llvm-aie/bin/clang++ -std=c++20 -Os -DNDEBUG
  -D__AIE_API_AIE_ADF_HPP__ --target=aie2-none-unknown-elf
  -DGW=… -DGK=… -DKC=… -DMM=8 -DOB=3456` with
  `SITE=/home/lovedheart/projects/mlir-aie/ironenv314/lib/python3.14/site-packages`.
* Reference GGUF source: `/home/lovedheart/qwentts.cpp`
  (`ggml/src/gguf.cpp`, `ggml-common.h` for Q8_0 = f16 d + 32×int8).
