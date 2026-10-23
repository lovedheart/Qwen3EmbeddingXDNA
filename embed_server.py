# OpenAI-compatible /v1/embeddings server (P4b) for Qwen3-Embedding-0.6B.
# Parity target: vllm-embedding-rocm10 on :8999 (same JSON shape).
# Backend: EMBED_BACKEND=npu (when the IRON runtime lands) | host (float64).
import argparse
import time

import numpy as np
from flask import Flask, jsonify, request

from embed import EmbedEngine

app = Flask(__name__)
ENG = None
MODEL_ID = "Qwen3-Embedding-0.6B"


@app.get("/v1/models")
def models():
    return jsonify(object="list", data=[dict(
        id=MODEL_ID, object="model", created=int(time.time()),
        owned_by="qwen3embeddingxdna")])


@app.get("/health")
def health():
    return "ok"


@app.post("/v1/embeddings")
def embeddings():
    req = request.get_json(force=True)
    inp = req.get("input", "")
    if isinstance(inp, str):
        inp = [inp]
    t0 = time.time()
    embs = ENG.embed(inp)
    data = [dict(object="embedding", index=i, embedding=e.tolist())
            for i, e in enumerate(embs)]
    return jsonify(object="list", data=data, model=req.get("model", MODEL_ID),
                   usage=dict(prompt_tokens=0, total_tokens=0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8996)
    ap.add_argument("--gguf", default=None,
                    help="path to a Qwen3-style Q8_0 GGUF (backend=gguf)")
    ap.add_argument("--tokenizer", default=None,
                    help="HF tokenizer.json for the GGUF (default: Qwen3-Emb snapshot)")
    a = ap.parse_args()
    import os
    import sys
    backend = os.environ.get("EMBED_BACKEND", "host")
    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "scripts"))
    if backend == "npu":
        from p4_npu_engine import NpuEmbed
        ENG = NpuEmbed()
    elif backend == "jina":
        from jina_q80_eval import Jina
        ENG = Jina()
        globals()["MODEL_ID"] = "jina-v5-text-small-retrieval"
    elif backend == "gguf":
        import glob
        from gguf_embed import GgufEmbed
        if not a.gguf:
            raise SystemExit("--gguf PATH required for EMBED_BACKEND=gguf")
        tok = a.tokenizer or glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B"
            "/snapshots/*/tokenizer.json"))[0]
        ENG = GgufEmbed(a.gguf, tok)
        globals()["MODEL_ID"] = os.path.basename(a.gguf)
    elif backend != "host":
        raise SystemExit(f"unknown backend {backend}")
    else:
        ENG = EmbedEngine()
    app.run(a.host, a.port, threaded=False)
