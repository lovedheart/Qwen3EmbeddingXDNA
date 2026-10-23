# Convert Qwen3-Embedding-0.6B (HF safetensors, bf16) -> Q8_0 GGUF.
# Tensor naming follows llama.cpp's Qwen3 convention (identical to the
# jina-v5 GGUFs), so scripts/gguf_embed.py serves either model.
# Usage: python scripts/safetensors_to_gguf_q80.py [OUT.gguf]
import glob
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_ref import q8_0_pack, QK8_0, BLOCK_B

SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/*"))[0]
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/models/qwen3-embedding/qwen3-embedding-0.6b-Q8_0.gguf")
ALIGN = 32

MAP2D = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
         "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
         "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
         "mlp.down_proj": "ffn_down"}
MAP1D = {"input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm",
         "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm"}
F32, Q8_0 = 0, 8


def load_bf16(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
        blob = f.read()
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        assert v["dtype"] == "BF16", (k, v["dtype"])
        u = np.frombuffer(blob[a:b], np.uint16).astype(np.uint32) << 16
        out[k] = u.view(np.float32).reshape(v["shape"]).astype(np.float64)
    return out


def wstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def kv(kind, k, v):
    if kind == "s":
        return wstr(k) + struct.pack("<I", 8) + wstr(v)
    if kind == "u":
        return wstr(k) + struct.pack("<II", 4, int(v))
    return wstr(k) + struct.pack("<If", 6, float(v))


def pack_q8(w):
    """[out,in] -> Q8_0 block bytes (34 B per 32 weights along in)."""
    assert w.shape[1] % QK8_0 == 0
    qs, d16 = q8_0_pack(w)
    buf = np.empty((qs.size // QK8_0, BLOCK_B), np.uint8)
    buf[:, :2] = d16.reshape(-1, 1).view(np.uint16).view(np.uint8)
    buf[:, 2:] = qs.reshape(-1, QK8_0).view(np.uint8)
    return buf.tobytes()


def main():
    sd = load_bf16(os.path.join(SNAP, "model.safetensors"))
    cfg = json.load(open(os.path.join(SNAP, "config.json")))
    meta = [("general.architecture", "s", "qwen3"),
            ("general.name", "s", "Qwen3-Embedding-0.6B"),
            ("general.file_type", "u", 7),
            ("qwen3.block_count", "u", cfg["num_hidden_layers"]),
            ("qwen3.embedding_length", "u", cfg["hidden_size"]),
            ("qwen3.feed_forward_length", "u", cfg["intermediate_size"]),
            ("qwen3.attention.head_count", "u", cfg["num_attention_heads"]),
            ("qwen3.attention.head_count_kv", "u", cfg["num_key_value_heads"]),
            ("qwen3.rope.freq_base", "f", cfg["rope_theta"]),
            ("qwen3.attention.layer_norm_rms_epsilon", "f", cfg["rms_norm_eps"]),
            ("qwen3.attention.key_length", "u", cfg["head_dim"]),
            ("qwen3.attention.value_length", "u", cfg["head_dim"])]
    tensors, off = [], 0

    def add(name, dt, dims, data):
        nonlocal off
        off = (off + ALIGN - 1) & -ALIGN      # ggml aligns each tensor
        tensors.append((name, dt, tuple(dims), data, off))
        off += len(data)

    add("token_embd.weight", Q8_0,
        (cfg["vocab_size"], cfg["hidden_size"]),
        pack_q8(sd["embed_tokens.weight"]))
    for l in range(cfg["num_hidden_layers"]):
        pre = f"layers.{l}."
        for hf, gg in MAP2D.items():
            w = sd[pre + hf + ".weight"]
            add(f"blk.{l}.{gg}.weight", Q8_0, w.shape, pack_q8(w))
        for hf, gg in MAP1D.items():
            v = sd[pre + hf + ".weight"]
            add(f"blk.{l}.{gg}.weight", F32, v.shape,
                v.astype(np.float32).tobytes())
    add("output_norm.weight", F32, (cfg["hidden_size"],),
        sd["norm.weight"].astype(np.float32).tobytes())

    body = b"".join(kv(t, k, v) for k, t, v in meta)
    tinfo = b"".join(wstr(n) + struct.pack("<I", len(d)) +
                     struct.pack(f"<{len(d)}Q", *reversed([int(x) for x in d])) +
                     struct.pack("<IQ", dt, o)   # GGML dims: fastest-varying first
                     for n, dt, d, _, o in tensors)
    hdr = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) +
           struct.pack("<Q", len(meta)) + body + tinfo)
    pad = (-len(hdr)) % ALIGN
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(hdr + b"\0" * pad)
        for i, (_, _, _, data, o) in enumerate(tensors):
            assert f.tell() - (len(hdr) + pad) == o, (i, f.tell(), o)
            f.write(data)
            if i + 1 < len(tensors):
                f.write(b"\0" * ((-len(data)) % ALIGN))
    print(f"wrote {OUT} tensors={len(tensors)} size={os.path.getsize(OUT)}")


if __name__ == "__main__":
    main()
