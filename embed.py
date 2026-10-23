# Qwen3-Embedding-0.6B host runtime (P4): tokenizer + dense forward in
# float64 (reference; NPU path plugs into the same EmbedEngine).
# Pooling = last token + L2 (1_Pooling/config.json), EOS appended.
import glob
import json
import os
import numpy as np

_SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/*"))[0]
CFG = dict(H=1024, I=3072, L=28, qh=16, kvh=8, hd=128, rope_theta=1e6,
           eps=1e-6, vocab=151669, eos=151645)


def _load_safetensors(path):
    import struct
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            f.seek(base + a)
            raw = f.read(b - a)
            if v["dtype"] == "BF16":
                u = np.frombuffer(raw, np.uint16).astype(np.uint32) << 16
                out[k] = u.view(np.float32).reshape(v["shape"])
            else:
                dt = {"F32": np.float32, "F16": np.float16}[v["dtype"]]
                out[k] = np.frombuffer(raw, dt).reshape(v["shape"]).astype(np.float32)
    return out


class EmbedEngine:
    def __init__(self, weights=None):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(os.path.join(_SNAP, "tokenizer.json"))
        self.tok.enable_truncation(8192)
        self.w = weights if weights is not None else _load_safetensors(
            os.path.join(_SNAP, "model.safetensors"))
        c = CFG
        d = c["hd"]
        inv = 1.0 / c["rope_theta"] ** (np.arange(0, d, 2, np.float64) / d)
        self._inv = inv
        self._cw = {k: v.astype(np.float64) for k, v in self.w.items()
                    if v.ndim == 1 and "norm" in k}
        self._emb = self.w["embed_tokens.weight"].astype(np.float64)

    def encode(self, texts):
        return [self.tok.encode(t).ids + [CFG["eos"]] for t in texts]

    def _rope(self, x, pos):
        # x [T, heads, hd], pos [T]. HF Qwen3 uses rotate-half (NOT interleaved).
        ang = pos[:, None, None].astype(np.float64) * self._inv[None, None, :]  # [T,1,hd/2]
        cos = np.concatenate([np.cos(ang)] * 2, -1)
        sin = np.concatenate([np.sin(ang)] * 2, -1)
        x1, x2 = x[..., :64], x[..., 64:]
        return x * cos + np.concatenate([-x2, x1], -1) * sin

    def forward(self, ids, out_hidden=False):
        c = CFG
        x = self._emb[np.asarray(ids)]                     # [T,H]
        T = len(ids)
        pos = np.arange(T)
        g = lambda n, s: self.w[f"layers.{n}.{s}"].astype(np.float64)
        causal = np.triu(np.full((1, T, T), -np.inf), 1)
        for l in range(c["L"]):
            h = x
            xr = x / np.sqrt((x * x).mean(-1, keepdims=True) + c["eps"])
            xr = xr * self._cw[f"layers.{l}.input_layernorm.weight"]
            qw, kw, vw = (g(l, "self_attn.q_proj.weight"),
                          g(l, "self_attn.k_proj.weight"),
                          g(l, "self_attn.v_proj.weight"))
            q = (xr @ qw.T).reshape(T, c["qh"], c["hd"])
            k = (xr @ kw.T).reshape(T, c["kvh"], c["hd"])
            v = (xr @ vw.T).reshape(T, c["kvh"], c["hd"])
            qn, kn = self._cw[f"layers.{l}.self_attn.q_norm.weight"], \
                     self._cw[f"layers.{l}.self_attn.k_norm.weight"]
            q = q / np.sqrt((q * q).mean(-1, keepdims=True) + c["eps"]) * qn
            k = k / np.sqrt((k * k).mean(-1, keepdims=True) + c["eps"]) * kn
            q, k = self._rope(q, pos), self._rope(k, pos)
            k = np.repeat(k, c["qh"] // c["kvh"], 1)
            v = np.repeat(v, c["qh"] // c["kvh"], 1)
            s = q.transpose(1, 0, 2) @ k.transpose(1, 2, 0) / np.sqrt(c["hd"]) + causal  # [qh,T,T]
            s -= s.max(-1, keepdims=True)
            p = np.exp(s)
            p /= p.sum(-1, keepdims=True)
            o = (p @ v.transpose(1, 0, 2)).transpose(1, 0, 2).reshape(T, c["qh"] * c["hd"])
            x = h + o @ g(l, "self_attn.o_proj.weight").T
            xr = x / np.sqrt((x * x).mean(-1, keepdims=True) + c["eps"])
            xr = xr * self._cw[f"layers.{l}.post_attention_layernorm.weight"]
            gu = xr @ g(l, "mlp.gate_proj.weight").T
            xu = xr @ g(l, "mlp.up_proj.weight").T
            x = x + (gu / (1.0 + np.exp(-gu)) * xu) @ g(l, "mlp.down_proj.weight").T
        x = x / np.sqrt((x * x).mean(-1, keepdims=True) + c["eps"])
        x = x * self._cw["norm.weight"]
        e = x[-1]                                          # last-token pool
        e = e / np.linalg.norm(e)
        return (e, x) if out_hidden else e

    def embed(self, texts):
        return np.stack([self.forward(ids) for ids in self.encode(texts)])


if __name__ == "__main__":
    import sys, time
    eng = EmbedEngine()
    t0 = time.time()
    e = eng.embed(["It is a lovely day outside.", "hello world"])[0]
    print(f"dim {e.shape[0]} first {e[:3]} ({time.time()-t0:.1f}s)")
