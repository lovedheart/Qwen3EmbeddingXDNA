# Generic GGUF (Q8_0) embedding engine: last-token pooling + L2, host f64
# forward. Works with any llama.cpp-style Qwen3 GGUF (jina-v5, qwen3-emb).
# rope theta / dims are read from the GGUF metadata; tokenizer comes from the
# HF snapshot tokenizer.json (GGUF carries no tokenizer in this project).
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gguf_read
from quant_ref import q8_0_unpack


class GgufEmbed:
    def __init__(self, gguf_path, tokenizer_json):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(tokenizer_json)
        g = gguf_read.GGUF(gguf_path)
        self.g = g
        m = g.meta
        self.H = int(m["qwen3.embedding_length"])
        self.I = int(m["qwen3.feed_forward_length"])
        self.L = int(m["qwen3.block_count"])
        self.qh = int(m["qwen3.attention.head_count"])
        self.kvh = int(m["qwen3.attention.head_count_kv"])
        self.hd = int(m["qwen3.attention.key_length"])
        self.eps = float(m["qwen3.attention.layer_norm_rms_epsilon"])
        self.theta = float(m.get("qwen3.rope.freq_base", 1e6))
        self.gqa = self.qh // self.kvh
        self.inv = 1.0 / self.theta ** (np.arange(0, self.hd, 2, np.float64)
                                        / self.hd)
        self._wcache = {}
        self.emb = self.wq("token_embd.weight")
        self.on = self._f32("output_norm.weight")

    def _f32(self, n):
        dt, dims, off = self.g.tensors[n]
        assert dt == 0, (n, dt)
        r = int(np.prod([int(d) for d in dims]))
        a = np.frombuffer(self.g.mm, np.float32, r, self.g.data0 + off)
        return a.reshape(dims).astype(np.float64)

    def wq(self, n):
        if n not in self._wcache:
            qs, d16 = self.g.q8_0(n)
            self._wcache[n] = q8_0_unpack(qs, d16)
        return self._wcache[n]

    def _rope(self, x, pos):
        # HF Qwen3 rotate-half (NOT interleaved pairs).
        ang = pos[:, None, None].astype(np.float64) * self.inv[None, None, :]
        cos = np.concatenate([np.cos(ang)] * 2, -1)
        sin = np.concatenate([np.sin(ang)] * 2, -1)
        x1, x2 = x[..., :self.hd // 2], x[..., self.hd // 2:]
        return x * cos + np.concatenate([-x2, x1], -1) * sin

    def forward(self, ids):
        H, hd, eps = self.H, self.hd, self.eps
        x = self.emb[np.asarray(ids)]
        T = len(ids)
        pos = np.arange(T)
        causal = np.triu(np.full((1, T, T), -np.inf), 1)
        for l in range(self.L):
            p = f"blk.{l}."
            h = x
            xr = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
            xr = xr * self._f32(p + "attn_norm.weight")
            q = (xr @ self.wq(p + "attn_q.weight").T).reshape(T, self.qh, hd)
            k = (xr @ self.wq(p + "attn_k.weight").T).reshape(T, self.kvh, hd)
            v = (xr @ self.wq(p + "attn_v.weight").T).reshape(T, self.kvh, hd)
            qn = self._f32(p + "attn_q_norm.weight")
            kn = self._f32(p + "attn_k_norm.weight")
            q = q / np.sqrt((q * q).mean(-1, keepdims=True) + eps) * qn
            k = k / np.sqrt((k * k).mean(-1, keepdims=True) + eps) * kn
            q, k = self._rope(q, pos), self._rope(k, pos)
            k = np.repeat(k, self.gqa, 1)
            v = np.repeat(v, self.gqa, 1)
            s = (q.transpose(1, 0, 2) @ k.transpose(1, 2, 0)) / np.sqrt(hd)
            s = s + causal
            s -= s.max(-1, keepdims=True)
            pr = np.exp(s)
            pr /= pr.sum(-1, keepdims=True)
            o = (pr @ v.transpose(1, 0, 2)).transpose(1, 0, 2)
            o = o.reshape(T, self.qh * hd)      # qh*hd (=2048) != H (=1024)
            x = h + o @ self.wq(p + "attn_output.weight").T
            xr = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
            xr = xr * self._f32(p + "ffn_norm.weight")
            g = xr @ self.wq(p + "ffn_gate.weight").T
            u = xr @ self.wq(p + "ffn_up.weight").T
            x = x + (g / (1.0 + np.exp(-g)) * u) @ self.wq(p + "ffn_down.weight").T
        x = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * self.on
        e = x[-1]
        return e / np.linalg.norm(e)

    def encode(self, texts, eos=False):
        # matches vLLM raw-string handling (no EOS append) by default
        return [self.tok.encode(t).ids + ([int(self.g.meta["tokenizer.ggml.eos_token_id"])] if eos else [])
                for t in texts]

    def embed(self, texts, eos=False):
        return np.stack([self.forward(i) for i in self.encode(texts, eos)])

    def reload_weights(self):
        self._wcache.clear()


if __name__ == "__main__":
    e = GgufEmbed(sys.argv[1], sys.argv[2])
    v = e.embed(["hello world"])[0]
    print("dim", v.shape, "norm", round(float(np.linalg.norm(v)), 4))
