# jina-embeddings-v5-text-small-retrieval Q8_0 (GGUF) forward on host f64,
# vs the running vLLM reference on :8999 (same weights, fp16, ROCm).
# Qwen3 backbone (28L, H1024, I3072, 16q/8kv x hd128, rope 3.5e6), last-token
# pooling + L2 (1_Pooling lasttoken), EOS appended.
import json
import os
import struct
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gguf_read
from quant_ref import q8_0_unpack

GGUF_PATH = os.path.expanduser("~/models/jina-v5/v5-small-retrieval-Q8_0.gguf")
TOK_PATH = os.path.expanduser("~/models/jina-v5/tokenizer.json")
C = dict(H=1024, I=3072, L=28, qh=16, kvh=8, hd=128, theta=3.5e6, eps=1e-6,
         eos=151645)


class Jina:
    def __init__(self):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(TOK_PATH)
        self.tok.enable_truncation(8192)
        g = gguf_read.GGUF(GGUF_PATH)
        self.g = g
        self.emb = self.wq("token_embd.weight")
        self.on = self._f32("output_norm.weight")
        self.inv = 1.0 / C["theta"] ** (np.arange(0, C["hd"], 2, np.float64) / C["hd"])

    def _f32(self, n):
        """dtype 0 (F32) vector/tensor -> f64."""
        dt, dims, off = self.g.tensors[n]
        assert dt == 0, (n, dt)
        r = int(np.prod([int(d) for d in dims]))
        a = np.frombuffer(self.g.mm, np.float32, r, self.g.data0 + off)
        return a.reshape(dims).astype(np.float64)

    def wq(self, n):
        """Q8_0 weight -> dequantized f64 [out, in]."""
        qs, d16 = self.g.q8_0(n)
        return q8_0_unpack(qs, d16)

    def _rope(self, x, pos):
        # Qwen3 / llama rotate-half convention (NOT interleaved pairs).
        ang = pos[:, None, None].astype(np.float64) * self.inv[None, None, :]
        cos = np.concatenate([np.cos(ang)] * 2, -1)
        sin = np.concatenate([np.sin(ang)] * 2, -1)
        x1, x2 = x[..., :64], x[..., 64:]
        return x * cos + np.concatenate([-x2, x1], -1) * sin

    def forward(self, ids):
        x = self.emb[np.asarray(ids)]
        T = len(ids)
        pos = np.arange(T)
        causal = np.triu(np.full((1, T, T), -np.inf), 1)
        for l in range(C["L"]):
            p = f"blk.{l}."
            h = x
            xr = x / np.sqrt((x * x).mean(-1, keepdims=True) + C["eps"])
            xr = xr * self._f32(p + "attn_norm.weight")
            q = (xr @ self.wq(p + "attn_q.weight").T).reshape(T, C["qh"], C["hd"])
            k = (xr @ self.wq(p + "attn_k.weight").T).reshape(T, C["kvh"], C["hd"])
            v = (xr @ self.wq(p + "attn_v.weight").T).reshape(T, C["kvh"], C["hd"])
            qn = self._f32(p + "attn_q_norm.weight")
            kn = self._f32(p + "attn_k_norm.weight")
            q = q / np.sqrt((q * q).mean(-1, keepdims=True) + C["eps"]) * qn
            k = k / np.sqrt((k * k).mean(-1, keepdims=True) + C["eps"]) * kn
            q, k = self._rope(q, pos), self._rope(k, pos)
            k = np.repeat(k, 2, 1)
            v = np.repeat(v, 2, 1)
            s = q.transpose(1, 0, 2) @ k.transpose(1, 2, 0) / np.sqrt(C["hd"]) + causal
            s -= s.max(-1, keepdims=True)
            p_ = np.exp(s)
            p_ /= p_.sum(-1, keepdims=True)
            o = (p_ @ v.transpose(1, 0, 2)).transpose(1, 0, 2).reshape(T, C["qh"] * C["hd"])
            x = h + o @ self.wq(p + "attn_output.weight").T
            xr = x / np.sqrt((x * x).mean(-1, keepdims=True) + C["eps"])
            xr = xr * self._f32(p + "ffn_norm.weight")
            gu = xr @ self.wq(p + "ffn_gate.weight").T
            xu = xr @ self.wq(p + "ffn_up.weight").T
            x = x + (gu / (1.0 + np.exp(-gu)) * xu) @ self.wq(p + "ffn_down.weight").T
        x = x / np.sqrt((x * x).mean(-1, keepdims=True) + C["eps"])
        x = x * self.on
        e = x[-1]
        return e / np.linalg.norm(e)

    def embed(self, texts, eos=False):
        # vLLM (:8999) does NOT append EOS to raw input strings; match that.
        ids = [self.tok.encode(t).ids + ([C["eos"]] if eos else [])
               for t in texts]
        return np.stack([self.forward(i) for i in ids])


def remote(port, texts):
    mid = json.loads(urllib.request.urlopen(
        f"http://localhost:{port}/v1/models", timeout=10).read())["data"][0]["id"]
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/embeddings",
        data=json.dumps({"input": texts, "model": mid}).encode(),
        headers={"content-type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return np.array([x["embedding"] for x in sorted(d["data"], key=lambda x: x["index"])])


if __name__ == "__main__":
    texts = [
        "A soccer game with multiple males playing.",
        "Some men are playing a sport.",
        "A man playing the guitar.",
        "The guitar is being played by a woman.",
        "A woman slicing bread with a knife.",
        "The bread was cut with a sharp knife.",
        "The stock market rallied on strong earnings.",
        "Investors celebrated as equities climbed sharply.",
        "The cat sat on the mat and purred.",
        "A feline rested on a rug and made a rumbling sound.",
        "Comets are icy bodies from the outer solar system.",
        "Garbage trucks collect household waste every week.",
        "She typed the report before the meeting began.",
        "The deadline for the report was moved to Friday.",
        "Ocean temperatures are rising due to greenhouse gases.",
        "Global warming is warming the seas.",
    ]
    j = Jina()
    t0 = __import__("time").time()
    Q = j.embed(texts)
    print(f"local fwd {__import__('time').time()-t0:.1f}s", flush=True)
    R = remote(8999, texts)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    d = (Q * R).sum(1)
    print("agree-cos per text:", np.round(d, 5))
    print("min", round(float(d.min()), 5))
    sq, sr = Q @ Q.T, R @ R.T
    print("max |simQ - simR|", round(float(np.abs(sq - sr).max()), 4))
    def nn(S):
        r = []
        for i in range(S.shape[0]):
            row = S[i].copy(); row[i] = -9
            r.append(int(np.argmax(row)))
        return r
    print("NN(q8_0)", nn(sq))
    print("NN(vllm)", nn(sr))
