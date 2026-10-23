# Host side: load Qwen3-Embedding-0.6B weights, Q8_0-pack, build MMUL tile
# streams for the NPU int8 kernel (P1). MMUL layout (aie::mmul<8,8,8,i8,i8,
# acc32>, per llama_gemv_mma.cc): per 8-k tile, A[64]=[m][i], B[64]=[i*8+n].
import glob
import json
import os
import struct
import numpy as np

from quant_ref import q8_0_pack, q8_0_unpack, QK8_0

_HK = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots")
_PATH = glob.glob(os.path.join(_HK, "*", "model.safetensors"))[0]


def tensor(name):
    """Load one safetensors tensor by name -> float64 [shape]."""
    with open(_PATH, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
        v = hdr[name]
        a, b = v["data_offsets"]
        f.seek(8 + n + a)
        raw = f.read(b - a)
    if v["dtype"] == "BF16":
        u = np.frombuffer(raw, np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).astype(np.float64)
    dt = {"F32": np.float32, "F16": np.float16}[v["dtype"]]
    return np.frombuffer(raw, dt).astype(np.float64)


CFG = dict(H=1024, I=3072, L=28, qh=16, kvh=8, hd=128, vocab=151669)
QACT = 8191          # act quant level (13-bit); int32 headroom 8.98e7 @K=3072


def shred_w8(qs):
    """qs [R,K] int8 -> [rg][kt][i*8+n] = qs[rg*8+n][kt*8+i] (B operand,
    llama_gemv_mma layout; KT=K/8 tiles per row group, 64 B/tile)."""
    R, K = qs.shape
    g = qs.reshape(R // 8, 8, K // 8, 8)     # [rg, n, kt, i]
    return np.ascontiguousarray(g.transpose(0, 2, 3, 1))  # [rg,kt,i,n] 64B


def shred_s16(d16):
    """d16 [R,K/32] u16 -> [rg][b][8] u16 (same rg/b as shred_w8)."""
    R, nb = d16.shape
    return np.transpose(d16.reshape(R // 8, 8, nb), (0, 2, 1)).copy()


def shred_a(actq):
    """actq [M,K] int8 (M<=8) -> [kt][m*8+i] = actq[m][kt*8+i], m-padded."""
    M, K = actq.shape
    g = actq.reshape(M, K // 8, 8)            # [m, kt, i]
    v = np.zeros((K // 8, 8, 8), np.int8)     # [kt][m][i]
    v[:, :M] = np.transpose(g, (1, 0, 2))
    return np.ascontiguousarray(v.reshape(K // 8, 64))