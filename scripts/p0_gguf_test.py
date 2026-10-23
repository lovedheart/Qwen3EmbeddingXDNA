# P0 self-test: synthetic GGUF writer+reader round-trip, Q8_0 parity vs oracle.
import struct
import sys
import numpy as np

sys.path.insert(0, "/home/lovedheart/projects/Qwen3EmbeddingXDNA")
from quant_ref import QK8_0, BLOCK_B, q8_0_pack, q8_0_unpack
import gguf_read


def wstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def kv_str(k, v):
    return struct.pack("<I", 8) + wstr(v)


def kv_u32(k, v):
    return struct.pack("<II", 4, v)


def build(path, qs, d16, rows, cols, meta):
    body = b""
    for k, v in meta.items():
        body += wstr(k) + (kv_str(k, v) if isinstance(v, str) else kv_u32(k, v))
    tinfo = (wstr("blk") + struct.pack("<I", 2) +
             struct.pack("<2Q", cols, rows) + struct.pack("<I", 8) + struct.pack("<Q", 0))
    hdr = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1) + \
        struct.pack("<Q", len(meta)) + body + tinfo
    pad = (-len(hdr)) % 32
    data = np.empty((rows * cols // QK8_0, BLOCK_B), np.uint8)
    data[:, :2] = d16.reshape(-1, 1).view(np.uint16).view(np.uint8).reshape(-1, 2)
    data[:, 2:] = qs.reshape(-1, QK8_0).view(np.uint8)
    with open(path, "wb") as f:
        f.write(hdr + b"\0" * pad + data.tobytes())


rng = np.random.default_rng(0)
for (R, C) in [(4, 32), (3, 128), (17, 1024)]:
    w = rng.standard_normal((R, C)) * 0.05
    qs, d16 = q8_0_pack(w)
    build("/tmp/t_q8.gguf", qs, d16, R, C,
          {"general.architecture": "qwen3", "general.alignment": 32,
           "blk.shape": 1})
    g = gguf_read.GGUF("/tmp/t_q8.gguf")
    qs2, d162 = g.q8_0("blk")
    assert (qs2 == qs).all() and (d162 == d16).all(), (R, C, "read mismatch")
    deq = q8_0_unpack(qs2, d162)
    rel = np.linalg.norm(deq - w) / np.linalg.norm(w)
    print(f"GGUF round-trip {R}x{C} OK, dequant rel={rel:.5f}")
    g.close()
