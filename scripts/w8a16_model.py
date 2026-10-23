# Host model of kernels/i8gemm_w8a16.cc: A16 (13-bit, hi/lo int8 pair) x
# per-channel int8, streaming emulation, device-free (SMU-wedged). Also
# asserts int32 overflow headroom. Pattern mirrors pc16_model.py.
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from host_w import shred_w8, shred_a, QACT

GW = int(os.environ.get("GW", "1024"))
GK = int(os.environ.get("GK", "1024"))
MM = int(os.environ.get("MM", "16"))
NW = 8
OB = 3456
KC = int(os.environ.get("KC", str(GK)))
assert GK % KC == 0
KCH = GK // KC
KT = GK // 8
KTC = KC // 8
M8 = MM // 8
RGW = GW // NW
RG = RGW // 8
TILES = RG * KTC
TPO = 52
NWT = -(-TILES // TPO)
ABYTES = 2 * M8 * KTC * 64
CX = -(-ABYTES // OB)
NOD = -(-(MM * RGW * 4) // OB)
NOBJ = 1 + KCH * (CX + NWT)
# HARD RULE: wedge root cause = resident > 32 KB L1/tile
L1 = 32768
need = ABYTES + MM * RGW * 4 + RGW * 4
assert need <= L1, f"L1 OVERFLOW {need} > {L1}  (this is what wedged the NPU)"

rng = np.random.default_rng(11)
W = rng.standard_normal((GW, GK)) * 0.02
X = rng.standard_normal((MM, GK)) * (1.0 if GW <= 1024 else 2.0)
wq = np.rint(W / np.maximum(np.abs(W).max(1, keepdims=True) / 127, 1e-9)) \
       .clip(-127, 127).astype(np.int8)
wd = (np.abs(W).max(1) / 127).astype(np.float32)
s = np.maximum(np.abs(X).max(1) / QACT, 1e-30).astype(np.float32)   # [MM]
a = np.rint(X / s[:, None]).astype(np.int64)      # int16-equivalent, |a|<=8191
ah = np.clip(np.rint(a / 256.0), -127, 127).astype(np.int8)
al = (a - 256 * ah.astype(np.int64)).astype(np.int8)
assert np.abs(al).max() <= 128
acc_hi = ah.astype(np.int64) @ wq.T.astype(np.int64)
acc_lo = al.astype(np.int64) @ wq.T.astype(np.int64)
assert np.abs(acc_hi).max() < 2**31 and np.abs(acc_lo).max() < 2**31, "int32"
Y = (256 * acc_hi + acc_lo) * (s.astype(np.float64)[:, None] * wd.astype(np.float64)[None, :])

# A stream: chunk pairs per row block: [2*rb*KT + kt]=ah, [(2rb+1)*KT+kt]=al
ahs = np.concatenate([shred_a(ah[b * 8:(b + 1) * 8]) for b in range(M8)], 0)  # [M8*KT][64]
als = np.concatenate([shred_a(al[b * 8:(b + 1) * 8]) for b in range(M8)], 0)
# per chunk c: rows for k in [c*KTC,(c+1)*KTC) -> global rows rb*KT + c*KTC + j
wt_all = shred_w8(wq)                                    # [RG][KT][64]
wtb = np.zeros((NW, NOBJ, OB), np.uint8)
for w in range(NW):
    wtb[w, 0, :4] = np.frombuffer(np.int32(w * RG).tobytes(), np.uint8)
    wtb[w, 0, 4:4 + 4 * MM] = np.frombuffer(s.astype(np.float32).tobytes(), np.uint8)
    ds = wd[w * RGW:(w + 1) * RGW]
    wtb[w, 0, 4 + 4 * MM:4 + 4 * MM + ds.size * 4] = np.frombuffer(ds.tobytes(), np.uint8)
    base = 1
    for c in range(KCH):
        ab = np.concatenate([np.concatenate(
                [ahs[b * KT + c * KTC: b * KT + (c + 1) * KTC],
                 als[b * KT + c * KTC: b * KT + (c + 1) * KTC]], 0)
                for b in range(M8)], 0).reshape(-1)      # [2*M8*KTC*64]
        for i in range(CX):
            n = min(OB, len(ab) - i * OB)
            wtb[w, base + i, :n] = ab[i * OB:(i + 1) * OB]
        wt = wt_all[w * RG:(w + 1) * RG, c * KTC:(c + 1) * KTC]
        for i in range(NWT):
            blk = wt.reshape(-1, 64)[i * TPO:(i + 1) * TPO]
            wtb[w, base + CX + i, :blk.size] = blk.reshape(-1)
        base += CX + NWT


def worker(w):
    A = np.zeros(ABYTES, np.int8)
    out = np.zeros((MM, RGW), np.float64)
    rg0 = int(np.frombuffer(wtb[w, 0, :4].tobytes(), np.int32)[0])
    out[:, :] = 0.0
    gsf = np.frombuffer(wtb[w, 0, 4:4 + 4 * MM].tobytes(), np.float32)
    gd = np.frombuffer(wtb[w, 0, 4 + 4 * MM:4 + 4 * MM + RGW * 4].tobytes(), np.float32)
    for c in range(KCH):
      base_o = 1 + c * (CX + NWT)
      base = base_o + CX
      for o in range(CX):
        n = min(OB, ABYTES - o * OB)
        A[o * OB:(o + 1) * OB] = wtb[w, base_o + o, :n].astype(np.int8)
      Ac = A[:2 * M8 * KTC * 64].reshape(2 * M8 * KTC, 8, 8).astype(np.int64)
      for i in range(NWT):
        nt = min(TPO, TILES - i * TPO)
        for j in range(nt):
            tt = i * TPO + j
            t = np.frombuffer(wtb[w, base + i, j * 64:(j + 1) * 64]
                              .tobytes(), np.int8).reshape(8, 8).astype(np.int64)
            rg, kt = rg0 + tt // KTC, tt % KTC
            if kt == 0:
                accH = [np.zeros((8, 8), np.int64) for _ in range(M8)]
                accL = [np.zeros((8, 8), np.int64) for _ in range(M8)]
            for rb in range(M8):
                accH[rb] += Ac[(2 * rb) * KTC + kt] @ t
                accL[rb] += Ac[(2 * rb + 1) * KTC + kt] @ t
            if kt == KTC - 1:
                rl = rg - rg0
                for rb in range(M8):
                    v = 256 * accH[rb] + accL[rb]
                    out[rb * 8:(rb + 1) * 8, rl * 8:rl * 8 + 8] += \
                        v * (gsf[rb * 8:(rb + 1) * 8, None] * gd[None, rl * 8:rl * 8 + 8])
    return out


got = np.zeros((MM, GW), np.float64)
for w in range(NW):
    got[:, w * RGW:(w + 1) * RGW] = worker(w)
cos = np.dot(got.ravel(), Y.ravel()) / (np.linalg.norm(got) * np.linalg.norm(Y))
rel = np.abs(got - Y).max() / np.abs(Y).max()
print("w8a16 model cos", cos, "maxrel", rel)
assert cos > 0.999999 and rel < 1e-6, (cos, rel)
print("w8a16 model PASS")
