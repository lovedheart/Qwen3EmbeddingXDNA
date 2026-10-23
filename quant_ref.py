# Q8_0 quant/dequant numpy oracle — ONLY dtype handled in this project.
# Bit-exact with ggml block_q8_0 (qwentts.cpp/ggml/src/ggml-common.h:251):
#   34-byte block = { fp16 delta d; int8 qs[32] }, block spans K (row) axis.
# Symmetric absmax scale, round-to-nearest-even quantization.
import numpy as np

QK8_0 = 32
BLOCK_B = 2 + QK8_0  # 34 bytes


def _f16_bits(d):
    return np.asarray(d, np.float32).astype(np.float16).view(np.uint16)


def _f16_vals(bits):
    return np.asarray(bits, np.uint16).view(np.float16).astype(np.float64)


def q8_0_pack(w):
    """w [R,K] float -> (qs int8 [R,K], d16 uint16 [R,K/32]) — GGUF semantics."""
    w = np.asarray(w, np.float64)
    assert w.ndim == 2 and w.shape[1] % QK8_0 == 0
    R, K = w.shape
    b = w.reshape(R, K // QK8_0, QK8_0)
    amax = np.abs(b).max(2)
    d = np.where(amax > 0, amax / 127.0, 1.0)
    d16 = _f16_bits(d)
    q = np.clip(np.rint(b / _f16_vals(d16)[:, :, None]), -127, 127).astype(np.int8)
    return q.reshape(R, K), d16


def q8_0_unpack(qs, d16):
    """-> [R,K] float64 (exactly what the kernel computes)."""
    R, K = qs.shape
    return (_f16_vals(d16)[:, :, None] * qs.reshape(R, K // QK8_0, QK8_0)).reshape(R, K)


def q8_0_blob(qs, d16):
    """GGUF on-disk interleaved bytes [R, 34*K/32] (d then 32 qs, per block)."""
    R, K = qs.shape
    out = np.empty((R, K // QK8_0, BLOCK_B), np.uint8)
    out[:, :, :2] = d16.reshape(R, K // QK8_0, 1).view(np.uint16).view(np.uint8).reshape(R, K // QK8_0, 2)
    out[:, :, 2:] = qs.reshape(R, K // QK8_0, QK8_0).view(np.uint8)
    return out.reshape(R, -1)


def blob_split(blob, K):
    """GGUF blob bytes [R, 34*K/32] -> (qs int8 [R,K], d16 [R,K/32])."""
    R = blob.shape[0]
    g = blob.reshape(R, K // QK8_0, BLOCK_B)
    return (g[:, :, 2:].reshape(R, K).astype(np.int8),
            g[:, :, :2].view(np.uint16).reshape(R, K // QK8_0))


def quant_act(x):
    """Per-token (row) symmetric int8 activation: -> (q int8 [R,K], s f32 [R])."""
    x = np.asarray(x, np.float64)
    amax = np.abs(x).max(1, keepdims=True)
    s = np.where(amax > 0, amax / 127.0, 1.0)
    return np.clip(np.rint(x / s), -127, 127).astype(np.int8), s
