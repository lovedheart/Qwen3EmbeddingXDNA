# Minimal GGUF reader (Qwen3EmbeddingXDNA) — format per ggml/src/gguf.cpp
# (qwentts.cpp ref). Only accepted tensor dtype: Q8_0 (enum 8; 34 B per 32-w).
# GGUF stores dims inner-first; we expose logical [out, in] (out = dims[0]
# in gguf order -> row count after transpose-free reshape; verified in P3).
import mmap
import struct
import numpy as np

from quant_ref import QK8_0, BLOCK_B, blob_split

_F = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
      10: "<Q", 11: "<q", 12: "<d"}


class GGUF:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        assert self.mm[:4] == b"GGUF", "not GGUF"
        (self.version,) = struct.unpack_from("<I", self.mm, 4)
        p = 8
        # GGUF v3: magic, u32 ver, u64 n_tensors, u64 n_kv, KVs, tensor infos
        assert self.version == 3, f"GGUF v{self.version} unsupported"
        (nt,) = struct.unpack_from("<Q", self.mm, p); p += 8
        (nk,) = struct.unpack_from("<Q", self.mm, p); p += 8
        self.n_tensors = int(nt)
        self.meta = {}
        for _ in range(nk):
            k, p = self._str(p)
            t, v, p = self._val(p)
            self.meta[k] = v
        self.tensors = {}  # name -> (dtype, dims[out..], off)
        order = []
        for _ in range(self.n_tensors):
            name, p = self._str(p)
            # gguf.md tensor info: n_dims u32, dims u64[n], type u32, off u64
            (nd,) = struct.unpack_from("<I", self.mm, p); p += 4
            dims = struct.unpack_from(f"<{nd}Q", self.mm, p); p += 8 * nd
            (dt,) = struct.unpack_from("<I", self.mm, p); p += 4
            (off,) = struct.unpack_from("<Q", self.mm, p); p += 8
            self.tensors[name] = (dt, tuple(reversed(dims)), off)
            order.append(name)
        self.order = order
        self.alignment = int(self.meta.get("general.alignment", 32))
        self.data0 = (p + self.alignment - 1) & -self.alignment

    def _str(self, p):
        (n,) = struct.unpack_from("<Q", self.mm, p); p += 8
        return self.mm[p:p + n].decode("utf-8"), p + n

    def _val(self, p):
        (t,) = struct.unpack_from("<I", self.mm, p); p += 4
        if t == 8:
            s, p = self._str(p)
            return 8, s, p
        if t == 7:
            return 7, bool(self.mm[p]), p + 1
        if t == 9:
            (et,) = struct.unpack_from("<I", self.mm, p); p += 4
            (n,) = struct.unpack_from("<Q", self.mm, p); p += 8
            # Array elements carry NO per-element type tag; all are of type et.
            out = []
            for _ in range(n):
                if et == 8:
                    v, p = self._str(p)
                elif et == 9:
                    _, v, p = self._val(p)
                elif et == 7:
                    v, p = bool(self.mm[p]), p + 1
                else:
                    (v,) = struct.unpack_from(_F[et], self.mm, p)
                    p += struct.calcsize(_F[et])
                out.append(v)
            return 9, out, p
        (v,) = struct.unpack_from(_F[t], self.mm, p)
        return t, v, p + struct.calcsize(_F[t])

    def q8_0(self, name):
        """-> (qs int8 [rows, cols], d16 uint16 [rows, cols/32]) for a Q8_0 tensor."""
        dt, dims, off = self.tensors[name]
        assert dt == 8, f"{name}: dtype {dt} != Q8_0 (8) — only Q8_0 supported"
        rows, cols = int(dims[0]), int(dims[1])
        assert cols % QK8_0 == 0
        nb = rows * cols // QK8_0
        blob = np.frombuffer(self.mm, np.uint8, nb * BLOCK_B,
                             self.data0 + off).reshape(nb, BLOCK_B)
        qs, d16 = blob_split(blob, QK8_0)  # [nb,32], [nb,1]
        return qs.reshape(rows, cols), d16.reshape(rows, cols // QK8_0)

    def close(self):
        try:
            self.mm.close()
        except BufferError:  # numpy views of the mmap still alive
            pass
        self.f.close()
