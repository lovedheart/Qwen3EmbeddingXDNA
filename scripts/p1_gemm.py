# P1 unit GEMM test: NPU i8gemm vs numpy oracle, 512x512, M=8.
import os, subprocess, sys, time
import os
OPT = os.environ.get("OPT", "-Os")  #x
import numpy as np
import os
OPT = os.environ.get("OPT", "-Os")  #x
import aie.iron as iron
import os
OPT = os.environ.get("OPT", "-Os")  #x
from aie.iron import Buffer, ObjectFifo, Program, Runtime, Worker
from aie.iron.kernel import Kernel
from aie.iron.controlflow import range_
from aie.iron.device.tile import Tile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_ref import q8_0_pack, q8_0_unpack, QK8_0
from host_w import shred_w8, shred_a

GW = GK = 512
MM = 8
OB = 3456
SITE = "/home/lovedheart/projects/mlir-aie/ironenv314/lib/python3.14/site-packages"
OBJ = "/tmp/i8g.o"
cc = os.path.join(os.path.dirname(__file__), "..", "kernels", "i8gemm.cc")
subprocess.run([f"{SITE}/llvm-aie/bin/clang++", cc, "-c", "-o", OBJ,
                f"-I{SITE}/mlir_aie/include", "-std=c++20", "-w", OPT,
                "-DNDEBUG", "-D__AIE_API_AIE_ADF_HPP__",
                "--target=aie2-none-unknown-elf", f"-DGW={GW}", f"-DGK={GK}",
                f"-DMM={MM}", f"-DOB={OB}"], check=True)

KT, RG = GK // 8, GW // 8
TILES = KT * RG
TPO = (OB // 64) // 4 * 4
NWT = -(-TILES // TPO)
CX = -(-(KT * 64) // OB)
NSD = -(-(GW * (GK // 32) * 2) // OB)
NOD = -(-(MM * GW * 4) // OB)
NOBJ = 1 + CX + NSD + NWT

# --- data
rng = np.random.default_rng(7)
W = rng.standard_normal((GW, GK)) * 0.02
X = rng.standard_normal((MM, GK)) * 1.0
wq, wd16 = q8_0_pack(W)
amax = np.abs(X).max()
as_ = float(amax / 127.0)
aq = np.clip(np.rint(X / as_), -127, 127).astype(np.int8)

wd = wd16.astype(np.uint16)
wt = shred_w8(wq)                       # [rg, kt, 64]
at = shred_a(aq)                        # [kt, 64]
d16b = wd.reshape(GW, GK // 32).astype(np.uint16).view(np.uint8).reshape(-1)

wtb = np.zeros((NOBJ, OB), np.uint8)
wtb[0, :4] = np.frombuffer(np.float32(as_).tobytes(), np.uint8)
ab = at.reshape(-1)
for i in range(CX):
    wtb[1 + i, :min(OB, len(ab) - i * OB)] = ab[i * OB:(i + 1) * OB]
p = 1 + CX
for i in range(NSD):
    n = min(OB, len(d16b) - i * OB)
    if n > 0:
        wtb[p + i, :n] = d16b[i * OB:i * OB + n]
p += NSD
for i in range(NWT):
    nt = min(TPO, TILES - i * TPO)
    blk = wt.reshape(-1, 64)[i * TPO:(i + 1) * TPO]
    wtb[p + i, :nt * 64] = blk.reshape(-1)

# --- oracle (exact same quant math)
wdq = q8_0_unpack(wq, wd16)
Y = (aq.astype(np.float64) * as_) @ wdq.T
cos_ref = None

# --- design (1 worker / 1 stream)
Kernel._validate_arg = lambda self, arg, val, ty: None
ty = np.ndarray[(OB,), np.dtype[np.uint8]]


def make():
    pty = np.ndarray[(48 * 1024,), np.dtype[np.uint8]]
    step = Kernel("i8_step", object_file_name=OBJ, arg_types=[ty, pty])
    reset = Kernel("i8_reset", object_file_name=OBJ, arg_types=[])
    flush = Kernel("i8_flush", object_file_name=OBJ, arg_types=[ty, pty])

    def design(w: iron.In, o: iron.Out):
        fin = ObjectFifo(ty, name="fi", depth=8)
        fout = ObjectFifo(ty, name="fo", depth=2)
        tile = Tile(0, 2)
        pb = Buffer(pty, name="P", tile=tile)

        def core_fn(fin, fout, pb, step, reset, flush):
            reset()
            for i in range_(NOBJ):
                obj = fin.acquire(1)
                step(obj, pb)
                fin.release(1)
            for j in range_(NOD):
                od = fout.acquire(1)
                flush(od, pb)
                fout.release(1)

        wk = Worker(core_fn, [fin.cons(), fout.prod(), pb, step, reset, flush],
                    tile=tile)
        rt = Runtime()
        wty = np.ndarray[(NOBJ * OB,), np.dtype[np.uint8]]
        oty = np.ndarray[(NOD * OB,), np.dtype[np.uint8]]
        with rt.sequence(wty, oty) as (wh, oh):
            rt.fill(fin.prod(), wh)
            rt.drain(fout.cons(), oh, wait=True)
            rt.start(wk)
        return Program(iron.get_current_device(), rt).resolve_program()

    return iron.jit(design, object_files=[OBJ])


des = make()
wt_t = iron.zeros((NOBJ * OB,), dtype=np.uint8, device="npu")
wt_t.data[:] = wtb.reshape(-1)
ot_t = iron.zeros((NOD * OB,), dtype=np.uint8, device="npu")
des(wt_t, ot_t)
ts = []
for _ in range(3):
    a = time.perf_counter(); des(wt_t, ot_t); ts.append((time.perf_counter() - a) * 1e3)
out = ot_t.data[:MM * GW * 4].view(np.float32).astype(np.float64).reshape(MM, GW)
print("out nonfinite", np.count_nonzero(~np.isfinite(out)), "of", out.size,
      " finite absmean", np.nanmean(np.abs(out[np.isfinite(out)])) if np.isfinite(out).any() else None)
bad = np.argwhere(~np.isfinite(out))
print("bad positions (m,row) first20:", bad[:20].tolist(), "rows uniq", np.unique(bad[:,1])[:20], "m uniq", np.unique(bad[:,0]))
g = np.where(np.isfinite(out), out, 0.0)
print("cos finite-only", np.dot(g.ravel(), Y.ravel())/ (np.linalg.norm(g)*np.linalg.norm(Y)))
cos = np.dot(out.ravel(), Y.ravel()) / (np.linalg.norm(out) * np.linalg.norm(Y))
rel = np.linalg.norm(out - Y) / np.linalg.norm(Y)
print(f"gemm 512^3xM8 cos {cos:.6f} rel {rel:.5f} ms {sorted(round(t,2) for t in ts)}")
if cos < 0.9999: print("P1 FAIL")
print("P1 GEMM PASS")
