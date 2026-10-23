# P3 shape: 1024x1024 x M8 across 8 MAC tiles.
import os, subprocess, sys, time, faulthandler
faulthandler.enable()
import numpy as np
import aie.iron as iron
from aie.iron import Buffer, ObjectFifo, Program, Runtime, Worker
from aie.iron.kernel import Kernel
from aie.iron.controlflow import range_
from aie.iron.device.tile import Tile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_ref import q8_0_pack, q8_0_unpack
from host_w import shred_w8, shred_a
from aie.iron.runtime.data import TensorAccessPattern


def TAP4(tdims, off, nobjs):
    d1 = max(1, nobjs // 64)
    while nobjs % d1:
        d1 -= 1
    d2 = nobjs // d1
    return TensorAccessPattern((tdims,), off, [d1, d2, 1, OB],
                               [d2 * OB, OB, 0, 1])

GW = 1024
GK = 1024
MM = 8
OB = 3456
NW = int(os.environ.get('NW', 8))
SITE = "/home/lovedheart/projects/mlir-aie/ironenv314/lib/python3.14/site-packages"
OBJ = "/tmp/i8g8.o"
cc = os.path.join(os.path.dirname(__file__), "..", "kernels", "i8gemm.cc")
subprocess.run([f"{SITE}/llvm-aie/bin/clang++", cc, "-c", "-o", OBJ,
                f"-I{SITE}/mlir_aie/include", "-std=c++20", "-w", "-Os",
                "-DNDEBUG", "-D__AIE_API_AIE_ADF_HPP__",
                "--target=aie2-none-unknown-elf", f"-DGW={GW // NW}", f"-DGK={GK}",
                f"-DMM={MM}", f"-DOB={OB}"], check=True)

KT, RG = GK // 8, (GW // NW) // 8
TILES = KT * RG                             # 512
TPO = 52
NWT = -(-TILES // TPO)
CX = 1                                      # A = 4 KB fits 2 objs? 4096>3456 -> 2
CX = -(-(KT * 64) // OB)
NSD = -(-(GW // NW * (GK // 32) * 2) // OB)   # d16 rows for this slice
NOD = -(-(MM * (GW // NW) * 4) // OB)       # 2 KB -> 1
NOBJ = 1 + CX + NSD + NWT

rng = np.random.default_rng(7)
W = rng.standard_normal((GW, GK)) * 0.02
X = rng.standard_normal((MM, GK)) * 1.0
wq, wd16 = q8_0_pack(W)
amax = np.abs(X).max()
as_ = float(amax / 127.0)
aq = np.clip(np.rint(X / as_), -127, 127).astype(np.int8)
Y = (aq.astype(np.float64) * as_) @ q8_0_unpack(wq, wd16).T

wt = shred_w8(wq)                           # [RGtot=64][kt][64]
at = shred_a(aq)
ab = at.reshape(-1)
assert len(ab) <= CX * OB

wtb = np.zeros((NW, NOBJ, OB), np.uint8)
for w in range(NW):
    wtb[w, 0, :4] = np.frombuffer(np.float32(as_).tobytes(), np.uint8)
    wtb[w, 0, 4:8] = np.frombuffer(np.int32(w * 8).tobytes(), np.uint8)
    for i in range(CX):
        wtb[w, 1 + i, :min(OB, len(ab) - i * OB)] = ab[i * OB:(i + 1) * OB]
    ds = wd16.astype(np.uint16)[w * (GW // NW):(w + 1) * (GW // NW)].view(np.uint8).reshape(-1)
    for j in range(NSD):
        n = min(OB, len(ds) - j * OB)
        if n > 0:
            wtb[w, 1 + CX + j, :n] = ds[j * OB:j * OB + n]
    p = 1 + CX + NSD
    for i in range(NWT):
        nt = min(TPO, TILES - i * TPO)
        blk = wt[w * RG:(w + 1) * RG].reshape(-1, 64)[i * TPO:(i + 1) * TPO]
        nb = blk.size
        wtb[w, p + i, :nb] = blk.reshape(-1)

Kernel._validate_arg = lambda self, arg, val, ty: None
ity = np.ndarray[(OB,), np.dtype[np.uint8]]
oty = np.ndarray[(OB,), np.dtype[np.uint8]]
ABYTES_P = -(-(KT * 64) // 4096) * 4096
NDB_P = (GW // NW) * (GK // 32) * 2
DOFF_P = (ABYTES_P + 4095) & ~4095
POFF_P = (DOFF_P + NDB_P + 4095) & ~4095
PBS = POFF_P + MM * (GW // NW) * 4
assert PBS <= 32768, PBS
pty = np.ndarray[(PBS,), np.dtype[np.uint8]]


NO = 1 + CX + NSD + NWT          # objs per pass
NFB = NW * NO                    # total objs through router




def make(hh=None):
    ws = [Tile(c % 4, 2 + c // 4) for c in range(NW)]
    rb = [ObjectFifo(ity, name=f"rb{i}", depth=8) for i in range(NW)]
    fouts = [ObjectFifo(oty, name=f"fo{i}", depth=2) for i in range(NW)]
    steps = [Kernel(f"i8_step{i}", object_file_name=OBJ, arg_types=[ity, pty])
             for i in range(NW)]
    resets = [Kernel(f"i8_reset{i}", object_file_name=OBJ, arg_types=[])
              for i in range(NW)]
    flushes = [Kernel(f"i8_flush{i}", object_file_name=OBJ, arg_types=[ity, pty])
               for i in range(NW)]
    pbs = [Buffer(pty, name=f"P{i}", tile=t) for i, t in enumerate(ws)]

    def core_fn(fin_c, fout_p, pb, step, reset, flush):
        reset()
        for i in range_(NOBJ):
            o = fin_c.acquire(1)
            step(o, pb)
            fin_c.release(1)
        for j in range_(NOD):
            d = fout_p.acquire(1)
            flush(d, pb)
            fout_p.release(1)

    wks = [Worker(core_fn, [rb[i].cons(), fouts[i].prod(), pbs[i], steps[i],
                            resets[i], flushes[i]], tile=ws[i])
           for i in range(NW)]
    rt = Runtime()
    wty = np.ndarray[(NW * NOBJ * OB,), np.dtype[np.uint8]]
    oty_all = np.ndarray[(NW * NOD * OB,), np.dtype[np.uint8]]
    with rt.sequence(wty, oty_all) as (wh, oh):
        for i in range(NW):
            rt.fill(rb[i].prod(), wh,
                    tap=TAP4(NW * NOBJ * OB, i * NOBJ * OB, NOBJ))
        for wk in wks:
            rt.start(wk)
        for i in range(NW):
            rt.drain(fouts[i].cons(), oh,
                     tap=TAP4(NW * NOD * OB, i * NOD * OB, NOD),
                     wait=(i == NW - 1))
    return Program(iron.get_current_device(), rt).resolve_program()


def design(w: iron.In, o: iron.Out):
    return make()




import shutil, subprocess as _sp
_orig = _sp.run
def _spy(cmd, *a, **k):
    try:
        r = _orig(cmd, *a, **k)
    finally:
        s_ = " ".join(map(str, cmd)) if isinstance(cmd, (list, tuple)) else str(cmd)
        import re as _re
        m = _re.search(r"(\S+)\.mlir", s_)
        if m: shutil.copy(m.group(0), "/tmp/fail8.mlir")
    return r
if os.environ.get("SPY"): _sp.run = _spy
des = iron.jit(design, object_files=[OBJ])
print("jited", flush=True)
wt_t = iron.zeros((NW * NOBJ * OB,), dtype=np.uint8, device="npu")
ot_t = iron.zeros((NW * NOD * OB,), dtype=np.uint8, device="npu")

ts = []
for _ in range(6):
    wt_t.data[:] = wtb.reshape(-1)
    t0 = time.time()
    des(wt_t, ot_t)
    ms = (time.time() - t0) * 1e3
    print("ms", ms, flush=True)
    ts.append(ms)

out = np.zeros((MM, GW), np.float64)
for w in range(NW):
    blk = ot_t.data[w * NOD * OB:w * NOD * OB + MM * (GW // NW) * 4]
    ncol = GW // NW                        # kernel writes [m][rg_local*8+n]
    blk = blk.view(np.float32).astype(np.float64).reshape(MM, RG, 8)
    for n8 in range(8):
        out[:, w * ncol + n8:(w + 1) * ncol:8] = blk[:, :, n8]
cos = np.dot(out.ravel(), Y.ravel()) / (np.linalg.norm(out) * np.linalg.norm(Y))
print("fanout cos", cos, flush=True)
if cos < 0.99:
    for w in range(NW):
        blk = ot_t.data[w * NOD * OB:w * NOD * OB + MM * (GW // NW) * 4]
        blk = blk.view(np.float32).astype(np.float64).reshape(MM, RG, 8)
        got = np.zeros((MM, GW // NW))
        for n8 in range(8):
            got[:, n8::8] = blk[:, :, n8]
        exp = Y[:, w * (GW // NW):(w + 1) * (GW // NW)]
        nz = (got != 0).sum()
        r = np.corrcoef(got.ravel(), exp.ravel())[0, 1] if nz > 8 else float("nan")
        print("w", w, "nz", nz, "corr", round(r, 4), flush=True)
if cos < 0.99:
    ref = (aq.astype(np.float64) * as_) @ W.T   # unquantized reference
    for w in range(2):
        blk = ot_t.data[w * NOD * OB:w * NOD * OB + MM * (GW // NW) * 4].view(np.float32).astype(np.float64)
        blk = blk.reshape(MM, RG, 8)
        got = np.zeros((MM, GW // NW))
        for n8 in range(8):
            got[:, n8::8] = blk[:, :, n8]
        exp = Y[:, w * (GW // NW):(w + 1) * (GW // NW)]
        nz = (got != 0).sum()
        r = np.corrcoef(got.ravel(), exp.ravel())[0, 1] if nz > 8 else float("nan")
        print("worker", w, "nonzero", nz, "corr", r)
