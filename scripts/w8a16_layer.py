# P3 device gate: chunked W8A16 (kernels/i8gemm_w8a16.cc) vs numpy oracle.
# Env: GW GK [KC=GK]. Column-pass split keeps per-tile .bss <= 32 KB.
import os, subprocess, sys, time, faulthandler
faulthandler.enable()
import numpy as np
import aie.iron as iron
from aie.iron import Buffer, ObjectFifo, Program, Runtime, Worker
from aie.iron.kernel import Kernel
from aie.iron.controlflow import range_
from aie.iron.device.tile import Tile
from aie.iron.runtime.data import TensorAccessPattern

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from host_w import shred_w8, shred_a, QACT


def TAP4(tdims, off, nobjs):
    d1 = max(1, nobjs // 64)
    while nobjs % d1:
        d1 -= 1
    d2 = nobjs // d1
    return TensorAccessPattern((tdims,), off, [d1, d2, 1, OB],
                               [d2 * OB, OB, 0, 1])


GW_ORIG = int(os.environ.get("GW", "1024"))
GK = int(os.environ.get("GK", "1024"))
KC = int(os.environ.get("KC", str(GK)))
MM = 8
OB = 3456
NW = 8

# .bss guard: outbuf + d per tile <= ~11 KB -> cap columns per tile at 2560
NPASS = -(-GW_ORIG // 2560) if GW_ORIG > 2560 else 1
GW = GW_ORIG // NPASS
KCH = GK // KC
KTC = KC // 8
RGW = GW // NW
RG = RGW // 8
M8 = MM // 8
TILES = RG * KTC
TPO = 52
NWT = -(-TILES // TPO)
ABYTES = 2 * M8 * KTC * 64
CX = -(-ABYTES // OB)
NOD = -(-(MM * RGW * 4) // OB)
NOBJ = 1 + KCH * (CX + NWT)

# DEVICE ALLOW-LIST: unvalidated shapes hung a kernel once (ERT timeout ->
# failed suspend_fw -> SMU wedge #2). Only shapes proven on device/mirror:
ALLOW = {(1024, 1024, 1024), (2048, 1024, 1024), (2560, 1024, 1024),
         (1536, 1024, 1024), (1024, 3072, 1024), (1024, 2048, 1024),
         (1024, 2048, 2048), (3072, 3072, 1024), (1024, 1024, 512)}
assert (GW, GK, KC) in ALLOW, f"shape {GW}x{GK} KC{KC} not device-validated"

# wedge guard: pbuf + static outbuf + d <= 32 KB L1 / tile (SMU-wedge rule)
PBS = -(-ABYTES // 1024) * 1024
assert PBS + MM * RGW * 4 + RGW * 4 + 64 <= 32768, \
    f"L1 OVERFLOW {PBS + MM*RGW*4 + RGW*4} (this is what wedged the NPU)"

SITE = "/home/lovedheart/projects/mlir-aie/ironenv314/lib/python3.14/site-packages"
OBJ = "/tmp/i8a16.o"
cc = os.path.join(os.path.dirname(__file__), "..", "kernels", "i8gemm_w8a16.cc")
subprocess.run([f"{SITE}/llvm-aie/bin/clang++", cc, "-c", "-o", OBJ,
                f"-I{SITE}/mlir_aie/include", "-std=c++20", "-w", "-Os",
                "-DNDEBUG", "-D__AIE_API_AIE_ADF_HPP__",
                "--target=aie2-none-unknown-elf", f"-DGW={RGW}",
                f"-DGK={GK}", f"-DKC={KC}", f"-DMM={MM}", f"-DOB={OB}"],
               check=True)

rng = np.random.default_rng(7)
X = rng.standard_normal((MM, GK)) * 1.0
W = rng.standard_normal((GW_ORIG, GK)) * 0.02
wdall = (np.abs(W).max(1) / 127).astype(np.float32)
wqall = np.rint(W / np.maximum(np.abs(W).max(1)[:, None] / 127, 1e-9)) \
         .clip(-127, 127).astype(np.int8)
s = np.maximum(np.abs(X).max(1) / QACT, 1e-30).astype(np.float32)
a = np.rint(X / s[:, None]).astype(np.int64)
ah = np.clip(np.rint(a / 256.0), -127, 127).astype(np.int8)
al = (a - 256 * ah.astype(np.int64)).astype(np.int8)
wt_all = shred_w8(wqall)                     # [RGtot][KT][64]
ahs = np.concatenate([shred_a(ah[b * 8:(b + 1) * 8]) for b in range(M8)], 0)
als_ = np.concatenate([shred_a(al[b * 8:(b + 1) * 8]) for b in range(M8)], 0)
KTP = GK // 8


def build(wd, wta):
    wtb = np.zeros((NW, NOBJ, OB), np.uint8)
    for w in range(NW):
        wtb[w, 0, :4] = np.frombuffer(np.int32(w * RG).tobytes(), np.uint8)
        wtb[w, 0, 4:4 + 4 * MM] = np.frombuffer(s.tobytes(), np.uint8)
        ds = wd[w * RGW:(w + 1) * RGW]
        wtb[w, 0, 4 + 4 * MM:4 + 4 * MM + ds.size * 4] = \
            np.frombuffer(ds.tobytes(), np.uint8)
        base = 1
        for c in range(KCH):
            abv = np.concatenate([np.concatenate(
                    [ahs[b * KTP + c * KTC: b * KTP + (c + 1) * KTC],
                     als_[b * KTP + c * KTC: b * KTP + (c + 1) * KTC]], 0)
                    for b in range(M8)], 0).reshape(-1)
            for i in range(CX):
                n = min(OB, len(abv) - i * OB)
                wtb[w, base + i, :n] = abv[i * OB:(i + 1) * OB]
            wt = wta[w * RG:(w + 1) * RG, c * KTC:(c + 1) * KTC].reshape(-1, 64)
            for i in range(NWT):
                blk = wt[i * TPO:(i + 1) * TPO]
                wtb[w, base + CX + i, :blk.size] = blk.reshape(-1)
            base += CX + NWT
    return wtb


Kernel._validate_arg = lambda self, arg, val, ty: None
ity = np.ndarray[(OB,), np.dtype[np.uint8]]
oty = np.ndarray[(OB,), np.dtype[np.uint8]]
pty = np.ndarray[(PBS,), np.dtype[np.uint8]]


def make(hh=None):
    ws = [Tile(c % 4, 2 + c // 4) for c in range(NW)]
    rb = [ObjectFifo(ity, name=f"rb{i}", depth=8) for i in range(NW)]
    fouts = [ObjectFifo(oty, name=f"fo{i}", depth=2) for i in range(NW)]
    steps = [Kernel(f"w8a16_step{i}", object_file_name=OBJ, arg_types=[ity, pty])
             for i in range(NW)]
    resets = [Kernel(f"w8a16_reset{i}", object_file_name=OBJ, arg_types=[])
              for i in range(NW)]
    flushes = [Kernel(f"w8a16_flush{i}", object_file_name=OBJ, arg_types=[ity, pty])
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


des = iron.jit(design, object_files=[OBJ])
print("jited", flush=True)
wt_t = iron.zeros((NW * NOBJ * OB,), dtype=np.uint8, device="npu")
ot_t = iron.zeros((NW * NOD * OB,), dtype=np.uint8, device="npu")

GWh = GW_ORIG // NPASS
RGo = GWh // 8
ts = []
for p in range(NPASS):
    sl = slice(p * GWh, (p + 1) * GWh)
    wq, wd = wqall[sl], wdall[sl]
    accH = ah.astype(np.int64) @ wq.T.astype(np.int64)
    accL = al.astype(np.int64) @ wq.T.astype(np.int64)
    Y = (256 * accH + accL) * (s.astype(np.float64)[:, None]
                               * wd.astype(np.float64)[None, :])
    wtb = build(wd, wt_all[p * RGo:(p + 1) * RGo])
    pt = []
    for _ in range(6):
        wt_t.data[:] = wtb.reshape(-1)
        t0 = time.time()
        des(wt_t, ot_t)
        pt.append((time.time() - t0) * 1e3)
    ts.extend(pt[1:])
    out = np.zeros((MM, GWh), np.float64)
    for w in range(NW):
        blk = ot_t.data[w * NOD * OB:w * NOD * OB + MM * RGW * 4]
        blk = blk.view(np.float32).astype(np.float64).reshape(MM, RG, 8)
        for n8 in range(8):
            out[:, w * RGW + n8:(w + 1) * RGW:8] = blk[:, :, n8]
    cos = np.dot(out.ravel(), Y.ravel()) / (np.linalg.norm(out) * np.linalg.norm(Y))
    print(f"pass{p} {GWh}x{GK} KC{KC}: cos {cos:.6f} med {np.median(pt[1:]):.2f} ms",
          flush=True)
    if cos < 0.9999:
        for w in range(NW):
            blk = ot_t.data[w * NOD * OB:w * NOD * OB + MM * RGW * 4]
            blk = blk.view(np.float32).astype(np.float64).reshape(MM, RG, 8)
            got = np.zeros((MM, RGW))
            for n8 in range(8):
                got[:, n8::8] = blk[:, :, n8]
            exp = Y[:, w * RGW:(w + 1) * RGW]
            print("w", w, "nz", (got != 0).sum(), "corr",
                  round(float(np.corrcoef(got.ravel(), exp.ravel())[0, 1]), 4),
                  flush=True)
    assert cos > 0.9999, cos
print(f"PASS {GW_ORIG}x{GK} med {np.median(ts):.2f} ms/pass", flush=True)
