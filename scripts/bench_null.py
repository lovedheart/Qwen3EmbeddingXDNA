# P0 microbench: pure-DMA launch floor + streaming bandwidth (null kernel).
#   python bench_null.py [N] [NOW] [NOD]
# N=streams, NOW=weight objs/queue, NOD=drain objs/queue. Reports ms.
import os, subprocess, sys, time
import numpy as np
import aie.iron as iron
from aie.iron import Buffer, ObjectFifo, Program, Runtime, Worker
from aie.iron.kernel import Kernel
from aie.iron.controlflow import range_
from aie.iron.device.tile import Tile
from aie.helpers.taplib import TensorAccessPattern as TAP

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NOW = int(sys.argv[2]) if len(sys.argv) > 2 else 64
NOD = int(sys.argv[3]) if len(sys.argv) > 3 else 1
OB = 3456
SITE = "/home/lovedheart/projects/mlir-aie/ironenv314/lib/python3.14/site-packages"
OBJ = "/tmp/null.o"
if not os.path.exists(OBJ):
    subprocess.run([f"{SITE}/llvm-aie/bin/clang++",
                    os.path.join(os.path.dirname(__file__), "..", "kernels", "null.cc"),
                    "-c", "-o", OBJ, f"-I{SITE}/mlir_aie/include", "-std=c++20",
                    "-w", "-Os", "-DNDEBUG", "-D__AIE_API_AIE_ADF_HPP__",
                    "--target=aie2-none-unknown-elf", f"-DOB={OB}"], check=True)

Kernel._validate_arg = lambda self, arg, val, ty: None
u8 = np.uint8
ty = np.ndarray[(OB,), np.dtype[u8]]
CX = 1  # 1 x-object per queue


def make():
    step = Kernel("null_step", object_file_name=OBJ, arg_types=[ty])
    reset = Kernel("null_reset", object_file_name=OBJ, arg_types=[])
    flush = Kernel("null_flush", object_file_name=OBJ, arg_types=[ty])
    fifos = []

    def design(w: iron.In, x: iron.In, o: iron.Out):
        ws = []
        for e in range(N):
            fin = ObjectFifo(ty, name=f"fi{e}", depth=4)
            fout = ObjectFifo(ty, name=f"fo{e}", depth=2)
            tile = Tile(e % 4, 2 + e // 4)
            xb = Buffer(np.ndarray[(OB,), np.dtype[u8]], name=f"X{e}", tile=tile)

            def core_fn(fin, fout, xb, step, reset, flush,
                        _n=CX + NOW, _nod=NOD):
                reset()
                for i in range_(_n):
                    obj = fin.acquire(1)
                    step(obj)
                    fin.release(1)
                for j in range_(_nod):
                    od = fout.acquire(1)
                    flush(od)
                    fout.release(1)

            ws.append(Worker(core_fn, [fin.cons(), fout.prod(), xb, step,
                                       reset, flush], tile=tile))
            fifos.append((fin, fout))
        rt = Runtime()
        wty = np.ndarray[(N * NOW * OB,), np.dtype[u8]]
        xty = np.ndarray[(N * CX * OB,), np.dtype[u8]]
        oty = np.ndarray[(N * NOD * OB,), np.dtype[u8]]
        with rt.sequence(wty, xty, oty) as (wh, xh, oh):
            for e in range(N):
                rt.fill(fifos[e][0].prod(), xh,
                        tap=TAP(tensor_dims=[N * CX * OB], offset=e * CX * OB,
                                sizes=[1, CX, 1, OB], strides=[0, OB, 0, 1]))
                rt.fill(fifos[e][0].prod(), wh,
                        tap=TAP(tensor_dims=[N * NOW * OB], offset=e * NOW * OB,
                                sizes=[1, NOW, 1, OB], strides=[0, OB, 0, 1]))
                rt.drain(fifos[e][1].cons(), oh,
                         tap=TAP(tensor_dims=[N * NOD * OB], offset=e * NOD * OB,
                                 sizes=[1, NOD, 1, OB], strides=[0, OB, 0, 1]),
                         wait=True)
            for w_ in ws:
                rt.start(w_)
        return Program(iron.get_current_device(), rt).resolve_program()

    return iron.jit(design, object_files=[OBJ])


des = make()
wt = iron.zeros((N * NOW * OB,), dtype=np.uint8, device="npu")
xt = iron.zeros((N * CX * OB,), dtype=np.uint8, device="npu")
ot = iron.zeros((N * NOD * OB,), dtype=np.uint8, device="npu")
des(wt, xt, ot)  # compile+run
ts = []
for _ in range(7):
    a = time.perf_counter(); des(wt, xt, ot); ts.append((time.perf_counter() - a) * 1e3)
ts.sort()
tot_mb = N * (NOW + CX + NOD) * OB / 1e6
print(f"N{N} objs(={NOW}+1) drains{NOD} bytes {tot_mb:.1f} MB -> "
      f"median {ts[len(ts)//2]:.3f} ms  min {ts[0]:.3f}  "
      f"tput {tot_mb/1e3/(ts[len(ts)//2]/1e3):.1f} GB/s")
