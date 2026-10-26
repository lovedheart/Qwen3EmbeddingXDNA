# P4: Qwen3-Embedding full-model NPU engine. All 7 weight matmuls/layer go
# through kernels/i8gemm_w8a16 (5 jitted designs, allow-listed shapes);
# attention/norms/rope stay host f64 (embed.py math). M=8-row blocks.
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
from embed import CFG, _SNAP, _load_safetensors
from host_w import shred_w8, shred_a, QACT

OB, NW, MM = 3456, 8, 8
SITE = "/home/lovedheart/projects/mlir-aie/ironenv314/lib/python3.14/site-packages"
ALLOW = {(1024, 1024, 1024), (2048, 1024, 1024), (1536, 1024, 1024),
         (1024, 2048, 1024), (1024, 2048, 2048), (1024, 3072, 1024)}
DESIGNS = {"q": (2048, 1024, 1024), "kv": (1024, 1024, 1024),
           "o": (1024, 2048, 1024), "gu": (1536, 1024, 1024),
           "dn": (1024, 3072, 1024)}


def TAP4(tdims, off, nobjs):
    d1 = max(1, nobjs // 64)
    while nobjs % d1:
        d1 -= 1
    d2 = nobjs // d1
    return TensorAccessPattern((tdims,), off, [d1, d2, 1, OB],
                               [d2 * OB, OB, 0, 1])


Kernel._validate_arg = lambda self, arg, val, ty: None
ity = np.ndarray[(OB,), np.dtype[np.uint8]]
oty = np.ndarray[(OB,), np.dtype[np.uint8]]


class Design:
    """One jitted 8-tile design for a fixed (GW, GK, KC)."""

    def __init__(self, name, gw, gk, kc):
        assert (gw, gk, kc) in ALLOW, (gw, gk, kc)
        self.name, self.gw, self.gk, self.kc = name, gw, gk, kc
        self.kch = gk // kc
        self.ktc = kc // 8
        self.rgw = gw // NW
        self.rg = self.rgw // 8
        self.tiles = self.rg * self.ktc
        self.tpo = 52
        self.nwt = -(-self.tiles // self.tpo)
        self.abytes = 2 * self.ktc * 64
        self.cx = -(-self.abytes // OB)
        self.nod = -(-(MM * self.rgw * 4) // OB)
        self.nobj = 1 + self.kch * (self.cx + self.nwt)
        pbs = -(-self.abytes // 1024) * 1024
        assert pbs + MM * self.rgw * 4 + self.rgw * 4 + 64 <= 32768
        self.pbs = pbs
        pty = np.ndarray[(pbs,), np.dtype[np.uint8]]
        self.obj = f"/tmp/i8w8a16_{name}.o"
        src = os.path.join(os.path.dirname(__file__), "..",
                           "kernels", "i8gemm_w8a16.cc")
        subprocess.run([f"{SITE}/llvm-aie/bin/clang++", src, "-c", "-o", self.obj,
                        f"-I{SITE}/mlir_aie/include", "-std=c++20", "-w", "-Os",
                        "-DNDEBUG", "-D__AIE_API_AIE_ADF_HPP__",
                        "--target=aie2-none-unknown-elf", f"-DGW={self.rgw}",
                        f"-DGK={gk}", f"-DKC={kc}", f"-DMM={MM}", f"-DOB={OB}"],
                       check=True)
        self.wt_t = iron.zeros((NW * self.nobj * OB,), dtype=np.uint8, device="npu")
        self.slots = {}
        self.ot_t = iron.zeros((NW * self.nod * OB,), dtype=np.uint8, device="npu")
        self.tmpl = None
        self.dirty = True
        nobj, nod, cx, nwt, kch = self.nobj, self.nod, self.cx, self.nwt, self.kch
        rg = self.rg

        def make(hh=None):
            ws = [Tile(c % 4, 2 + c // 4) for c in range(NW)]
            rb = [ObjectFifo(ity, name=f"{name}rb{i}", depth=8) for i in range(NW)]
            fouts = [ObjectFifo(oty, name=f"{name}fo{i}", depth=2) for i in range(NW)]
            steps = [Kernel(f"w8a16_step{i}", object_file_name=self.obj, arg_types=[ity, pty])
                     for i in range(NW)]
            resets = [Kernel(f"w8a16_reset{i}", object_file_name=self.obj, arg_types=[])
                      for i in range(NW)]
            flushes = [Kernel(f"w8a16_flush{i}", object_file_name=self.obj, arg_types=[ity, pty])
                       for i in range(NW)]
            bufs = [Buffer(pty, name=f"{name}P{i}", tile=t) for i, t in enumerate(ws)]

            def core_fn(fin_c, fout_p, pb, step, reset, flush):
                reset()
                for i in range_(nobj):
                    o = fin_c.acquire(1)
                    step(o, pb)
                    fin_c.release(1)
                for j in range_(nod):
                    d = fout_p.acquire(1)
                    flush(d, pb)
                    fout_p.release(1)

            wks = [Worker(core_fn, [rb[i].cons(), fouts[i].prod(), bufs[i],
                                    steps[i], resets[i], flushes[i]], tile=ws[i])
                   for i in range(NW)]
            rt = Runtime()
            wty = np.ndarray[(NW * nobj * OB,), np.dtype[np.uint8]]
            oty_all = np.ndarray[(NW * nod * OB,), np.dtype[np.uint8]]
            with rt.sequence(wty, oty_all) as (wh, oh):
                for i in range(NW):
                    rt.fill(rb[i].prod(), wh,
                            tap=TAP4(NW * nobj * OB, i * nobj * OB, nobj))
                for wk in wks:
                    rt.start(wk)
                for i in range(NW):
                    rt.drain(fouts[i].cons(), oh,
                             tap=TAP4(NW * nod * OB, i * nod * OB, nod),
                             wait=(i == NW - 1))
            return Program(iron.get_current_device(), rt).resolve_program()

        def design(w: iron.In, o: iron.Out):
            return make()

        self.jit = iron.jit(design, object_files=[self.obj])

    def set_weights(self, wq, wd, key=None):
        if key is not None and key in self.slots:   # clean device pages
            self.wt_t, self.dirty = self.slots[key], False
            return
        t = np.zeros((NW, self.nobj, OB), np.uint8)
        wtb_w = shred_w8(wq)
        for w in range(NW):
            t[w, 0, :4] = np.frombuffer(np.int32(w * self.rg).tobytes(), np.uint8)
            ds = wd[w * self.rgw:(w + 1) * self.rgw]
            t[w, 0, 4 + 4 * MM:4 + 4 * MM + ds.size * 4] = np.frombuffer(ds.tobytes(), np.uint8)
            base = 1
            for c in range(self.kch):
                wt = wtb_w[w * self.rg:(w + 1) * self.rg, c * self.ktc:(c + 1) * self.ktc]
                wt = wt.reshape(-1, 64)
                for i in range(self.nwt):
                    blk = wt[i * self.tpo:(i + 1) * self.tpo]
                    t[w, base + self.cx + i, :blk.size] = blk.reshape(-1)
                base += self.cx + self.nwt
        self.tmpl = t
        self.dirty = True
        if key is not None:
            slot = iron.zeros((NW * self.nobj * OB,), dtype=np.uint8, device="npu")
            slot.data[:] = t.reshape(-1)
            self.slots[key] = slot
            self.wt_t = slot
            self.dirty = False

    def call(self, ah, al, s):
        """ah/al [nb*8,K] int8, s [nb*8] f32 -> Y [nb*8,gw] f64."""
        nb = ah.shape[0] // 8
        Y = np.zeros((nb * 8, self.gw), np.float64)
        if self.dirty:
            self.wt_t.data[:] = self.tmpl.reshape(-1)
            self.dirty = False
        for b in range(nb):
            r = slice(b * 8, b * 8 + 8)
            for w in range(NW):
                bb = w * self.nobj * OB
                self.wt_t.data[bb + 4:bb + 4 + 32] = \
                    np.ascontiguousarray(s[r], np.float32).view(np.uint8)
                for c in range(self.kch):
                    blk = np.concatenate([
                        shred_a(np.ascontiguousarray(ah[r, c * self.kc:(c + 1) * self.kc])),
                        shred_a(np.ascontiguousarray(al[r, c * self.kc:(c + 1) * self.kc]))]).reshape(-1)
                    o = bb + (1 + c * (self.cx + self.nwt)) * OB
                    for i in range(self.cx):
                        n = min(OB, len(blk) - i * OB)
                        self.wt_t.data[o + i * OB:o + i * OB + n] = blk[i * OB:(i + 1) * OB]
            self.jit(self.wt_t, self.ot_t)
            for w in range(NW):
                blk = self.ot_t.data[w * self.nod * OB:w * self.nod * OB + MM * self.rgw * 4]
                blk = blk.view(np.float32).astype(np.float64).reshape(MM, self.rg, 8)
                for n8 in range(8):
                    Y[b * 8:b * 8 + 8, w * self.rgw + n8:(w + 1) * self.rgw:8] = blk[:, :, n8]
        return Y


def quant_act(X):
    s = np.maximum(np.abs(X).max(1) / QACT, 1e-30).astype(np.float32)
    a = np.rint(X / s[:, None]).astype(np.int64)
    ah = np.clip(np.rint(a / 256.0), -127, 127).astype(np.int8)
    al = (a - 256 * ah.astype(np.int64)).astype(np.int8)
    pad = (-X.shape[0]) % 8
    if pad:
        ah = np.pad(ah, ((0, pad), (0, 0)))
        al = np.pad(al, ((0, pad), (0, 0)))
        s = np.concatenate([s, np.zeros(pad, np.float32)])
    return ah, al, s


class NpuEmbed:
    def __init__(self):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(os.path.join(_SNAP, "tokenizer.json"))
        self.tok.enable_truncation(8192)
        self.w = _load_safetensors(os.path.join(_SNAP, "model.safetensors"))
        c = CFG
        self._cw = {k: v.astype(np.float64) for k, v in self.w.items()
                    if v.ndim == 1 and "norm" in k}
        self._emb = self.w["embed_tokens.weight"].astype(np.float64)
        self._inv = 1.0 / c["rope_theta"] ** (np.arange(0, c["hd"], 2, np.float64) / c["hd"])
        self._q = {}
        self._cur = {}
        t0 = time.time()
        self.d = {n: Design(n, *p) for n, p in DESIGNS.items()}
        print(f"jited {len(self.d)} designs in {time.time()-t0:.0f}s", flush=True)

    def encode(self, texts):
        return [self.tok.encode(t).ids + [CFG["eos"]] for t in texts]

    def gemm(self, dname, W, X, cols=None):
        """X [T,K] f64 -> X @ W[cols].T f64 via NPU (W quantized+cached)."""
        key = (id(W), cols)
        if key not in self._q:
            Wf = W if cols is None else W[cols[0]:cols[1]]
            wd = (np.abs(Wf).max(1) / 127).astype(np.float32)
            wq = np.rint(Wf / np.maximum(np.abs(Wf).max(1)[:, None] / 127, 1e-9)) \
                   .clip(-127, 127).astype(np.int8)
            self._q[key] = (wq, wd)
        des = self.d[dname]
        if self._cur.get(dname) != key:
            des.set_weights(*self._q[key], key=key)
            self._cur[dname] = key
        ah, al, s = quant_act(X)
        return des.call(ah, al, s)[:X.shape[0]]

    def _rope(self, x, pos):
        ang = pos[:, None, None].astype(np.float64) * self._inv[None, None, :]
        cos = np.concatenate([np.cos(ang)] * 2, -1)   # rotate-half (HF Qwen3)
        sin = np.concatenate([np.sin(ang)] * 2, -1)
        x1, x2 = x[..., :64], x[..., 64:]
        return x * cos + np.concatenate([-x2, x1], -1) * sin

    def forward(self, ids):
        c = CFG
        x = self._emb[np.asarray(ids)]
        T = len(ids)
        pos = np.arange(T)
        causal = np.triu(np.full((1, T, T), -np.inf), 1)
        g = lambda n, s: self.w[f"layers.{n}.{s}"]
        for l in range(c["L"]):
            h = x
            xr = x / np.sqrt((x * x).mean(-1, keepdims=True) + c["eps"])
            xr = xr * self._cw[f"layers.{l}.input_layernorm.weight"]
            q = self.gemm("q", g(l, "self_attn.q_proj.weight"), xr).reshape(T, c["qh"], c["hd"])
            k = self.gemm("kv", g(l, "self_attn.k_proj.weight"), xr).reshape(T, c["kvh"], c["hd"])
            v = self.gemm("kv", g(l, "self_attn.v_proj.weight"), xr).reshape(T, c["kvh"], c["hd"])
            qn, kn = self._cw[f"layers.{l}.self_attn.q_norm.weight"], \
                     self._cw[f"layers.{l}.self_attn.k_norm.weight"]
            q = q / np.sqrt((q * q).mean(-1, keepdims=True) + c["eps"]) * qn
            k = k / np.sqrt((k * k).mean(-1, keepdims=True) + c["eps"]) * kn
            q, k = self._rope(q, pos), self._rope(k, pos)
            k = np.repeat(k, c["qh"] // c["kvh"], 1)
            v = np.repeat(v, c["qh"] // c["kvh"], 1)
            s = q.transpose(1, 0, 2) @ k.transpose(1, 2, 0) / np.sqrt(c["hd"]) + causal
            s -= s.max(-1, keepdims=True)
            p = np.exp(s)
            p /= p.sum(-1, keepdims=True)
            o = (p @ v.transpose(1, 0, 2)).transpose(1, 0, 2).reshape(T, c["qh"] * c["hd"])
            x = h + self.gemm("o", g(l, "self_attn.o_proj.weight"), o)
            xr = x / np.sqrt((x * x).mean(-1, keepdims=True) + c["eps"])
            xr = xr * self._cw[f"layers.{l}.post_attention_layernorm.weight"]
            act = []
            for src, tag in (("gate", "mlp.gate_proj.weight"), ("up", "mlp.up_proj.weight")):
                y0 = self.gemm("gu", g(l, tag), xr, (0, 1536))
                y1 = self.gemm("gu", g(l, tag), xr, (1536, 3072))
                act.append(np.concatenate([y0, y1], 1))
            gu, xu = act
            x = x + self.gemm("dn", g(l, "mlp.down_proj.weight"),
                              gu / (1.0 + np.exp(-gu)) * xu)
        x = x / np.sqrt((x * x).mean(-1, keepdims=True) + c["eps"])
        x = x * self._cw["norm.weight"]
        e = x[-1]
        return e / np.linalg.norm(e)

    def embed(self, texts):
        return np.stack([self.forward(ids) for ids in self.encode(texts)])


def _q8(W):
    d = (np.abs(W).max(1) / 127).astype(np.float64)
    q = np.rint(W / np.maximum(np.abs(W).max(1)[:, None] / 127, 1e-9)).clip(-127, 127)
    return q * d[:, None]


class Q8:
    """EmbedEngine with per-row int8 weight quant — the shared quant floor."""

    @staticmethod
    def forward(eng, ids):
        keep = eng.w
        eng.w = {k: (_q8(v.astype(np.float64)) if v.ndim == 2 else v)
                 for k, v in eng.w.items()}
        try:
            return eng.forward(ids)
        finally:
            eng.w = keep


if __name__ == "__main__":
    from embed import EmbedEngine
    T = ["It is a lovely day outside.",
         "The capital of France is Paris.",
         "We the people of the United States, in order to form a more perfect union, "
         "establish justice, ensure domestic tranquility, provide for the common defence, "
         "promote the general welfare, and secure the blessings of liberty to ourselves "
         "and our posterity, do ordain and establish the Constitution of the United "
         "States of America."]
    eng = EmbedEngine()

    ids = eng.encode(T)
    refs = [Q8.forward(eng, i) for i in ids]   # parity: int8 weights both sides
    ne = NpuEmbed()
    ok = True
    for t, i, r in zip(T, ids, refs):
        t0 = time.time()
        e = ne.forward(i)
        cos = float(np.dot(e, r))
        ok &= cos > 0.99
        print(f"T={len(i)} cos {cos:.6f}  {time.time()-t0:.2f}s", flush=True)
    print("PASS" if ok else "FAIL", flush=True)
