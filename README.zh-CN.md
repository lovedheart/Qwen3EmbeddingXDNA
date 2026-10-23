# Qwen3EmbeddingXDNA

在 **Phoenix XDNA-1 NPU**（AIE-ML，8 个 MAC tile）上通过 IRON / mlir-aie
**完整运行 Qwen3-Embedding-0.6B** —— 所有权值矩阵乘法都在 NPU 上完成；
attention、RoPE 和归一化在主机侧以 float64 计算。附带 **OpenAI 兼容的
`/v1/embeddings` 服务**，与 `:8999` 上的 ROCm vLLM 容器 JSON 格式完全兼容。

**支持的模型 —— Q8_0，已测试：**

| 模型 | 路径 | 实测结果 |
|---|---|---|
| Qwen3-Embedding-0.6B | 全模型跑在 NPU 上（W8A16 引擎） | 对比 fp64 cos 0.9960–0.9967，服务 `:8996` 在线可用 |
| Qwen3-Embedding-0.6B（Q8_0 GGUF） | 通用 GGUF 引擎，主机 f64；对比 vLLM `:8993` | agree-cos ≥ 0.9994，最近邻检索一致 |
| jina-embeddings-v5-text-small（0.6B Q8_0 GGUF） | Q8_0 前向 + 对比 vLLM `:8999` 一致性测试 | agree-cos ≥ 0.9998，最近邻检索完全一致 |

以上均可**同时运行** —— 每个模型一个进程、各占一个端口（`npu` 后端例外：
独占 NPU 设备）。三者主干同为 28 层 Qwen3，NPU design 几何完全通用（jina
仅 vocab 151936 和 prompt/pooling 预处理不同；接入 NPU 引擎只需换 vocab
偏移并过设备验证）。English version: [README.md](README.md)。

本项目是专用单模型运行时（"Route B"），从 `~/projects/iron-moe` 分叉而来。
总体计划与进度记录在 ReMe：`digest/tasks/qwen3-embed-npu-dedicated-plan.md`。

## 量化方案 —— W8A16

权重为**按输出通道的 int8**（`d[n] = max|w[n,:]| / 127`）。激活为 16-bit
等效，用 **hi/lo 两个 int8 分量**表示（有效 13-bit，`QACT = 8191`），以利用
NPU 的 int8×int8→int32 MAC：

```
acc = 256 * (a_hi @ W) + (a_lo @ W)          # int32，tile 上完成
每个 row-group、每个 K-chunk 折叠一次 (256*acc) * s[m] * d[n]，drain 累加到 f32
```

精度阶梯（对比 float64 参考 embedding 的 cos）：

| 方案 | cos |
|---|---|
| Q8_0（GGUF） | 0.9997 |
| **本方案 W8A16，全模型上 NPU** | **0.9960 – 0.9967** |
| 纯 int8 激活 | 0.989（否决） |

验收标准是"Q8_0 等效区间"（≥ 0.995）。

## 目录结构

| 路径 | 作用 |
|---|---|
| `kernels/i8gemm_w8a16.cc` | **正式 kernel** —— W8A16 GEMM，K 分块、列分遍 |
| `kernels/i8gemm.cc`、`kernels/null.cc` | P1 单元 GEMM kernel、健康探针 null kernel |
| `embed.py` | 主机 float64 参考前向（`EmbedEngine`），last-token + L2 池化 |
| `scripts/p4_npu_engine.py` | **设备引擎** `NpuEmbed`：5 个 JIT design，按权重的流模板 |
| `embed_server.py` | OpenAI 兼容服务；`EMBED_BACKEND=npu` 走 `NpuEmbed` |
| `scripts/w8a16_layer.py` | 逐 shape 设备门禁（新 shape 上硬件的唯一入口） |
| `scripts/w8a16_model.py` | 免设备镜像模型 —— **任何设备测试之前必须先跑** |
| `scripts/bench_null.py` | 健康探针（null 流 ~0.26 ms = NPU 正常） |
| `gguf_read.py` | GGUF 读取器（metadata + tensor info + Q8_0 载荷） |
| `quant_ref.py` / `host_w.py` | Q8_0 数值基准；int8 权重 shredding（供数据流用） |
| `scripts/gguf_embed.py` | 通用 Qwen3 Q8_0 GGUF 引擎（读 GGUF 元数据；qwen3-embedding / jina-v5 通用） |
| `scripts/safetensors_to_gguf_q80.py` | HF safetensors → Q8_0 GGUF 转换器（llama.cpp 命名） |
| `scripts/jina_q80_eval.py`、`jina_fp64_ref.py` | jina-v5 Q8_0 与 vLLM 一致性测试 |
| `scripts/run.sh` | 激活 `ironenv314` 并带 AI 工具链运行脚本 |

## 运行方法

先做 NPU 健康检查（应得 ~0.26 ms，无报错；`host`/`jina` 后端不碰设备，可跳过）：

```bash
bash scripts/run.sh scripts/bench_null.py 8 4 1
```

启动 embedding 服务 —— 用 `EMBED_BACKEND` 选后端（`run.sh` 会自动激活
`ironenv314` 工具链）：

| 后端 | 模型 / 引擎 | 示例 |
|---|---|---|
| `npu` | Qwen3-Embedding-0.6B，**全前向跑在 NPU 上**（`p4_npu_engine.NpuEmbed`） | `EMBED_BACKEND=npu bash scripts/run.sh embed_server.py --port 8996` |
| `jina` | jina-v5-text-small Q8_0 GGUF，主机 f64 前向（`jina_q80_eval.Jina`） | `EMBED_BACKEND=jina bash scripts/run.sh embed_server.py --port 8995` |
| `gguf` | 任意 Qwen3 结构 Q8_0 GGUF —— **同时支持** Qwen3-Embedding-0.6B 与 jina-v5（主机 f64，`gguf_embed.GgufEmbed`；GGUF 自带分词器不同时用 `--tokenizer` 指定，jina 必须指定） | `EMBED_BACKEND=gguf bash scripts/run.sh embed_server.py --port 8994 --gguf <path.gguf> [--tokenizer <tokenizer.json>]` |
| `host` | Qwen3-Embedding-0.6B，主机 float64 参考实现（`embed.EmbedEngine`） | `EMBED_BACKEND=host bash scripts/run.sh embed_server.py --port 8990` |

`npu` 后端首次启动会 JIT 5 个 design（约 8 秒），之后开始服务；省略
`--port` 时默认 8996。后台常驻（退出终端后继续运行）：

```bash
EMBED_BACKEND=npu nohup bash scripts/run.sh embed_server.py --port 8996 \
    > /tmp/srv8996.log 2>&1 &
```

调用（OpenAI 兼容格式，model 名称与 `/v1/models` 一致即可）：

```bash
curl -s localhost:8996/health
curl -s localhost:8996/v1/embeddings -H 'content-type: application/json' \
     -d '{"input":["hello world"],"model":"Qwen3-Embedding-0.6B"}'
```

运行其他 `run.sh` 设备脚本前先杀掉服务 —— 它们争抢同一个设备。
改动 kernel 后要清 `~/.npu/cache`（JIT 产物按 design 缓存，不跟源码失效）。

权重：主机参考从 `~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/*/` 读取。

## 构建（生成 Q8_0 模型文件）

Qwen3-Embedding GGUF —— 用本仓库脚本从 HF safetensors 本地转换（llama.cpp
命名；Q8_0 = f16 缩放 + 32×int8 块；瞬时内存约 12 GB）：

```bash
~/projects/mlir-aie/ironenv314/bin/python scripts/safetensors_to_gguf_q80.py
# -> ~/models/qwen3-embedding/qwen3-embedding-0.6b-Q8_0.gguf（约 633 MB）
```

jina-v5 GGUF —— 直接下载，不要转换（该仓库自带 tokenizer.json；Qwen3 的会
多附加一个 jina 不用的特殊 token）：

```bash
# v5-small-retrieval-Q8_0.gguf -> ~/models/jina-v5/
```

两个模型同时跑（示例：8994 = Qwen3，8995 = jina）：

```bash
EMBED_BACKEND=gguf nohup bash scripts/run.sh embed_server.py --port 8994 \
  --gguf ~/models/qwen3-embedding/qwen3-embedding-0.6b-Q8_0.gguf \
  > /tmp/srv8994.log 2>&1 &
EMBED_BACKEND=gguf nohup bash scripts/run.sh embed_server.py --port 8995 \
  --gguf ~/models/jina-v5/v5-small-retrieval-Q8_0.gguf \
  --tokenizer ~/models/jina-v5/tokenizer.json \
  > /tmp/srv8995.log 2>&1 &
```

各端口的 `/v1/models` 以 GGUF 文件名作为模型 id。

## 硬件安全规则（学费很贵）

两种情况会把 NPU 卡死，且**只有断电重启能救**：

1. **单 tile L1 溢出** —— 常驻 pbuf + `g_outbuf`（M·RGW·4）+ `g_d` 每个
   tile 必须 ≤ 32768 B。`w8a16_model.py` 里有断言强制检查。
2. **kernel 挂起** —— 任何空转 kernel 触发 `ERT_CMD_STATE_TIMEOUT`，随后
   拆除阶段 `aie2_suspend_fw` 失败（ret −32），SMU 即死。

因此：

* **只有 `scripts/w8a16_layer.py` 允许清单内的 `(GW, GK, KC)` 能上设备**
  （q 2048×1024、kv 1024×1024、o 1024×2048、gu 1536×1024 两遍列分块、
  dn 1024×3072 K 分块）。脚本在触碰硬件前先断言清单成员资格。
* **先镜像后上机**：候选 shape 必须先 PASS `scripts/w8a16_model.py`（免设备）
  —— 镜像通过但设备挂起说明是 DMA 边界问题，不是数学问题。
* **只跑小的 shape**：设备端流 ≤ ~4 MB，`MM=8`；禁止奇形怪状或超小探针
  （512×128 级别的 shape 会卡死 shim）。
* 静态 `.bss` 每 tile 上限约 11.5 KB → 每 tile `GW ≤ 2560` 列，这就是
  gate/up 分两遍列计算的原因。

卡死症状：`RuntimeError: DRM_IOCTL_AMDXDNA_GET_INFO IOCTL failed
(err=-22)` → 重启后用 `bench_null.py` 重新探活。

## 设计不变量

* 每次 drain ≤ `OB = 3456` B；激活以对象流传输，绝不整块大张量。tile 映射
  `Tile(c%4, 2+c//8)`；每权重对象 `TPO = 52` 字节。
* 每层数据流：obj0 元信息（`i32 rg0 | f32 s[MM] | f32 d[GW]`），随后每个
  K-chunk：A 对象（`ah`/`al` 按 row-block 与 K-tile 交织）+ 权重对象。
  K 分块保证 A ≤ 24 KB/tile（down_proj：`KC=1024, KCH=3`）。
* Attention/RoPE 用 **rotate-half** 约定（两半各复制 cos/sin，旋转 =
  `cat(-x2, x1)`）。交错配对式 RoPE 是静默错误，28 层复利后 agree-cos 掉到
  ~0.96 —— 调试时先查这个。

## 验证状态

| 项目 | 结果 |
|---|---|
| Null 派发探针 | 0.267 ms |
| 逐 shape 设备门禁（`w8a16_layer.py`） | 全 shape cos 1.000000：1024² 0.75 ms、2048×1024 1.10、1536×1024 0.88、1024×3072 1.70 ms |
| 全模型上 NPU（`p4_npu_engine.py`） | 对比 fp64 cos 0.9960–0.9967，1.0–3.3 s/条（性能调优待做） |
| 服务 `:8996` | `/health` ok，`/v1/embeddings` → (N, 1024)，单位范数 |
| Q8_0 GGUF vs vLLM `:8999`（jina-v5，16 条 STS 文本） | agree-cos ≥ 0.9998，最近邻检索一致，相似度矩阵最大偏差 0.0028 |
| Q8_0 GGUF vs vLLM `:8993`（Qwen3-Embedding Q8_0，16 条 STS 文本） | agree-cos ≥ 0.9994（均值 0.9996），最近邻检索一致，相似度矩阵最大偏差 0.0074 |

## 工具链

* Python：`~/projects/mlir-aie/ironenv314`（3.14）；设备运行统一走
  `bash scripts/run.sh <script>`。
* kernel 编译：`$SITE/llvm-aie/bin/clang++ -std=c++20 -Os -DNDEBUG
  -D__AIE_API_AIE_ADF_HPP__ --target=aie2-none-unknown-elf
  -DGW=… -DGK=… -DKC=… -DMM=8 -DOB=3456`，其中
  `SITE=/home/lovedheart/projects/mlir-aie/ironenv314/lib/python3.14/site-packages`。
* GGUF 格式参考源码：`/home/lovedheart/qwentts.cpp`
  （`ggml/src/gguf.cpp`；Q8_0 = f16 d + 32×int8 见 `ggml-common.h`）。
