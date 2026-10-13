# NanoQoS: 在线推理调度与性能测量

## 目标与代码边界

本项目在 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的执行引擎上增加在线请求调度和测量工具。上游已有 Qwen3 模型、FlashAttention、Paged KV Cache、prefix caching、chunked prefill、CUDA Graph 与 tensor parallel。本扩展修改 `Sequence`、`Scheduler`、`BlockManager` 和 `LLMEngine`，并新增 HTTP 服务、benchmark 与测试。

当前实现适用于单实例、单模型、单 GPU 控制进程。没有实现生产级认证、跨节点调度、持久化请求队列或多模型路由。

## 运行环境

- GPU 路径需要 Linux 或 WSL2、可用的 NVIDIA CUDA/NCCL，以及本项目 `pyproject.toml` 中的依赖。原引擎在初始化时调用 NCCL、Triton 和 FlashAttention。
- CPU 仿真路径只需要 Python 和 `pip install -e ".[test]"` 安装的轻量依赖。仿真会运行真实 `Scheduler`/`BlockManager`，以定时睡眠代替模型计算；它验证功能与指标管线，不代表 GPU 性能。
- 本机 Windows Python 有 CUDA PyTorch 但没有 NCCL、Triton、FlashAttention，因此目前已验证 CPU 仿真与本地 HTTP 假模型集成测试，尚未取得真实 GPU 推理数字。
- 本地 `Qwen/` 为 Qwen3-0.6B 权重目录；已在 `.gitignore` 中忽略，避免误提交大文件。

CPU 测试与仿真可在项目根目录安装：

```bash
python -m pip install -e ".[test]"
```

真实 GPU 路径的依赖列在 `pyproject.toml` 的 `gpu` extra；需要先准备兼容 CUDA 的 PyTorch/NCCL 与编译环境，再安装 Triton、Transformers、FlashAttention。FlashAttention 的构建方式随 CUDA/PyTorch 组合而变化，需在目标环境确认安装成功；本机尚未验证这一步。

## 先跑测试与 CPU 仿真

```bash
python -m pytest tests -q
python -m benchmarks.run --backend synthetic --requests 24 \
  --arrivals burst --seed 7 --max-input 64 --max-output 16 \
  --shared-prefix 16 --block-size 8 --num-blocks 128 \
  --max-num-batched-tokens 64 --deadline 0.1 --deadline-jitter 0.05 \
  --policy fcfs --output results/fcfs.json
```

依次把 `--policy` 改为 `shortest` 和 `slo`，输出到各自文件，再比较：

```bash
python -m benchmarks.compare results/fcfs.json results/shortest.json results/slo.json
```

`--arrivals poisson --rate 5` 可生成持续到达请求；`--shared-prefix` 可构造共享 prompt 前缀。`--deadline-jitter` 为各请求生成不同截止时间，以检查 SLO 策略的实际排序。比较脚本要求各结果拥有相同的 backend、model、workload 和 engine 配置。由于调度时间依赖系统时钟和睡眠精度，CPU 仿真数值只能用于功能检查。

## 真实 GPU benchmark

GPU 环境准备好后，先用小负载和 eager 模式冒烟：

```bash
python -m benchmarks.run --backend gpu --model ./Qwen --eager \
  --requests 8 --max-input 128 --max-output 16 \
  --max-model-len 1024 --max-num-batched-tokens 512 \
  --max-num-seqs 8 --gpu-memory-utilization 0.75 \
  --output results/gpu-smoke.json
```

正式对照实验需要固定模型、GPU、PyTorch/CUDA 版本、随机种子、输入/输出长度分布、到达模式、预热方式以及 KV cache 配置。每个策略至少重复多次并报告分布；不能把一次 CPU 仿真或上游 README 的吞吐写成自己的 GPU 性能提升。

## HTTP 接口

```bash
python -m nanovllm.server --model ./Qwen --host 127.0.0.1 --port 8000 \
  --policy slo --max-pending 32 --max-model-len 2048 \
  --max-num-seqs 16 --max-num-batched-tokens 1024 --eager
```

支持：

- `POST /v1/completions`：`prompt`、`max_tokens`、`temperature`、`stream`；
- `POST /v1/chat/completions`：`messages` 由模型 tokenizer 的 chat template 转换；
- 扩展参数 `deadline_seconds` 和 `priority`；
- `GET /health`、`GET /metrics`（在途请求、完成/失败数、输出 token、缓存/抢占和 TTFT/TPOT 直方图）；
- HTTP SSE 增量输出，请求断开时取消推理，队列满时返回 429。

示例：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"stream":true}'
```

当前接口覆盖常用 completion/chat 字段，不保证覆盖 OpenAI API 的所有参数。为减少显存压力，服务器默认绑定本机地址。`temperature=0` 尚不受上游 sampler 支持。`deadline_seconds` 在服务器中也是超时上限；benchmark 则允许超时请求继续完成，以统计违反率。

## 核心机制

```text
HTTP handler / benchmark
    → LLMEngine.add_request()
    → Scheduler.waiting
    → 策略排序、prefill/decode 轮换、KV block 分配
    → ModelRunner.run() / Attention
    → Scheduler.postprocess() 记录 token 时间与完成状态
    → SSE token 事件 / benchmark 汇总
```

- `fcfs` 按入队顺序；`shortest` 按剩余 prefill token 近似排序；`slo` 按截止时间排序。`shortest` 与 `slo` 对等待时间加入 aging，已开始的 chunked prefill 优先完成。
- 连续 prefill 最多运行 `max_consecutive_prefills` 轮，然后让正在运行的 decode 获得一轮调度机会。该值默认 4，可在 `Config` 中调整。
- KV Cache 仍使用上游的 prefix hash 和引用计数机制。本扩展记录首次入队时可复用的 prompt token、cache 命中/淘汰、峰值使用 block、抢占次数和重算 token。
- 服务端用单个模型线程拥有 CUDA context；HTTP handler 通过线程安全队列提交请求，并使用有界信号量限制在途请求。

## 指标口径

- TTFT：首次输出 token 时间减请求提交时间（HTTP 包含服务队列等待；benchmark 为请求注入引擎的时间）。
- TPOT：同一请求相邻两个输出 token 之间的间隔；只对至少两个输出 token 的请求统计。
- E2E：请求提交至完成时间，起点与 TTFT 一致。
- P50/P95/P99：对请求或 token 间隔样本做线性插值分位数。
- 输出吞吐：所有完成请求的输出 token 数除以整场实验 wall time。
- SLO goodput：deadline 内完成的请求所生成的 token 数除以整场实验 wall time。
- Prefix cache hit fraction：首次进入引擎时复用的 prompt token 总数 / 完成请求的 prompt token 总数。
- Recomputed tokens：抢占后重新入队时，原先已计算且未被 prefix cache 重新命中的 token 数。
- KV peak used blocks：整个实验中同时被引用的物理 block 峰值。

## 目前验证与后续实验

`python -m pytest tests -q` 在本机通过 9 个测试，覆盖 prefix cache 复用、调度顺序、aging、取消、抢占、指标、workload 可复现性、HTTP 补全及聊天流式输出。`results/` 的 CPU 仿真 JSON 是可本地重新生成的功能示例，不是求职简历中的 GPU 加速证据。

在 GPU 上完成多轮真实对照后，再从结果文件提取简历数据。建议先分析瓶颈与负载范围；如果三种策略没有带来可重复的收益，就保留负面结果和适用边界。
