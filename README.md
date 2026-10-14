# NanoQoS

[![CPU validation](https://github.com/Atyinzi/NanoQoS/actions/workflows/ci.yml/badge.svg)](https://github.com/Atyinzi/NanoQoS/actions/workflows/ci.yml)

基于 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的在线推理调度与性能测量项目。项目把原本偏离线的推理执行链扩展为可接收动态请求、观测请求级延迟、比较调度策略的实验平台。底层 Qwen3 模型、注意力内核、Paged KV Cache、CUDA Graph 等来自上游；NanoQoS 的新增范围和验证边界在下文明确列出。

## 解决什么问题

离线吞吐不能直接说明在线服务的表现。不同请求到达时间、输入长度和截止时间会影响排队、首 token 延迟和按时完成率。NanoQoS 提供三个可切换的等待队列策略及统一的负载和指标口径，用同一批请求比较这些取舍，而不是只展示一个 tokens/s 数字。

| 模块 | NanoQoS 增量 |
| --- | --- |
| 调度 | FCFS、短任务优先、SLO 截止时间优先；aging、prefill/decode 公平性控制、取消和抢占计数；chunked prefill 的 KV block 按需扩容 |
| 服务 | 有界队列、单模型工作线程、补全与聊天接口、SSE 流式输出、超时和 Prometheus 格式指标 |
| 测量 | TTFT、TPOT、E2E、吞吐、SLO goodput、prefix cache 命中、KV block 峰值、重算 token |
| 实验 | 固定种子的 burst/Poisson 到达负载、共享前缀、策略对比、CPU 仿真与 GPU benchmark 入口；显式选择 FlashAttention 或 SDPA 注意力后端 |

请求路径：`HTTP / benchmark → LLMEngine → Scheduler → BlockManager / ModelRunner → token 事件 → 指标或 SSE`。具体实现、接口字段与指标定义见 [技术说明](docs/NANOQOS.md)。

## 快速验证：无需 GPU

Python 3.10–3.12：

```bash
python -m pip install -e ".[test]"
python -m pytest tests -q
python -m benchmarks.run --backend synthetic --requests 24 --arrivals burst --seed 7 --max-input 64 --max-output 16 --shared-prefix 16 --block-size 8 --num-blocks 128 --max-num-batched-tokens 64 --deadline 0.1 --deadline-jitter 0.05 --policy fcfs --output results/fcfs.json
```

将最后一条命令的 `--policy` 和输出文件分别改为 `shortest` / `results/shortest.json`、`slo` / `results/slo.json`，再运行：

```bash
python -m benchmarks.compare results/fcfs.json results/shortest.json results/slo.json
```

CPU 仿真运行真实调度器与 KV block 管理器，但用睡眠代替模型计算；生成的吞吐和延迟不能当作 GPU 推理性能。仓库的 [CPU validation 工作流](.github/workflows/ci.yml) 在 Linux/Windows、Python 3.11/3.12 上执行单元与 HTTP 集成测试，以及相同参数的三策略冒烟测试。

## 真实模型运行

GPU 路径需要兼容的 Linux/WSL2、NVIDIA CUDA/NCCL、Triton 和模型权重。默认 FlashAttention 后端还需要 `flash-attn`；本机小负载功能验证可选择 PyTorch SDPA，无需安装 `flash-attn`。先检查所选后端的环境：

```bash
python -m pip install -e .
python -m benchmarks.preflight --model ./Qwen --attention-backend sdpa
python -m benchmarks.run --backend gpu --model ./Qwen --attention-backend sdpa --eager --requests 1 --max-input 24 --max-output 2 --max-model-len 256 --max-num-batched-tokens 8 --max-num-seqs 1 --gpu-memory-utilization 0.9 --output results/gpu-sdpa-chunked-smoke.json
```

`preflight` 只检查模型文件和环境是否具备启动前提，不能替代真实推理验收；非零退出码表示仍有缺项。SDPA 后端按请求聚合 paged KV，仅用于功能冒烟，不能把其结果当作 FlashAttention 路径的性能数字。上述 `0.9` 是本机 4 GB 显存成功冒烟时使用的配置，不保证其他机器适用；显存不足时先查看其他进程占用。命令、固定实验条件与接口示例见 [技术说明](docs/NANOQOS.md)。`Qwen/` 权重与 `results/` 输出被 Git 忽略，不随仓库发布。

## 验证范围

本地已通过 21 个单元／HTTP 集成测试（其中 3 个 SDPA 数值测试需要 PyTorch），并运行 CPU 合成负载。[此前代码的 CI 验证](https://github.com/Atyinzi/NanoQoS/actions/runs/38566582283)在 Linux/Windows、Python 3.11/3.12 四组环境全部通过；本次后端变更的 CI 状态以最新 Actions 为准。SDPA 在 WSL 的 RTX 3050 上已完成单请求、分块 Prefill 加 Decode 冒烟；FlashAttention 路径仍未验收。因此本仓库目前**没有可声称的 GPU 加速或 SLO 改善数字**；需更严格的正确性和多轮同条件性能实验。详细验收矩阵见 [验证记录](docs/VALIDATION.md)。

## 来源与许可

本仓库基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)。原项目的模型与执行内核不是 NanoQoS 的独立实现；保留原始 [MIT 许可与版权声明](LICENSE)。
