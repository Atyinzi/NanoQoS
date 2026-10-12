# NanoQoS

基于 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的在线推理调度与性能测量项目。NanoQoS 在原有推理引擎上实现了请求级调度、HTTP 服务和可复现的负载测试；它不是从零实现的 vLLM。

## 本项目新增

- FCFS、短任务优先和 SLO 截止时间优先三种等待队列策略，含 aging 与 prefill/decode 公平性控制。
- 有界请求队列、取消、超时、补全及聊天接口、SSE 流式输出。
- TTFT、TPOT、E2E、输出吞吐、SLO goodput、prefix cache 与 KV block 指标。
- 固定随机种子的突发／泊松到达负载、策略对比脚本及 CPU 单元与 HTTP 集成测试。

模型、FlashAttention、Paged KV Cache、prefix caching、chunked prefill、CUDA Graph 和 tensor parallel 的底层实现来自上游 nano-vLLM。本项目对 `Sequence`、`Scheduler`、`BlockManager` 和 `LLMEngine` 做了扩展，新增的服务、测量与测试代码见 [`nanovllm/server.py`](nanovllm/server.py)、[`nanovllm/metrics.py`](nanovllm/metrics.py)、[`benchmarks/`](benchmarks/) 和 [`tests/`](tests/)。上游版权与许可见 [`LICENSE`](LICENSE)。

## 运行

真实模型推理需要 Linux/WSL2、NVIDIA GPU、NCCL、Triton 和 FlashAttention。安装后可运行：

```bash
python -m pip install -e .
python -m nanovllm.server --model ./Qwen --host 127.0.0.1 --port 8000 --eager
```

CPU 环境可先验证调度与指标管线：

```bash
python -m pytest tests -q
python -m benchmarks.run --backend synthetic --requests 24 --policy fcfs --output results/fcfs.json
```

然后使用相同参数分别运行 `--policy shortest` 与 `--policy slo`，通过 `python -m benchmarks.compare` 比较 JSON 结果。完整参数、接口示例、指标口径和环境要求见 [NanoQoS 使用与实验说明](docs/NANOQOS.md)。

## 验证状态

目前 9 个测试在本地通过，CPU 合成负载可生成指标。当前 Windows 环境缺少 NCCL、Triton 与 FlashAttention，尚未完成真实 GPU 推理和性能对照；CPU 合成吞吐不是模型推理性能数据。
