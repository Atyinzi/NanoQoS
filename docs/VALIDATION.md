# NanoQoS 验证记录

本文件区分「已执行并通过」「由 CI 持续检查」与「尚未验证」，防止把 CPU 仿真误写成真实 GPU 推理结果。

| 验证项 | 当前状态 | 可复现入口 | 能证明什么 |
| --- | --- | --- | --- |
| 调度、增量 KV block 分配、指标单元测试 | 本地通过 | `python -m pytest tests -q` | 受测状态转换与计算口径，尚不证明 GPU 正确性 |
| Python 安装包元数据与 wheel | 本地通过 | `python -m pip wheel --no-deps --wheel-dir dist .` | 可构建包含源码与原始 MIT 许可证的安装包 |
| HTTP 补全、聊天 SSE、指标集成测试 | 本地通过（假模型） | `tests/test_server.py` | 服务协议、事件流与指标管线 |
| 三策略 CPU 合成负载 | 本地通过 | `python -m benchmarks.run --backend synthetic ...` | 工作负载可生成、调度可推进、结果可比较 |
| Linux/Windows CPU 自动回归 | 2026-10-14 本次代码四组作业通过 | [本次成功的 CI 运行](https://github.com/Atyinzi/NanoQoS/actions/runs/38566582283) | 独立环境重复安装、wheel 构建、测试和合成负载检查 |
| GPU 启动条件静态预检 | WSL SDPA 后端通过；FlashAttention 缺失 | `python -m benchmarks.preflight --model ./Qwen --attention-backend sdpa` | 所选后端依赖齐备；不证明真实推理可运行 |
| PyTorch SDPA 数值参考测试 | 本地通过 3 项 | `python -m pytest tests/test_sdpa_attention.py -q` | CPU 上验证分块因果掩码、GQA 与分页 KV 聚合；不证明 GPU 端到端正确性 |
| SDPA 小负载 GPU 冒烟 | WSL RTX 3050 通过 | `docs/NANOQOS.md` 的 SDPA 冒烟命令 | 单请求分块 Prefill 与 Decode 可完成；不证明性能收益 |
| FlashAttention 后端 GPU 推理与真实模型 HTTP 服务 | 未验证 | `docs/NANOQOS.md` 的默认后端与 HTTP 命令 | 需安装 FlashAttention 并分别做模型与服务验收 |
| GPU 策略性能与 SLO 对照 | 未验证 | `docs/NANOQOS.md` 的实验说明 | 需多轮同条件测量及原始结果 |

本地记录：2026-10-14，Windows，`21 passed`（其中 3 个 SDPA 数值测试需要 PyTorch，本地回环端口测试需要允许监听/连接）。WSL SDPA 后端静态预检 `ready=true`，RTX 3050 Laptop GPU / PyTorch 2.11.0+cu128 / Transformers 4.57.6 上运行单请求 GPU 冒烟成功：24-token 上限的 prompt、8-token prefill budget、2-token 输出；原始 JSON 在被 Git 忽略的 `results/gpu-sdpa-chunked-smoke.json`，测试时工作树为 dirty，数值不用于性能声明。此前 CI 记录：提交 `c8d3882` 的 [CPU validation](https://github.com/Atyinzi/NanoQoS/actions/runs/38566582283) 为 `success`，Linux/Windows × Python 3.11/3.12 四组作业均通过，包含增量 KV 变更；当前 SDPA 后端变更以新 CI 状态为准。

GPU 验收最低条件：固定模型与权重、设备、CUDA/PyTorch 版本、KV cache 配置、输入／输出分布、随机种子、到达模式和预热；每种策略重复多次，保留原始 JSON，报告分布和失败／超时请求。只有满足这些条件后，才能在简历或 README 中写具体性能收益。
