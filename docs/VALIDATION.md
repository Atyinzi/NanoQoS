# NanoQoS 验证记录

本文件区分「已执行并通过」「由 CI 持续检查」与「尚未验证」，防止把 CPU 仿真误写成真实 GPU 推理结果。

| 验证项 | 当前状态 | 可复现入口 | 能证明什么 |
| --- | --- | --- | --- |
| 调度、增量 KV block 分配、指标单元测试 | 本地通过 | `python -m pytest tests -q` | 受测状态转换与计算口径，尚不证明 GPU 正确性 |
| Python 安装包元数据与 wheel | 本地通过 | `python -m pip wheel --no-deps --wheel-dir dist .` | 可构建包含源码与原始 MIT 许可证的安装包 |
| HTTP 补全、聊天 SSE、指标集成测试 | 本地通过（假模型） | `tests/test_server.py` | 服务协议、事件流与指标管线 |
| 三策略 CPU 合成负载 | 本地通过 | `python -m benchmarks.run --backend synthetic ...` | 工作负载可生成、调度可推进、结果可比较 |
| Linux/Windows CPU 自动回归 | 2026-10-13 旧版四组作业通过；本次代码待 CI | [此前成功的 CI 运行](https://github.com/Atyinzi/NanoQoS/actions/runs/38405730903) | 旧版曾在独立环境重复安装、wheel 构建、测试和合成负载检查 |
| GPU 启动条件静态预检 | 本机不通过 | `python -m benchmarks.preflight --model ./Qwen` | 找出缺失依赖；不证明 GPU 推理可运行 |
| Qwen3 真实 GPU 推理与 HTTP 服务 | 未验证 | `docs/NANOQOS.md` 的 GPU 冒烟命令 | 需兼容的 Linux GPU 环境 |
| GPU 策略性能与 SLO 对照 | 未验证 | `docs/NANOQOS.md` 的实验说明 | 需多轮同条件测量及原始结果 |

本地记录：2026-10-14，Windows，`17 passed`（本地回环端口测试需要允许监听/连接）。CPU 合成 benchmark 已执行；其计时依赖宿主机睡眠精度，只用于功能检查，不用于性能声明。CI 记录：提交 `8178d63` 的 [CPU validation](https://github.com/Atyinzi/NanoQoS/actions/runs/38405730903) 为 `success`，Linux/Windows × Python 3.11/3.12 四组作业均通过；那次 CI 不包含本次增量 KV 变更。后续提交的状态以对应 Actions 运行页面为准。

GPU 验收最低条件：固定模型与权重、设备、CUDA/PyTorch 版本、KV cache 配置、输入／输出分布、随机种子、到达模式和预热；每种策略重复多次，保留原始 JSON，报告分布和失败／超时请求。只有满足这些条件后，才能在简历或 README 中写具体性能收益。
