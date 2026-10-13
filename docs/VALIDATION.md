# NanoQoS 验证记录

本文件区分「已执行并通过」「由 CI 持续检查」与「尚未验证」，防止把 CPU 仿真误写成真实 GPU 推理结果。

| 验证项 | 当前状态 | 可复现入口 | 能证明什么 |
| --- | --- | --- | --- |
| 调度、KV cache、指标单元测试 | 本地通过 | `python -m pytest tests -q` | 受测状态转换与计算口径 |
| Python 安装包元数据与 wheel | 本地通过 | `python -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .` | 可构建包含源码与原始 MIT 许可证的安装包 |
| HTTP 补全、聊天 SSE、指标集成测试 | 本地通过（假模型） | `tests/test_server.py` | 服务协议、事件流与指标管线 |
| 三策略 CPU 合成负载 | 本地通过 | `python -m benchmarks.run --backend synthetic ...` | 工作负载可生成、调度可推进、结果可比较 |
| Linux/Windows CPU 自动回归 | 工作流已配置；以 GitHub Actions 当次状态为准 | [CPU validation](../.github/workflows/ci.yml) | 在独立环境重复上述检查 |
| Qwen3 真实 GPU 推理与 HTTP 服务 | 未验证 | `docs/NANOQOS.md` 的 GPU 冒烟命令 | 需兼容的 Linux GPU 环境 |
| GPU 策略性能与 SLO 对照 | 未验证 | `docs/NANOQOS.md` 的实验说明 | 需多轮同条件测量及原始结果 |

本地记录：2026-10-13，Windows，`9 passed`。CPU 合成 benchmark 已执行；其计时依赖宿主机睡眠精度，只用于功能检查，不用于性能声明。自动回归的最终状态以仓库 Actions 页面为准，不以本文件推断。

GPU 验收最低条件：固定模型与权重、设备、CUDA/PyTorch 版本、KV cache 配置、输入／输出分布、随机种子、到达模式和预热；每种策略重复多次，保留原始 JSON，报告分布和失败／超时请求。只有满足这些条件后，才能在简历或 README 中写具体性能收益。
