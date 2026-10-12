# nano-vLLM 源码学习与改造课程

## 项目用途与学习目标

nano-vLLM 是一个面向离线大模型推理的轻量实现。当前代码包含请求调度、Paged KV Cache、prefix caching、chunked prefill、FlashAttention、CUDA Graph 和 tensor parallel 等机制。本课程面向刚完成 CS336 Assignment 1、尚未系统学习推理引擎的学习者，目标是先走通一次请求，再逐步完成“可测量的在线推理调度优化”。

推荐改造目标：在不虚构性能收益的前提下，为引擎加入请求级指标、可复现 workload 和可插拔调度策略，最终用 TTFT、TPOT、P95/P99 延迟、吞吐、缓存命中率和重计算 token 数验证效果。

## 源码版本与本地状态

- 源码位置：当前本地工作区。
- 版本信息：当前目录没有可读取的 Git 提交历史，无法确认精确上游 commit。
- 已观察到的本地内容：仓库包含本地 `Qwen` 权重目录；现已新增 `tests/`、`benchmarks/` 和 `nanovllm/server.py`。详细运行方式见 `docs/NANOQOS.md`。
- 结论边界：README 中的吞吐数字属于项目说明，本课程不把它当作当前机器上的实测结果。

## 前置基础

- Python 类、数据类、枚举、队列和多进程。
- PyTorch tensor shape、`nn.Module` 与推理模式。
- Transformer、RoPE、RMSNorm、GQA 和自回归生成。
- GPU 显存、kernel launch 和同步的基本概念。
- 完成本课程第 4 课前不要求掌握 Triton kernel 编写。

## 覆盖范围

本课程覆盖从 `LLM.generate()` 到 token 生成、调度、KV Cache、prefill/decode 执行、CUDA Graph 和 tensor parallel 的主流程，并给出可观测性及调度改造路线。

暂不覆盖：生产级 HTTP 服务、模型量化、speculative decoding、多节点容错、完整 OpenAI API 兼容，以及 FlashAttention 库内部实现。这些内容应在基础 benchmark 可信后再扩展。

---

## 第 1 课：一次生成请求如何从 prompt 走到输出？（大纲）

**读完能回答：** `LLM.generate()` 如何把多个 prompt 转为序列、反复调度并最终还原为文本？

**主链：**

`用户调用 generate` → `LLMEngine.add_request` → `Sequence` → `Scheduler.add` → 循环调用 `LLMEngine.step` → `Scheduler.schedule` → `ModelRunner.run` → `Scheduler.postprocess` → tokenizer decode → 返回结果。

**必读路径：**

1. `example.py::main`：输入模型路径、prompt 和采样参数；下一步进入 `LLM`；输出生成结果。
2. `nanovllm/llm.py::LLM`：确认公开类只是继承 `LLMEngine`；下一步阅读引擎实现。
3. `nanovllm/engine/llm_engine.py::LLMEngine.__init__`：输入模型配置；创建 worker、runner、tokenizer 和 scheduler。
4. `nanovllm/engine/llm_engine.py::LLMEngine.generate`：输入批量请求；循环执行 step；输出按 `seq_id` 排序后的文本与 token。
5. `nanovllm/engine/llm_engine.py::LLMEngine.step`：输入当前队列状态；连接调度、模型执行与后处理；输出完成序列。

**选读证据：** `nanovllm/sampling_params.py::SamplingParams`、`nanovllm/config.py::Config`。

**暂缓阅读：** `ModelRunner` 的 tensor 准备与 CUDA Graph，由第 4、5 课负责。

---

## 第 2 课：请求状态如何驱动 continuous batching？（大纲）

**读完能回答：** waiting/running/finished 三种状态如何变化，prefill 与 decode 为什么分开调度？

**主链：**

`SequenceStatus.WAITING` → `Scheduler.waiting` → prefill token budget → `RUNNING` → 每轮 decode 一个 token → EOS/长度判断 → `FINISHED`。

**必读路径：**

1. `nanovllm/engine/sequence.py::Sequence`：输入 prompt token 和 sampling params；维护请求状态、token 数和 block table。
2. `nanovllm/engine/scheduler.py::Scheduler.add`：输入新序列；追加到 waiting 队列。
3. `nanovllm/engine/scheduler.py::Scheduler.schedule` 的 prefill 分支：计算 token budget、prefix 命中和 chunked prefill。
4. 同一函数的 decode 分支：检查追加 block 的能力，必要时抢占序列，每个序列安排一个 token。
5. `Scheduler.postprocess`：累计已计算 token、追加采样 token、判断结束并释放 block。

**重要分支：** 当前实现只允许批次中的第一个序列使用 chunked prefill；decode 显存不足时从 running 尾部选择 victim。

**选读证据：** 在纸上模拟两个不同 prompt 长度的请求，记录 waiting/running 和 `num_cached_tokens` 的变化。

**暂缓阅读：** 调度策略改造与公平性分析，由第 8 课负责。

---

## 第 3 课：Paged KV Cache 与 prefix caching 如何协作？（大纲）

**读完能回答：** 逻辑 token、物理 block、引用计数和 prefix hash 之间是什么关系？

**主链：**

`Sequence.block(i)` → 链式 `xxhash` → `hash_to_block_id` 查找 → 命中 block 则增加引用 → 未命中则分配 free block → attention 根据 block table 访问 KV → 完整 block 写入 hash → 请求结束后降低引用计数。

**必读路径：**

1. `nanovllm/engine/sequence.py::num_blocks/block/last_block_num_tokens`：把 token 序列映射为逻辑 block。
2. `nanovllm/engine/block_manager.py::Block`：理解 `ref_count`、`hash`、`token_ids` 的职责。
3. `BlockManager.can_allocate`：输入待调度序列；查找可复用完整前缀 block；输出命中 block 数或 `-1`。
4. `BlockManager.allocate`：复用命中 block 并分配剩余 block；输出写入 `Sequence.block_table` 的物理编号。
5. `BlockManager.hash_blocks`：只对已经计算完成的完整 block 建立缓存索引。
6. `BlockManager.deallocate`：释放引用；引用归零的 block 回到 free 队列，但 hash 元数据在被重新分配前仍可用于 prefix cache。

**选读证据：** 为 BlockManager 编写纯 CPU 单元测试，覆盖共享前缀、引用计数、释放后复用、hash 冲突二次校验和 block 耗尽。

**暂缓阅读：** KV tensor 的真实布局和写入，由第 4 课负责。

---

## 第 4 课：prefill 和 decode 如何准备不同的 GPU 输入？（大纲）

**读完能回答：** 同一个模型为什么需要两套输入准备逻辑，`slot_mapping` 和 `block_tables` 分别解决什么问题？

**主链：**

`ModelRunner.run` → `prepare_prefill` 或 `prepare_decode` → `set_context` → Qwen3 forward → Attention 读取 context → 写入 KV Cache → FlashAttention 计算 → sampler 输出 token → `reset_context`。

**必读路径：**

1. `nanovllm/engine/model_runner.py::ModelRunner.run`：根据 `is_prefill` 选择输入路径并执行采样。
2. `ModelRunner.prepare_prefill`：构造扁平 input ids、positions、变长序列前缀和、slot mapping；prefix 命中时创建 block tables。
3. `ModelRunner.prepare_decode`：每条序列只提交 last token，并提供 context length 与 block table。
4. `nanovllm/utils/context.py::Context/set_context/get_context`：在模型层与执行器之间传递本轮元数据。
5. `nanovllm/models/qwen3.py::Qwen3ForCausalLM/Qwen3Model/Qwen3DecoderLayer/Qwen3Attention`：跟踪 hidden states、QKV、RoPE 和输出投影。
6. `nanovllm/layers/attention.py::store_kvcache/Attention.forward`：根据 slot 写 KV，并分别调用 prefill/decode attention。

**选读证据：** 在不运行大模型的前提下，为输入准备逻辑构造小型 `Sequence`，打印 tensor shape 和 block table。

**暂缓阅读：** FlashAttention 库内部 kernel；当前只把它视为经过验证的执行后端。

---

## 第 5 课：CUDA Graph 消除了哪一部分开销？（大纲）

**读完能回答：** 为什么只捕获 decode、为什么需要固定 batch bucket、padding 位置如何避免写坏 KV Cache？

**主链：**

`ModelRunner.__init__` → warmup → 分配 KV Cache → `capture_cudagraph` 按 batch size 捕获 → decode 时选择不小于真实 batch 的 graph → 拷贝输入并用 `-1` 填充无效 slot → graph replay → 截取真实输出。

**必读路径：**

1. `nanovllm/engine/model_runner.py::warmup_model`：制造最大工作负载并测量模型占用。
2. `ModelRunner.allocate_kv_cache`：根据剩余显存和单 block 字节数计算 block 数。
3. `ModelRunner.capture_cudagraph`：理解 graph batch buckets、静态 tensor 和共享 graph pool。
4. `ModelRunner.run_model`：比较 eager/prefill/大 batch 路径与 graph replay 路径。
5. `nanovllm/layers/attention.py::store_kvcache_kernel`：确认 `slot == -1` 时跳过 padding 写入。

**选读证据：** 对比 `enforce_eager=True/False` 的 decode throughput；必须固定模型、请求、输入输出 token 和预热方式。

**暂缓阅读：** 自定义 CUDA Graph 内存池与跨模型 graph 管理，不属于当前代码范围。

---

## 第 6 课：tensor parallel 如何加载与执行 Qwen3？（大纲）

**读完能回答：** rank 0 如何驱动其他 worker，列并行和行并行层在哪里发生集合通信？

**主链：**

`LLMEngine.__init__ spawn` → 各 rank 创建 `ModelRunner` → NCCL process group → rank 0 通过 shared memory/event 广播方法与参数 → 所有 rank 执行模型 → 并行 Linear/Embedding 完成通信 → rank 0 采样。

**必读路径：**

1. `nanovllm/engine/llm_engine.py::LLMEngine.__init__`：创建 rank 1..N-1 worker，rank 0 留在主进程。
2. `nanovllm/engine/model_runner.py::ModelRunner.__init__/loop/call/write_shm/read_shm`：理解控制面同步方式。
3. `nanovllm/layers/linear.py`：定位 QKV column parallel、merged column parallel 和 row parallel 的权重切分与通信。
4. `nanovllm/layers/embed_head.py`：定位 vocabulary parallel embedding 与 LM head。
5. `nanovllm/utils/loader.py::load_model`：确认 Hugging Face 权重如何映射到 packed modules。

**选读证据：** 单卡环境只做 shape 与分片单元测试；多卡吞吐不得在本机未实测时写入结论。

**暂缓阅读：** 多节点 rendezvous、故障恢复和弹性伸缩，当前实现没有提供。

---

## 第 7 课：第一个改造——建立可信的请求级 benchmark（大纲）

**读完能回答：** 如何证明一次调度或缓存修改改善了用户体验，而不是只让离线总吞吐偶然变快？

**主链：**

`workload 配置` → 生成带 arrival time 的请求 → 请求进入引擎 → 记录 enqueue/first-token/each-token/finish → 汇总 TTFT、TPOT、E2E、吞吐、P95/P99 → JSON 落盘 → 对照实验。

**必读路径：**

1. `bench.py::main`：识别现有随机输入、总吞吐测量及其局限。
2. `nanovllm/engine/sequence.py::Sequence`：计划新增请求时间戳与累计计数，但先写测试再改字段。
3. `nanovllm/engine/llm_engine.py::add_request/step/generate`：确定 enqueue、first token 和 finish 的准确打点位置。
4. `nanovllm/engine/scheduler.py::schedule/preempt/postprocess`：计划统计缓存命中 block、抢占次数和重计算 token。
5. 新增 `benchmarks/` 与 `tests/`：分离 workload、指标聚合、结果序列化和纯 CPU 单元测试。

**第一批交付：**

- 不改变调度语义的 metrics 数据结构。
- FCFS baseline 的 burst 与固定长度 workload。
- JSON 结果包含硬件、模型、配置、随机种子和预热信息。
- 纯 CPU 测试覆盖 percentile、TTFT/TPOT 定义和零 token 边界。

**暂缓阅读：** FastAPI、Prometheus 与线上部署；先保证离线指标口径可靠。

---

## 第 8 课：第二个改造——可插拔调度与 SLO 实验（大纲）

**读完能回答：** FCFS、短任务优先和 SLO-aware 策略怎样在吞吐、尾延迟与公平性之间取舍？

**主链：**

`请求携带 arrival/deadline/priority` → scheduler policy 排序或选择 → token budget 分配 → 显存不足时选择 victim → aging 防止饥饿 → metrics 计算 goodput 与重计算代价 → 与 FCFS 对照。

**必读路径：**

1. `nanovllm/engine/scheduler.py::Scheduler.schedule`：把当前固定策略抽象为最小 policy 接口，保持默认行为兼容。
2. `nanovllm/engine/sequence.py::Sequence`：只加入策略真正需要的元数据，不把指标逻辑全部塞进实体。
3. `nanovllm/engine/scheduler.py::Scheduler.preempt`：记录 victim 选择和重计算成本，验证不会产生队列饥饿。
4. 第 7 课新增的 workload/metrics：增加 Poisson 到达、混合长短请求和共享前缀场景。
5. 对照 FCFS、SRPT/短任务优先、SLO-aware；同时报告吞吐、P99 TTFT、goodput 与长请求等待时间。

**结果边界：** 在实验完成前只能写“实现并比较策略”，不能写“降低 X% 延迟”。若 4GB GPU 无法稳定运行某组实验，应记录 OOM 边界，而不是删除失败结果。

**暂缓阅读：** 分布式全局调度、跨 GPU KV Cache 迁移和生产流量回放。

---

## 建议执行顺序

1. 展开第 1 课并画出一次请求的时序图。
2. 完成第 2、3 课的纯 CPU 状态与 BlockManager 测试。
3. 展开第 4、5 课，能够解释 prefill、decode 和 CUDA Graph。
4. 实施第 7 课，得到可信 baseline。
5. 只有 baseline 稳定后才实施第 8 课。
6. 第 6 课可在等待多卡资源时学习，不阻塞单卡项目。

## 第一阶段验收标准

- 能口头走通 `generate → schedule → run → attention → postprocess`。
- 能画出 `Sequence` 状态转换和逻辑 block 到物理 block 的映射。
- BlockManager 核心行为有纯 CPU 单元测试。
- benchmark 输出可复现 JSON，而不仅是一行 tokens/s。
- 所有性能结论记录模型、硬件、输入分布、预热和随机种子。
- README 明确区分上游已有功能与个人新增功能。
