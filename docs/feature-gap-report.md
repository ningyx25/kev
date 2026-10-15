# 特性差距报告：旧 kev-4b 部署服务 vs 当前 kev-serve

> 对比对象：旧 monorepo 中部署 `jaredpalmer/kev-4b` 的服务（`kev/serve.py`，`Server` 类，505 行）
> 与当前项目 `kev-serve`（`kev_serve/`，clef + gui_actor 后端，`SyncLockEngine`）。
> 报告只收录**当前项目没有、且可实现**的特性，按优先级分组，每项给出旧实现位置、缺口、落地路径、依赖、风险与工作量估计。

---

## 背景

旧 `kev/serve.py` 在服务 kev-4b（Qwen3.5-4B-Base 混合骨干 + LoRA + pointer head）时，围绕吞吐与长文档构建了一整套引擎：连续批处理、state 前缀缓存、CUDA graphs、融合 kernel、MLX 路径、65k 上下文准入与截断、温度标定透出。当前 kev-serve 是从 monorepo 抽离的 orphan branch，保留了 TypeSafe API 契约和多后端路由，但引擎被裁剪为锁串行的 `SyncLockEngine`，且只服务 clef / gui_actor 两个后端。

报告中的旧实现引用格式为 `旧仓库路径:行号`，当前项目引用格式为 `kev_serve/路径:行号`。

---

## P0 — 纯服务层，无模型依赖，可立即实施

### P0-1 长上下文准入与截断

- **旧实现**：`kev/model.py:18`（`SERVE_MAX_STATE = 65536`）、`kev/model.py:130`（`admit()`：超长 state 抛 `ContextOverflow` 并附 token 数）、`kev/serve.py:41`（`KEV_TRUNCATE_STATES=1` 截断模式）、`kev/serve.py:222-231`（响应带 `truncated`、`usage.state_tokens` / `state_tokens_used`）
- **缺口**：当前项目对 state 长度无任何检查，超长输入直接进入后端，由后端/transformers 静默截断或报难懂的错误
- **实现路径**：
  1. 在 `kev_serve/api/request.py` 或新模块中加 token 计数准入：请求进入后端前用后端的 tokenizer 编码 state，超过上限抛 `HTTPException(422, "...; or start with KEV_TRUNCATE_STATES=1 ...")`
  2. `Settings` 加 `max_state_tokens` / `truncate_states`（环境变量 `KEV_MAX_STATE_TOKENS` / `KEV_TRUNCATE_STATES`），默认上限按 backbone 的 `max_position_embeddings` 推导，缺省回退 65536
  3. `http_app.py:62` 路由在 `_to_backend_dict` 之后、`backend.answer()` 之前执行准入；截断模式下在响应体补 `truncated` 与 `usage.state_tokens*`
- **依赖**：后端需暴露 tokenizer（`Backend` 协议加一个可选属性即可，clef / gui_actor 的 processor 均内含）
- **风险**：低。需注意 question 分支的预算（旧代码 `SERVE_MAX_BRANCH = state + 8192`），否则"state 卡在上限、问题再超"会漏检
- **工作量**：0.5–1 天

### P0-2 温度标定透出（`KEV_TEMPERATURE`）

- **旧实现**：kev-4b 的 `head.pt` 携带 T=2.41，加载时自动应用；`KEV_TEMPERATURE=1.0` 返回原始概率；`/v1/models` 卡片与描述中暴露 `temperature`（`kev/serve.py:423`）
- **缺口**：当前 `Settings`（`kev_serve/config.py`）无温度概念；clef 后端的校准温度（若有）既不可见也不可调；模型卡无 `temperature` 字段
- **实现路径**：`Backend.model_card()` 增加 `temperature`（值来自 checkpoint 元数据或 1.0）；`Settings` 加 `temperature` 覆盖项，在 `answer()` 返回前对 probabilities 做温度重标定（logits 不可得时对概率做 power 变换：`p^(1/T)` 后归一化，需在文档注明这是近似）
- **依赖**：确认 clef checkpoint 是否存有温度元数据；若模型输出的是 logits 则可精确重标定
- **风险**：低。概率域的 temperature 近似与 logits 域不完全等价，需测试锁定
- **工作量**：0.5 天

### P0-3 `--fallback` 检查点

- **旧实现**：`kev/serve.py:460-467`：`--run` 指向的 checkpoint 不存在时回退到 `--fallback`（默认 `runs/smoke`）并打印提示
- **缺口**：当前 `cli.py:71` 加载失败直接 `SystemExit`
- **实现路径**：`cli.py` 加载前加一次存在性检查（本地目录 / hub id），失败时用 fallback 路径重试并 `print` 提示；`--fallback` 参数默认 `None`（None = 不回退，保持现状语义）
- **风险**：低。注意 fallback 也要过后端 sniff
- **工作量**：0.5 天以内

### P0-4 HuggingFace Hub 直拉（`--run jaredpalmer/kev-4b@v1.0`）

- **旧实现**：`kev/checkpoint.py` 的 `is_hub_id()`；`kev/serve.py:114-116` 的 `--run <hub-id>[@revision]`；release date 从 Hub commit 解析（`kev/serve.py:119`）
- **缺口**：当前 `backends/base.py:42` 的 `load_backend` 只吃本地路径；`cli.py:19` 的 `parse_run` 虽在注释里说明"Hub id 保留自己的 `@revision`"，但加载链路上没有任何 hub 解析（无 `snapshot_download`）
- **实现路径**：`load_backend()` 之前加 `resolve_checkpoint(path)`：路径不存在且形如 `org/name[@rev]` 时 `huggingface_hub.snapshot_download` 到本地缓存再 sniff；revision 用 `@` 后缀解析（注意与 `@N` GPU 后缀的歧义，`cli.py:21` 的注释已有约定可沿用）；release date 从 `HfApi.model_info().lastModified` 取
- **依赖**：`huggingface_hub`（clef / gui_actor 的加载链已经间接依赖 transformers，引入成本低）
- **风险**：低-中。私有 checkpoint 需要 token；缓存目录策略（`HF_HOME`）需文档化
- **工作量**：1 天

### P0-5 演示/研究端点：`permute` 与 `separate`

- **旧实现**：`kev/serve.py:366`（`POST /v1/systemone/permute`：同一 choice 问题跑 n_perm 种选项顺序，返回每种顺序的 probabilities、`argmax_stable`、per-option `spread`）、`kev/serve.py:383`（`POST /v1/systemone/separate`：每问题一次独立前向，用于 packed-vs-separate 对比）
- **缺口**：当前 `http_app.py` 只有 `/v1/systemone` 和 `/v1/models` 两个路由
- **实现路径**：新增两个路由函数，复用 `_to_backend_dict` + `backend.answer()`；`permute` 按种子洗牌 criteria 顺序多次调用；`separate` 按问题拆分多次调用并聚合 usage/latency。guard 保持与旧版一致：仅对文本类后端开放（gui_actor 可 422 拒绝或同样支持，语义上仍成立）
- **依赖**：无（依赖 P0-1 的截断标记传播时参考 `kev/serve.py:398` 的 `truncation_marks`）
- **风险**：低。`permute` 的 `n_perm` 上限需保留（旧版 `ge=1, le=64`，防无界循环）
- **工作量**：1 天

---

## P1 — 引擎层，接口已预留，需后端配合

### P1-1 连续批处理引擎（`QueueBatchEngine`）

- **旧实现**：`kev/serve.py:104-202`：`Server` 持有一条模型线程 + `queue.Queue`，空闲时把队列里所有请求（`MAX_BATCH = 64`，`kev/serve.py:44`）合并成一个 batch 走 `model.probs_batch`；`kev/serve.py:129` 把 `sys.setswitchinterval` 降到 0.5 ms，避免模型线程在 CUDA sync 处被 GIL 拖慢（实测批时间 ~2x 差异）
- **缺口**：当前 `engine/base.py:17` 只有 `SyncLockEngine`（锁 + 线程池，一次一个请求）；`QueueBatchEngine` 仅存在于 docstring 的接口约定中
- **实现路径**：按 `engine/base.py:1-6` 的约定实现 `QueueBatchEngine`：模型线程 + 队列 + `asyncio.Future`；`Backend` 协议增加可选的 `answer_batch(list[dict]) -> list[dict]`。clef 后端的 forward 若支持 batch（transformers 路径通常支持），即可受益；gui_actor 因每pass batch=1 保持锁语义
- **依赖**：后端模型能接受 batch 输入；需要重写请求编码为张量化 batch（旧 `probs_batch` 在 `kev/model.py`，属 kev 模型代码，移植时按 P3 的约定处理）
- **风险**：中。batch 内单请求异常的处理、动态 shape 下的显存峰值、`latency_ms` 语义（旧版报告的是 batch 时间而非排队时间）均需与旧行为对齐并测试
- **工作量**：3–5 天（含测试）

### P1-2 State 前缀缓存（`PrefixCache`）

- **旧实现**：`kev/serve.py:48-100`：LRU 缓存每个 state 的 KV / DeltaNet 状态（含 hidden states，取决于骨干），跨请求复用；`KEV_PREFIX_CACHE`（条数，默认 4）、`KEV_PREFIX_MIN_TOKENS`、`KEV_PREFIX_MAX_TOKENS`（默认 65536）三档可调；`plan/make_room/store` 处理 batch 内的驱逐顺序
- **缺口**：当前项目无任何前缀缓存；重复 state 的请求每次全额 prefill
- **实现路径**：后端暴露 `prefix(state) -> handle` 与 `answer(dict, prefix=handle)` 两段式接口；引擎在 `QueueBatchEngine` 内挂 `PrefixCache`（可直接移植旧逻辑，其为纯 Python、不依赖 kev 模型）。transformers 后端用 `past_key_values` 作缓存载体；混合骨干需同时缓存 recurrent state
- **依赖**：P1-1（缓存语义是围绕 batch 定义的）；后端支持增量 prefill
- **风险**：中。显存预算（旧注释：Kev-27B 一个 64k state ~1.3 GB；MLX 上 2.2 GB）需按模型实测并在 `/v1/models` 暴露占用；驱逐时机不当会峰值翻倍
- **工作量**：3–5 天（含测试）

### P1-3 OOM 韧性与统计

- **旧实现**：`kev/serve.py:183-196`：batch 因缓存 state 耗尽显存时，清缓存 + `empty_cache` 后重跑一次；`prefix_cache.oom_retries` 计数并暴露在 `/v1/models`
- **缺口**：当前无任何 OOM 恢复路径
- **实现路径**：P1-2 落地后在 `_run` 重试循环中按旧逻辑处理；`out_of_memory(e)` 判定可移植（`kev/device.py`）
- **依赖**：P1-2
- **风险**：低（逻辑已退役验证过，含 issue #75 的场景）
- **工作量**：0.5 天（跟随 P1-2）

---

## P2 — 硬件 / 运行时特定优化

### P2-1 CUDA graphs

- **旧实现**：`kev/cuda_graphs.py`（约 500 行：长度分桶 `bucket/length_groups`、`BufferKV` 动态 KV 缓冲、空闲时 `capture_pending`）；`kev/serve.py:491` 在 CUDA 上默认开启，`/v1/models` 暴露 `cuda_graphs.stats()`
- **缺口**：当前无
- **实现路径**：两种选择——(a) 移植旧 `cuda_graphs.py`（它按 kev 模型结构定制，需按 clef 的 Qwen3.5 VL 结构调整）；(b) 用 `torch.cuda.CUDAGraph` + `transformers` 的 `static_cache`（`StaticCache` for generation / `sdpa` 路径）做等价物。建议先评估 (b)，社区方案维护成本低
- **依赖**：P1-1 的 batch 路径（graph 重放的收益在批量场景最大）；固定 shape 缓冲意味着要接受长度分桶的近似
- **风险**：中-高。混合骨干的 DeltaNet 层不在 transformers 标准图捕获路径内，这是旧版自研的主因；显存缓冲需探测（`Buffers(probe, tokens, entries, device)`）
- **工作量**：1–2 周（含正确性对齐测试：served vs fp32 概率差 < 0.017 的旧门槛）

### P2-2 Fused Qwen3.5 kernels

- **旧实现**：`kev/fused_qwen35.py` + 固定版本的 `flash-linear-attention`（`FLA_VERSION` pin）；`kev/serve.py:493` 默认启用（可用时），每 batch GPU 时间约少 1/3；未安装时打印提示
- **缺口**：当前无
- **实现路径**：确认 clef 的 Qwen3.5 骨干与旧 `fused_qwen35.py` 目标结构一致（同 base 系列，概率高）；vendor 该文件 + 在 `pyproject.toml` 增加 `vision-kernels` extra 的等价依赖并 pin 版本；加载时 sniff 可用性，不可用时静默回退 eager
- **依赖**：P3 或 clef 后端确认骨干结构一致；`fla` 与当前 transformers 版本的兼容性
- **风险**：中。版本矩阵（fla × transformers × torch × CUDA）需实测；回退路径必须默认安全
- **工作量**：3–5 天（若骨干结构一致）

### P2-3 MLX 后端（Apple Silicon）

- **旧实现**：`kev/mlx_model.py`（`MLXDecisionModel`：`PREFILL_CHUNK = 1024` 分块 prefill、fp32 recurrent state、chunked 与 one-pass 差异仅浮点重结合 ~1e-6）；`kev/serve.py:438-444,469,488`：hybrid checkpoint 在 macOS 上默认走 MLX，跳过 torch MPS 探测，attn 默认 sdpa
- **缺口**：当前 `device.py` 的 `mps` 是 torch 路径，无 MLX
- **实现路径**：作为第三个运行时后端：`backends/` 下加 `mlx` 变体或后端内双路径；`pyproject.toml` 增加 `mlx` extra（仅 macOS 可装）；LoRA merge 逻辑可移植旧 `mlx_model.py:97`
- **依赖**：kev 决策模型后端（P3）——MLX 路径只对 kev 家族 checkpoint 有意义（clef 是 VL 模型，MLX 支持另论）
- **风险**：中。MLX 与 transformers 的数值对齐需基线测试（旧标准：60 题概率差 < 0.018，无答案翻转）
- **工作量**：1 周

---

## P3 — 最大单项：kev 决策模型后端

- **旧实现**：整个 `kev` 模型家族（`model.py`、`checkpoint.py` 的 LoRA 加载与合并、`clef_model.py` 的姊妹实现、`cuda_graphs.py`、`mlx_model.py`），服务 `jaredpalmer/kev-4b`（LoRA r16 + pointer head on Qwen3.5-4B-Base，24 Gated DeltaNet + 8 全注意力层，T=2.41）
- **缺口**：当前项目只有 clef（noul/choice/score on Qwen3.5 VL）与 gui_actor（ground）；kev pointer-head 后端不存在，`engine/base.py:1-6` 的 `QueueBatchEngine` 说明直言"waits for the kev pointer head"
- **实现路径**：按当前架构这是最自然的一次扩展——
  1. 在 `backends/` 下新增 `kev/` 包，`__init__.py` 调 `register_backend("kev", sniff, factory)`（sniff：目录含 `head.pt` + adapter）
  2. vendor 所需模型代码（遵守本仓库既定约定：in-tree vendored、保留上游 licence、写 `ATTRIBUTION.md`、`tests/test_baseline.py` 锁定行为，参见 `backends/clef/ATTRIBUTION.md` 与 `backends/gui_actor/ATTRIBUTION.md` 的既有做法）
  3. `Backend` 实现 `answer()` / `model_card()`（卡片可一次性做厚：temperature、max_state_tokens、truncate_states、cuda_graphs / prefix_cache / batches 统计，对齐 `kev/serve.py:422-431`）
  4. 引擎切到 `QueueBatchEngine` + `PrefixCache`（P1）
- **依赖**：P1-1/P1-2 才能发挥旧版的吞吐优势；P2 各项均以它为前提
- **风险**：中-高。orphan branch 的约束是"imports nothing from the kev package"，移植即 vendored 代码，需评估许可与体积；混合骨干（DeltaNet）在纯 transformers 下不被原生支持，这正是旧版自研 `cuda_graphs`/`mlx_model` 的原因，移植前需确认目标 transformers 版本对 `qwen3_5` 混合架构的支持程度
- **工作量**：2–4 周（模型加载与数值对齐 1 周，引擎集成 1 周，测试与文档 1–2 周）

---

## 汇总矩阵

| # | 特性 | 旧实现 | 优先级 | 工作量 | 主要风险 |
|---|---|---|---|---|---|
| P0-1 | 长上下文准入/截断（422 + token 数 + `truncated`） | `kev/model.py:130`, `kev/serve.py:41,222` | P0 | 0.5–1 天 | 低 |
| P0-2 | `KEV_TEMPERATURE` + 卡片温度 | `kev/serve.py:423` | P0 | 0.5 天 | 低（概率域近似） |
| P0-3 | `--fallback` 检查点 | `kev/serve.py:460` | P0 | <0.5 天 | 低 |
| P0-4 | Hub 直拉 + revision pin | `kev/serve.py:114`, `kev/checkpoint.py` | P0 | 1 天 | 低-中 |
| P0-5 | `/permute`、`/separate` 演示端点 | `kev/serve.py:366,383` | P0 | 1 天 | 低 |
| P1-1 | `QueueBatchEngine` 连续批处理 | `kev/serve.py:104` | P1 | 3–5 天 | 中 |
| P1-2 | `PrefixCache` state 前缀缓存 | `kev/serve.py:48` | P1 | 3–5 天 | 中（显存） |
| P1-3 | OOM 韧性 + `oom_retries` | `kev/serve.py:183` | P1 | 0.5 天 | 低 |
| P2-1 | CUDA graphs | `kev/cuda_graphs.py` | P2 | 1–2 周 | 中-高 |
| P2-2 | Fused Qwen3.5 kernels | `kev/fused_qwen35.py` | P2 | 3–5 天 | 中（版本矩阵） |
| P2-3 | MLX 后端 | `kev/mlx_model.py` | P2 | 1 周 | 中（依赖 P3） |
| P3 | kev 决策模型后端（pointer head） | 整个 `kev` 模型家族 | P3 | 2–4 周 | 中-高（vendoring、混合骨干） |

## 建议路线图

1. **第一阶段（P0，约 1 周）**：把纯服务层缺口补齐——准入/截断、温度、fallback、Hub 直拉、演示端点。这批不触碰引擎与模型，收益直接（部署体验、可观测性、与研究脚本的兼容）。
2. **第二阶段（P1，约 1.5–2 周）**：实现 `QueueBatchEngine` + `PrefixCache` + OOM 韧性。先以 clef 后端验证（ transformers `past_key_values` 作缓存载体），接口按 `engine/base.py` 既有约定，为 P3 铺平道路。
3. **第三阶段（P2，按需）**：硬件特定优化按用户群决定——NVIDIA 用户多看 CUDA graphs / fused kernels，Mac 用户等 MLX（且依赖 P3）。
4. **第四阶段（P3）**：kev 决策模型后端落地，同步把 `/v1/models` 卡片做厚（旧版全部统计字段一次性到位），并遵守本仓库的 vendored 代码约定（ATTRIBUTION + baseline test 锁定）。

## 附注

- 两边**已共有**、不在本报告范围的特性：TypeSafe `/v1/systemone` 契约、CORS、`x-typesafe-request-id` + `server-timing`、`KEV_API_KEY` bearer 鉴权、`KEV_DATE_FACTS` 日期预处理、`kev-latest`/`jev-latest` 别名、多模型路由。
- `box` 原语后端（GUI-Actor per-patch 分布聚合连通区域）是当前项目自己文档中的"planned"项（`backends/base.py:6-9`），旧 kev-4b 部署同样未实现，故不计入差距。
- 旧版 kev-4b 的服务指标（L40S 51.4 req/s、H100 100.8 req/s、bf16 与 fp32 概率差 < 0.017、MLX 8k state 6.6 s prefill / 354 ms 缓存命中）可作为新引擎落地后的验收基线参考。
