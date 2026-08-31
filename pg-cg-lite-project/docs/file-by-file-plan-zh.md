# PG-CG Lite 逐文件实现记录与待验证项

## 1. 目标与状态

**目标：** 在 vLLM v0.27.1 中补齐 CUDA Graph 真实运行画像，增加一个离线动态规划器，从默认 capture-size 集合中选择最多 8 个尺寸，并在单卡 A6000 上验证初始化开销与稳态性能。

**当前状态：** 核心规划器已改为默认集合子集剪枝；按照执行环境约束，本机不运行测试，planner、日志链路、GPU 门禁和性能实验统一留到 SSH 登录的 Linux 服务器执行。

**源码基线：**

- 标签：`v0.27.1`
- commit：`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`
- 开发分支：`feature/pg-cg-lite`

## 2. 规模约束与实际结果

| 项目 | 约束 | 实际 |
|---|---:|---:|
| 代码与测试文件 | 不超过 5 个 | 5 个 |
| 新增生产代码 | 不超过 300 行 | 约 292 行 |
| 含测试总代码 | 不超过 550 行 | 约 537 行 |
| 第三方运行依赖 | 不新增 | 0 个 |
| 模型 | 1 个 | 1 个 |
| 正式 workload | 1 个 | 1 个 |
| A/B/C 组 | 3 个 | 3 个 |
| 每组重复 | 3 次 | 服务器待执行 |

明确不做：

- 不修改 scheduler、CUDA Graph dispatcher 或 CUDA kernel；
- 不增加新的服务端配置字段；
- 不做在线学习、动态重捕获或自动重启；
- 不增加 benchmark 子命令或大型实验框架；
- 不把 vLLM PR #52750 的回移补丁表述为原创创新；
- 不承诺吞吐一定上升。

## 3. 文件总览

| 文件 | 操作 | 核心职责 | 归属 |
|---|---|---|---|
| `vllm/v1/worker/gpu/model_runner.py` | 修改 | 将真实请求的 `CUDAGraphStat` 带入输出 | 上游依赖补丁 |
| `vllm/compilation/cuda_graph.py` | 修改 | 在原表格日志后输出稳定的 JSON 画像行 | PG-CG Lite |
| `tests/v1/cudagraph/test_cudagraph_logging.py` | 新增 | 验证 JSON 聚合、顺序、日志和 reset | PG-CG Lite 测试 |
| `vllm/benchmarks/pg_cg_lite.py` | 新增 | 解析日志、精确 DP、生成可直接应用的配置 | PG-CG Lite |
| `tests/benchmarks/test_pg_cg_lite.py` | 新增 | 验证解析、算法、错误输入和 CLI | PG-CG Lite 测试 |

## 4. 任务一：补齐 Model Runner V2 指标传播

### 文件

`vllm/v1/worker/gpu/model_runner.py`

### 实现内容

- 导入已有 `CUDAGraphStat`；
- 只在非 dummy run 且启用 `cudagraph_metrics` 时构造统计对象；
- 统计以下 4 个现成值：
  - `num_unpadded_tokens`；
  - `num_padded_tokens`；
  - `num_paddings`；
  - `runtime_mode`；
- 通过 `ExecuteModelState` 传递到采样和池化输出；
- 不改变调度、padding 或模型执行行为。

### 来源边界

该改动是 vLLM [PR #52750](https://github.com/vllm-project/vllm/pull/52750) 的最小回移，对应 issue #52728。提交说明必须保留来源。

### 完成证据

- 实际改动 16 行；
- `compileall` 通过；
- Ruff 检查与格式检查通过；
- `git diff --check` 通过；
- 独立提交：`fix: backport model runner v2 cudagraph metrics propagation (#52750)`。

## 5. 任务二：增加机器可读画像日志

### 生产文件

`vllm/compilation/cuda_graph.py`

### 实现内容

- 保留原有 Markdown 表格日志；
- 增加固定前缀 `PG_CG_PROFILE=`；
- 每个原有日志周期额外输出一行紧凑 JSON；
- JSON 包含：
  - `schema_version=1`；
  - 当前 `cudagraph_mode`；
  - 实际解析后的 `capture_sizes`；
  - 聚合后的 `bins`；
- `bins` 使用稳定排序，便于测试和文本 diff；
- 日志输出后沿用原有 reset 行为；
- 不增加文件句柄、后台线程或热路径搜索。

### 测试文件

`tests/v1/cudagraph/test_cudagraph_logging.py`

### 测试点

1. 重复事件能聚合为正确计数；
2. JSON 字段、排序和紧凑格式确定；
3. `log()` 先输出原表格、再输出画像行；
4. 输出后 stats 被清空。

### 完成证据

- 2 个聚焦测试通过；
- Ruff、格式、编译和 diff 检查通过；
- 独立提交：`feat: emit machine-readable cudagraph profile metrics`。

## 6. 任务三：实现 PG-CG Lite 离线规划器

### 生产文件

`vllm/benchmarks/pg_cg_lite.py`

### 输入

服务 stdout/stderr 日志，其中可能包含多个日志周期的：

```text
PG_CG_PROFILE={...}
```

### 解析规则

- 搜索行内固定前缀，不依赖日志时间戳格式；
- 合并多个周期的 `FULL` 和 `PIECEWISE` 事件；
- `CUDAGraphMode.FULL` 与 `FULL` 归一为同一模式；
- `NONE` 只计数，不参与优化；
- 拒绝空画像；
- 拒绝混合不同 capture-size 配置的日志；
- 拒绝超出原最大 capture size 的可优化事件。

### 选择算法

给定真实 token 数量直方图 `(x_i, w_i)`、默认 capture-size 集合 `C` 和尺寸预算 `K`，选择升序端点集合 `S`，最小化：

\[
L(S)=\sum_i w_i\left(\min\{s\in S\mid s\ge x_i\}-x_i\right)
\]

约束：

- 默认 `K=8`；
- `S ⊆ C`，不生成默认集合之外的新尺寸；
- `1 <= |S| <= K`；
- 必须保留原最大 capture size，不缩小覆盖上界；
- 相同 padding 先选择尺寸更少的集合，再使用字典序更小的 tuple。

实现使用默认候选上的连续区间动态规划与画像前缀和：

- 时间复杂度 `O(Km² log n)`，其中 `m=|C|`、`n` 为画像需求点数量；
- 空间复杂度 `O(Km)`；
- 默认集合只有几十个尺寸，不需要更复杂算法。

### 输出

命令：

```bash
.venv/bin/python -m vllm.benchmarks.pg_cg_lite \
  --log logs/profile-server.log \
  --max-sizes 8 \
  --output logs/plan.json
```

`plan.json` 同时包含：

- 默认与候选 capture-size 数量；
- `selection_policy`、尺寸预算和完整默认尺寸列表；
- 默认与候选预测 padding token 数；
- profile 和 NONE 事件数；
- `selected_capture_sizes`；
- 可直接传给原生 `--compilation-config` 的对象。

### 测试文件

`tests/benchmarks/test_pg_cg_lite.py`

### 20 个测试用例

1. 多日志周期合并，FULL/PIECEWISE 归一，NONE 单独计数；
2. 默认 `[1,2,4,8]`、画像 `{1:5, 3:3, 8:2}`、`K=2` 得到合法子集 `(1,8)`，padding 为 15；
3. 固定随机种子的 50 组小实例与所有合法默认子集的穷举最优解完全一致；
4. 结果属于默认集合、保留最大值且不超过预算；
5. 等 padding 时优先更少尺寸，预算足够且每个默认尺寸均命中时保留全集；
6. 空、乱序、重复、非正默认集合以及超出最大覆盖的画像均报错；
7. 空日志和混合不同 capture 配置报错；
8. 计划字段、可应用配置和新 CLI 参数保持一致。
9. 同预算等秩子集保持确定性、保留最大尺寸，并拒绝零预算。

### 服务器待验证门禁

- 规划器不导入 torch，不增加运行依赖；
- 统一从 `bash pg-cg-lite-project/scripts/verify-linux.sh` 进入，脚本在非 Linux 环境主动退出；
- 在 Linux 项目环境运行全部 20 个 planner 测试；
- 在同一环境运行画像日志测试、Ruff、格式、编译和 diff 检查；
- GPU 冒烟确认真实日志满足子集规划输入契约；
- 子集修复提交：`5f4c5a241 refactor(pg-cg): 将 planner 约束为默认 capture-size 子集`。

## 7. 任务四：A6000 验证

本任务不再修改源码。完整命令、停止条件、函数和结果脚本见：

[A6000 完整操作手册](a6000-runbook-zh.md)

固定条件：

| 参数 | 值 |
|---|---|
| 驱动起点 | `535.230.02` |
| wheel | vLLM v0.27.1 官方 `cu129` |
| Model Runner | 强制 V2 |
| 模型 | 固定本地 snapshot |
| `max_model_len` | 4096 |
| `max_num_seqs` | 128 |
| `max_num_batched_tokens` | 4096 |
| `gpu_memory_utilization` | 0.85 |
| 并发 | 16 |
| input/output | 512/128 |
| 采样 | temperature 0，ignore EOS |

### 必须依次通过的门禁

1. 普通 CUDA matmul；
2. `torch.compile`/Triton JIT；
3. 2+20 个聚焦测试和 Ruff；
4. vLLM CUDA Graph 服务冒烟；
5. 非空 `PG_CG_PROFILE=`；
6. 计划默认尺寸数大于 8、候选不超过 8；
7. 20/20 输出完全一致；
8. A/B/C 各 3 次都完成且无失败请求。

### R535 兼容策略

- `nvidia-smi` 的 CUDA 12.2 不是 Toolkit 版本；
- 强制 `cu129`，不使用 `auto` 或 `cu130`；
- R535 可以尝试 CUDA 12.x 小版本兼容；
- 任何 PTX/JIT/driver 错误都停止；
- 必要时升级到 `575.57.08` 或更新生产驱动后重试。

### 正式 A/B/C

- A：vLLM 默认 capture sizes；
- B：不读取画像的同预算等秩默认子集；
- C：画像生成的默认集合最优子集；
- 次序：`A1 → B1 → C1 → C2 → A2 → B2 → B3 → C3 → A3`；
- 每轮重启服务，等待 GPU 回到空闲温度；
- 只报告 3 次原始值和中位数，不挑最好一次。

### 结果指标

- capture-size 数量；
- 默认、等秩与 PG 子集的预测 padding；
- time-to-ready；
- `Graph capturing finished` 时间；
- CUDA Graph capture 显存；
- request throughput；
- median/p95 TTFT 与 TPOT；
- 20 条输出一致性。

## 8. 最终源码验证清单

服务器安装完整依赖后执行：

```bash
.venv/bin/python -m pytest \
  --confcutdir=tests/v1/cudagraph \
  tests/v1/cudagraph/test_cudagraph_logging.py -q

.venv/bin/python -m pytest \
  --confcutdir=tests/benchmarks \
  tests/benchmarks/test_pg_cg_lite.py -q

.venv/bin/ruff check \
  vllm/v1/worker/gpu/model_runner.py \
  vllm/compilation/cuda_graph.py \
  vllm/benchmarks/pg_cg_lite.py \
  tests/v1/cudagraph/test_cudagraph_logging.py \
  tests/benchmarks/test_pg_cg_lite.py

.venv/bin/ruff format --check \
  vllm/v1/worker/gpu/model_runner.py \
  vllm/compilation/cuda_graph.py \
  vllm/benchmarks/pg_cg_lite.py \
  tests/v1/cudagraph/test_cudagraph_logging.py \
  tests/benchmarks/test_pg_cg_lite.py

git diff --check v0.27.1..HEAD
git diff --name-only v0.27.1..HEAD
git status --short
```

## 9. 完成定义

以下条件全部满足才将项目标记为完成：

- 代码和测试只涉及约定的 5 个文件；
- 画像日志来自真实 A6000 请求，不是手工样例；
- 规划器输出不超过 8 个 sizes，并保留默认最大上界；
- 默认数量确实大于 8，形成清晰结构对比；
- 20/20 输出一致；
- A/B/C 各完成 3 次，并完成一次轻量 shift 检查；
- 原始日志、9 个主实验 benchmark JSON、3 个 shift JSON 和 `plan.json` 全部保留；
- 自动生成 `summary.json`、`results.md` 和 1 张对比图；
- 报告明确上游补丁来源和单机单 workload 的结论边界。

在服务器不可访问的当前阶段，代码实现已完成，但 GPU 实验尚未完成；不能提前填写或虚构性能数字。
