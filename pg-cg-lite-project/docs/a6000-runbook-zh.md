# PG-CG Lite：A6000 完整操作手册

> 目标：你拿到服务器后，从空目录开始，按本文顺序完成版本固定、环境门禁、单元测试、画像、计划生成、确定性正确性检查、性能模式预热、9 次 A/B/C 主实验、轻量 workload-shift 检查、结果汇总和证据归档。除用户名、服务器地址和工作目录外，不需要临时决定实验参数。

本文是项目唯一的完整操作入口。旧 `feature/pg-cg-lite` 分支及
`logs-performance-20260828-185154.tar.gz` 只属于历史预实验：其中候选尺寸不满足当前“默认集合子集”契约，也没有同预算 B 组，不能复用旧 `plan.json`、旧配置或旧结果作为正式证据。旧结果的完整审计见
[2026-08-28 历史预实验审计](legacy-results-20260828-zh.md)。

## 0. 本手册对应的固定版本

| 项目 | 固定值 |
|---|---|
| 工作分支 | `fix/pg-cg-lite-hardening` |
| GPU | 单卡 NVIDIA RTX A6000 48 GiB，计算能力 `8.6` |
| 当前驱动 | `535.230.02` |
| `nvidia-smi` 顶部 CUDA 字段 | `CUDA Version: 12.2` |
| vLLM 基线 | `v0.27.1`，`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| vLLM 轮子 | 官方 `vllm-0.27.1+cu129` x86_64 轮子 |
| Python | 3.12 |
| 模型 | `Qwen/Qwen2.5-7B-Instruct` |
| 模型 revision | `a09a35458c702b33eeacc393d103063234e8bc28` |
| dtype | BF16 |
| 正式 workload | 随机输入 512 token、输出 128 token、并发 16 |
| 画像请求数 | 300 |
| 正确性请求数 | 20 |
| 性能请求数 | 每次 500 |
| 正确性模式 | `VLLM_BATCH_INVARIANT=1`，A/B/C 输出逐字一致 |
| 性能模式 | 显式取消 `VLLM_BATCH_INVARIANT`，不做跨轮文本比较 |
| 性能分位数 | P95、P99 |
| A/B/C 次序 | `A1 → B1 → C1 → C2 → A2 → B2 → B3 → C3 → A3` |
| PG-CG Lite | `K=8` |

交付补丁中的 5 个提交按职责拆分为：

1. 回移 vLLM PR #52750 的 Model Runner V2 指标传播修复；
2. 增加 `PG_CG_PROFILE=` 机器可读日志；
3. 增加标准库实现的离线动态规划器；
4. 将规划器约束为默认 capture-size 集合的子集；
5. 增加同预算等秩基线及对应计划字段。

## 1. 先理解驱动结论

`nvidia-smi` 顶部的 `CUDA Version: 12.2` 表示当前驱动公开的最高 CUDA Driver API 能力，不等于服务器安装了 CUDA Toolkit 12.2，也不等于 Python 中 PyTorch 实际携带的 CUDA runtime。

NVIDIA 官方说明：Linux 上 CUDA 12.x 的小版本兼容最低驱动是 `525.60.13`，所以 `535.230.02` 可以尝试运行 CUDA 12.9 二进制；但小版本兼容存在限制，特别是使用 PTX/JIT 时可能要求更新驱动。CUDA 12.9 Update 1 对应的原生驱动是 `575.57.08`。参考：

- [NVIDIA CUDA 小版本兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [CUDA 12.9 Update 1 Release Notes](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html)
- [vLLM v0.27.1 GPU 安装文档](https://docs.vllm.ai/en/v0.27.1/getting_started/installation/gpu/)
- [vLLM v0.27.1 Release](https://github.com/vllm-project/vllm/releases/tag/v0.27.1)

因此本文采用以下规则：

- 保留现有 R535 时，必须显式使用 `cu129`，不能使用 `cu130`，也不要依赖 `auto` 猜测。
- 安装后必须依次通过普通 CUDA、`torch.compile`/Triton、vLLM CUDA Graph 三层门禁。
- 出现 `unsupported PTX`、`driver too old`、`cudaErrorCallRequiresNewerDriver` 或 `CUDA driver version is insufficient` 时立即停止；将驱动升级到 `575.57.08` 或更新的生产分支后，从环境门禁重新开始。
- 不设置 `NVIDIA_DISABLE_REQUIRE=true`，不靠关闭兼容性检查掩盖问题。

## 2. 默认前提与目录

下面命令假设服务器是 Ubuntu 22.04/24.04 x86_64，使用 Bash，有 sudo 权限，服务器可访问 GitHub、PyPI、PyTorch 和 Hugging Face。若模型已在本地，使用第 9 节的本地模型分支即可。

先在本地 Windows PowerShell 登录服务器（替换用户名和地址）：

```powershell
ssh 你的用户名@服务器地址
```

从这里到第 20 节的命令全部在远端 Linux Bash 中执行。本地工作站不运行测试、PyTorch、vLLM 或性能实验。登录后先执行：

```bash
set -euo pipefail

export PGCG_ROOT="$PWD/pg-cg-lite-work"
export PGCG_REPO="$PGCG_ROOT/vllm"
export PGCG_LOG_DIR="$PGCG_ROOT/logs"
export PGCG_MODEL_DIR="$PGCG_ROOT/models/Qwen2.5-7B-Instruct"
export PGCG_PORT=8000
export PGCG_BRANCH="fix/pg-cg-lite-hardening"
export PGCG_MAX_NUM_SEQS=128
export PGCG_GPU_MEMORY_UTILIZATION=0.85

mkdir -p "$PGCG_ROOT" "$PGCG_LOG_DIR" "$PGCG_ROOT/models" "$PGCG_ROOT/wheels"
```

建议至少预留：

- 80 GiB 可用磁盘；
- 32 GiB 系统内存，推荐 64 GiB；
- GPU 上没有其他计算进程。

## 3. 记录服务器原始状态

```bash
cd "$PGCG_ROOT"

{
  date --iso-8601=seconds
  uname -a
  cat /etc/os-release
  nvidia-smi
  nvidia-smi -L
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
    --format=csv,noheader
  nvcc --version 2>&1 || true
  df -h "$PGCG_ROOT"
  free -h
} | tee "$PGCG_LOG_DIR/00-host-before.txt"
```

人工确认：

- 只有目标 A6000，或后续通过 `CUDA_VISIBLE_DEVICES` 只暴露目标卡；
- 驱动是 `535.230.02`；
- 显存约 48 GiB；
- `nvcc` 不存在也不构成失败，因为本项目只改 Python，使用预编译轮子。

如果 GPU 上已有计算进程，先联系进程所有者；不要直接终止不属于你的进程。

## 4. 安装基础工具

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  ca-certificates curl git jq ripgrep build-essential

curl -LsSf https://astral.sh/uv/install.sh | \
  env UV_INSTALL_DIR="$PGCG_ROOT/bin" sh
export PATH="$PGCG_ROOT/bin:$PATH"
uv self version
```

通过条件：`uv self version` 正常显示版本。

## 5. 获取 hardening 分支并固定源码

直接克隆项目 hardening 分支。这样源码、测试、验证脚本、补丁交付件和本文会处于同一个提交；不要只把五个补丁应用到上游浅克隆后再调用并不存在的项目脚本。

```bash
cd "$PGCG_ROOT"
git clone --branch "$PGCG_BRANCH" --single-branch \
  https://github.com/WeiAsFan/vllm-pg-cg-lite.git "$PGCG_REPO"
cd "$PGCG_REPO"

git fetch origin \
  "+refs/heads/$PGCG_BRANCH:refs/remotes/origin/$PGCG_BRANCH"
test "$(git rev-parse HEAD)" = "$(git rev-parse "origin/$PGCG_BRANCH")"
test "$(git branch --show-current)" = "$PGCG_BRANCH"
test -x pg-cg-lite-project/scripts/verify-linux.sh
test -z "$(git status --porcelain)"
git status --short

{
  printf 'branch=%s\n' "$(git branch --show-current)"
  printf 'commit=%s\n' "$(git rev-parse HEAD)"
  printf 'remote=%s\n' "$(git remote get-url origin)"
} | tee "$PGCG_LOG_DIR/01-repository.txt"
```

通过条件：本地 `HEAD` 与远端 hardening 分支一致，验证脚本存在且 `git status --short` 没有输出。正式实验开始后不要在该 checkout 中修改文件或切换提交。

## 6. 核对 vLLM 基线与五补丁交付件

```bash
cd "$PGCG_REPO"
git fetch https://github.com/vllm-project/vllm.git \
  refs/tags/v0.27.1:refs/tags/v0.27.1
test "$(git rev-parse 'v0.27.1^{commit}')" = \
  "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac"

export PGCG_PATCH_DIR="$PGCG_REPO/pg-cg-lite-project/patches"
test "$(find "$PGCG_PATCH_DIR" -maxdepth 1 -type f -name '*.patch' | wc -l)" -eq 5
(cd "$PGCG_PATCH_DIR" && sha256sum --check SHA256SUMS.txt)

git log --oneline --decorate v0.27.1..HEAD
git diff --name-only v0.27.1..HEAD -- \
  tests/benchmarks/test_pg_cg_lite.py \
  tests/v1/cudagraph/test_cudagraph_logging.py \
  vllm/benchmarks/pg_cg_lite.py \
  vllm/compilation/cuda_graph.py \
  vllm/v1/worker/gpu/model_runner.py
test -z "$(git status --porcelain)"
git status --short
```

通过条件：

- `v0.27.1` 标签解析到固定基线提交；
- 5 个补丁哈希均为 `OK`；
- 上述限定范围的 `git diff --name-only` 显示以下 5 个生产代码与测试路径：

```text
tests/benchmarks/test_pg_cg_lite.py
tests/v1/cudagraph/test_cudagraph_logging.py
vllm/benchmarks/pg_cg_lite.py
vllm/compilation/cuda_graph.py
vllm/v1/worker/gpu/model_runner.py
```

- `git status --short` 没有输出。hardening 分支还包含项目文档、脚本和补丁交付件，因此不要求整个 `v0.27.1..HEAD` 差异只有这 5 个路径。

## 7. 安装固定的 cu129 开发环境

先下载并校验 vLLM 官方轮子：

```bash
cd "$PGCG_REPO"

export PGCG_VLLM_WHEEL="$PGCG_ROOT/wheels/vllm-0.27.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
curl -fL --retry 3 --retry-delay 5 \
  -o "$PGCG_VLLM_WHEEL" \
  'https://github.com/vllm-project/vllm/releases/download/v0.27.1/vllm-0.27.1%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl'

echo 'bf0d52faa2a51e7a01c6856a7a8a2d1307fd0ff711415d34168a67ffac0fa47b  '"$PGCG_VLLM_WHEEL" | \
  sha256sum --check
```

预期输出包含 `OK`。然后创建全新环境并用官方“Python-only editable build”方式安装：

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate

export VLLM_USE_PRECOMPILED=1
export VLLM_MAIN_CUDA_VERSION=12.9
export VLLM_PRECOMPILED_WHEEL_LOCATION="$PGCG_VLLM_WHEEL"

uv pip install --editable . --torch-backend=cu129
uv pip install -r requirements/lint.txt
uv pip install -r requirements/test/cuda.in
uv pip install matplotlib==3.9.2
.venv/bin/pre-commit install

unset VLLM_USE_PRECOMPILED
unset VLLM_MAIN_CUDA_VERSION
unset VLLM_PRECOMPILED_WHEEL_LOCATION
```

不要把 `--torch-backend=cu129` 改成 `auto`。本项目不修改 C++/CUDA kernel，所以使用预编译 wheel 支撑 Python-only editable 安装；测试和 lint 依赖仍按上游 requirements 安装，所有 Python 包管理都通过 `uv`。

## 8. 环境三层门禁

### 8.1 门禁一：普通 CUDA

```bash
cd "$PGCG_REPO"
source .venv/bin/activate

.venv/bin/python - <<'PY' | tee "$PGCG_LOG_DIR/02-python-cuda.txt"
import inspect
from pathlib import Path

import torch
import vllm

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("vLLM:", vllm.__version__)
print("vLLM source:", Path(inspect.getfile(vllm)).resolve())
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))

assert torch.cuda.is_available()
assert torch.version.cuda and torch.version.cuda.startswith("12.9")
assert "A6000" in torch.cuda.get_device_name(0)
assert torch.cuda.get_device_capability(0) == (8, 6)

x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
y = x @ x
torch.cuda.synchronize()
print("matmul checksum:", y.float().norm().item())
PY
```

### 8.2 门禁二：TorchInductor/Triton JIT

```bash
.venv/bin/python - <<'PY' | tee "$PGCG_LOG_DIR/03-torch-compile.txt"
import torch

@torch.compile(fullgraph=True)
def fused(x):
    return torch.nn.functional.silu(x + 1.0) * 0.5

x = torch.randn(1_048_576, device="cuda")
for _ in range(3):
    y = fused(x)
torch.cuda.synchronize()
print("torch.compile checksum:", y.sum().item())
PY
```

### 8.3 门禁三：项目单元测试

统一执行项目提供的 Linux 固定验证入口。脚本会拒绝在非 Linux 环境运行，并依次校验补丁哈希、画像日志测试、planner 测试以及两项 Ruff 检查：

```bash
bash pg-cg-lite-project/scripts/verify-linux.sh \
  |& tee "$PGCG_LOG_DIR/04-linux-verification.txt"
printf 'linux_verification_exit=0\n' | \
  tee -a "$PGCG_LOG_DIR/04-linux-verification.txt"
```

以下命令是该入口的展开形式，仅用于排查某一步失败；正式留档以脚本输出为准：

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
```

预期补丁哈希全部通过、2 个画像日志测试通过、20 个 planner 测试通过、两个 Ruff 命令通过。

### 8.4 任一 CUDA 门禁失败时怎么做

如果 8.1 或 8.2 出现以下任一关键词，不要继续下载模型或压测：

```text
unsupported PTX
provided PTX was compiled with an unsupported toolchain
CUDA driver version is insufficient
cudaErrorCallRequiresNewerDriver
no kernel image is available
```

Ubuntu 上先查看可用驱动包：

```bash
sudo apt-get install -y ubuntu-drivers-common
ubuntu-drivers list
apt-cache search '^nvidia-driver-[0-9]+$' | sort -V | tail -20
```

选择发行版提供的 `575.57.08` 或更新生产驱动；例如仓库存在 `nvidia-driver-580` 时：

```bash
apt-cache policy nvidia-driver-580
sudo apt-get install -y nvidia-driver-580
sudo reboot
```

重启后重新登录，先恢复第 2 节环境变量，再从第 3 节重新记录环境并重跑第 8 节。不要混用 apt 驱动和 NVIDIA `.run` 安装器。若你无驱动升级权限，把 `00-host-before.txt`、`02-python-cuda.txt`、`03-torch-compile.txt` 交给管理员即可；在门禁通过前不生成实验结论。

## 9. 固定模型快照

### 9.1 可联网时

```bash
cd "$PGCG_REPO"
source .venv/bin/activate

.venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    revision="a09a35458c702b33eeacc393d103063234e8bc28",
    local_dir=os.environ["PGCG_MODEL_DIR"],
)
print(path)
PY
```

### 9.2 已有本地模型时

把完整模型目录复制到 `$PGCG_MODEL_DIR`。它必须对应第 0 节固定的 revision，且至少包含 `config.json`、tokenizer 文件和全部 safetensors 分片；不要把“文件看起来齐全”当成 revision 已确认。

### 9.3 校验模型目录

```bash
test -f "$PGCG_MODEL_DIR/config.json"
find "$PGCG_MODEL_DIR" -maxdepth 1 -type f -name '*.safetensors' -print | sort
du -sh "$PGCG_MODEL_DIR"
export PGCG_MODEL="$PGCG_MODEL_DIR"

{
  printf 'model=Qwen/Qwen2.5-7B-Instruct\n'
  printf 'revision=a09a35458c702b33eeacc393d103063234e8bc28\n'
  (
    cd "$PGCG_MODEL_DIR"
    find . -maxdepth 1 -type f -print0 | \
      LC_ALL=C sort -z | xargs -0 sha256sum
  )
} | tee "$PGCG_LOG_DIR/05-model-manifest.txt"
```

该命令会顺序读取一次权重文件，以便留下可复核的模型字节哈希。后续所有组都使用这个绝对路径，不再使用会变化的 Hub 名称；若已有本地模型的 revision 来源无法确认，停止并重新按 9.1 下载固定快照，不要只在清单中手写目标 revision。

## 10. 定义唯一的一组服务与压测函数

整段复制到同一个 Bash 终端。关闭终端后，需要重新执行第 2 节的环境变量和本节函数定义；如果第 13 节已经生成计划，还要重新执行其中导出 `PGCG_EQUAL_CONFIG` 与 `PGCG_LITE_CONFIG` 的两个代码块，不能凭记忆重写配置。

```bash
cd "$PGCG_REPO"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V2_MODEL_RUNNER=1
export PGCG_MODEL="$PGCG_MODEL_DIR"
export PGCG_PID_FILE="$PGCG_ROOT/vllm-server.pid"

{
  date --iso-8601=seconds
  printf 'branch=%s\n' "$(git branch --show-current)"
  printf 'commit=%s\n' "$(git rev-parse HEAD)"
  printf 'model_path=%s\n' "$PGCG_MODEL"
  printf 'model_revision=%s\n' \
    'a09a35458c702b33eeacc393d103063234e8bc28'
  printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
  printf 'vllm_use_v2_model_runner=%s\n' "$VLLM_USE_V2_MODEL_RUNNER"
  printf 'service=max_model_len=4096,max_num_seqs=%s,' \
    "$PGCG_MAX_NUM_SEQS"
  printf 'max_num_batched_tokens=4096,gpu_memory_utilization=%s\n' \
    "$PGCG_GPU_MEMORY_UTILIZATION"
  printf 'workload=input=512,output=128,concurrency=16,seed=2026\n'
  printf 'correctness=batch_invariant=1,prompts=20,output=64\n'
  printf 'performance=batch_invariant=0,prompts=500,percentiles=95,99\n'
  "$PGCG_REPO/.venv/bin/python" -c \
    'import torch, vllm; print(f"torch={torch.__version__}"); print(f"vllm={vllm.__version__}")'
  printf '[packages]\n'
  uv pip freeze
} | tee "$PGCG_LOG_DIR/06-experiment-manifest.txt"

pgcg_start_server() {
  local run_name="$1"
  local compilation_config="${2:-}"
  local enable_metrics="${3:-0}"
  local batch_invariant="${4:-0}"
  local log_file="$PGCG_LOG_DIR/${run_name}-server.log"
  local mode_file="$PGCG_LOG_DIR/${run_name}-mode.txt"
  local start_ms
  start_ms="$(date +%s%3N)"
  local cmd=(
    "$PGCG_REPO/.venv/bin/vllm" serve "$PGCG_MODEL"
    --served-model-name pg-cg-qwen
    --dtype bfloat16
    --seed 2026
    --max-model-len 4096
    --max-num-seqs "$PGCG_MAX_NUM_SEQS"
    --max-num-batched-tokens 4096
    --gpu-memory-utilization "$PGCG_GPU_MEMORY_UTILIZATION"
    --generation-config vllm
    --port "$PGCG_PORT"
  )

  if [[ "$batch_invariant" != "0" && "$batch_invariant" != "1" ]]; then
    echo "batch_invariant 只能是 0 或 1"
    return 2
  fi

  if [[ -n "$compilation_config" ]]; then
    cmd+=(--compilation-config "$compilation_config")
  fi
  if [[ "$enable_metrics" == "1" ]]; then
    cmd+=(--cudagraph-metrics)
  fi

  local env_cmd=(env -u VLLM_BATCH_INVARIANT)
  if [[ "$batch_invariant" == "1" ]]; then
    env_cmd=(env VLLM_BATCH_INVARIANT=1)
  fi

  {
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'run_name=%s\n' "$run_name"
    printf 'batch_invariant=%s\n' "$batch_invariant"
    printf 'cudagraph_metrics=%s\n' "$enable_metrics"
    printf 'compilation_config=%s\n' "${compilation_config:-DEFAULT}"
    nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used \
      --format=csv,noheader
  } | tee "$mode_file"

  printf '启动命令：'
  printf ' %q' "${env_cmd[@]}" "${cmd[@]}"
  printf '\n'

  setsid "${env_cmd[@]}" "${cmd[@]}" >"$log_file" 2>&1 < /dev/null &
  local server_pid=$!
  printf '%s\n' "$server_pid" >"$PGCG_PID_FILE"

  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$PGCG_PORT/health" >/dev/null; then
      local ready_ms
      ready_ms="$(date +%s%3N)"
      awk -v start="$start_ms" -v ready="$ready_ms" \
        'BEGIN { printf "%.3f\n", (ready - start) / 1000 }' | \
        tee "$PGCG_LOG_DIR/${run_name}-ready-seconds.txt"
      echo "服务已就绪：$run_name，PID=$server_pid"
      grep -q 'Using V2 Model Runner' "$log_file"
      grep -q 'Graph capturing finished' "$log_file"
      return 0
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "服务提前退出，最后 100 行日志如下："
      tail -100 "$log_file"
      return 1
    fi
    sleep 2
  done

  echo "服务 360 秒内未就绪，最后 100 行日志如下："
  tail -100 "$log_file"
  return 1
}

pgcg_stop_server() {
  local server_pid
  server_pid="$(cat "$PGCG_PID_FILE")"
  kill -TERM -- "-$server_pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      : >"$PGCG_PID_FILE"
      nvidia-smi --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader || true
      return 0
    fi
    sleep 1
  done
  echo "服务未在 30 秒内退出；不要启动下一组，请先检查 PID $server_pid"
  return 1
}

pgcg_wait_cool() {
  local target_temp="${1:-55}"
  for _ in $(seq 1 120); do
    local temp util
    temp="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
    if [[ "$util" -le 2 && "$temp" -le "$target_temp" ]]; then
      echo "GPU 已回到空闲状态：${temp}°C，利用率 ${util}%"
      return 0
    fi
    sleep 5
  done
  echo "10 分钟内未达到 ${target_temp}°C；记录当前温度后继续，但不要跳过空闲检查"
  nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader
}

pgcg_correctness_bench() {
  local label="$1"
  "$PGCG_REPO/.venv/bin/vllm" bench serve \
    --backend vllm \
    --model pg-cg-qwen \
    --tokenizer "$PGCG_MODEL" \
    --host 127.0.0.1 \
    --port "$PGCG_PORT" \
    --endpoint /v1/completions \
    --dataset-name random \
    --input-len 512 \
    --output-len 64 \
    --num-prompts 20 \
    --request-rate inf \
    --max-concurrency 4 \
    --seed 2026 \
    --temperature 0 \
    --ignore-eos \
    --save-result \
    --save-detailed \
    --result-dir "$PGCG_LOG_DIR" \
    --result-filename "correctness-${label}.json"
}

pgcg_perf_bench() {
  local label="$1"
  local max_concurrency="${2:-16}"
  local num_prompts="${3:-500}"
  "$PGCG_REPO/.venv/bin/vllm" bench serve \
    --backend vllm \
    --model pg-cg-qwen \
    --tokenizer "$PGCG_MODEL" \
    --host 127.0.0.1 \
    --port "$PGCG_PORT" \
    --endpoint /v1/completions \
    --dataset-name random \
    --input-len 512 \
    --output-len 128 \
    --num-prompts "$num_prompts" \
    --request-rate inf \
    --max-concurrency "$max_concurrency" \
    --metric-percentiles 95,99 \
    --seed 2026 \
    --temperature 0 \
    --ignore-eos \
    --save-result \
    --save-detailed \
    --result-dir "$PGCG_LOG_DIR" \
    --result-filename "${label}.json"

  "$PGCG_REPO/.venv/bin/python" - \
    "$PGCG_LOG_DIR/${label}.json" "$num_prompts" <<'PY' | \
    tee "$PGCG_LOG_DIR/${label}-validation.txt"
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
assert result["completed"] == int(sys.argv[2]), result
assert result["failed"] == 0, result
assert not any(result["errors"]), result["errors"]
assert all(length == 512 for length in result["input_lens"]), result["input_lens"]
assert all(length == 128 for length in result["output_lens"]), result["output_lens"]
print("benchmark_exit=0")
print("本轮通过：", sys.argv[1])
PY
}
```

`pgcg_start_server` 的第四个参数是模式隔离开关：只有功能等价性检查传 `1`；画像、冒烟、预热和正式性能实验都传 `0`，函数会显式取消父 shell 中可能残留的 `VLLM_BATCH_INVARIANT`。每次启动的实际模式和配置都会写入对应的 `*-mode.txt`。`--generation-config vllm` 用于避免模型仓库中的生成默认值改变固定协议。

## 11. 门禁三的 GPU 部分：真实服务冒烟

```bash
pgcg_start_server smoke "" 1 0

curl -fsS "http://127.0.0.1:$PGCG_PORT/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"pg-cg-qwen","prompt":"CUDA Graph is","max_tokens":32,"temperature":0}' | \
  jq . | tee "$PGCG_LOG_DIR/07-smoke-response.json"

sleep 12
grep -E 'Using V2 Model Runner|Graph capturing finished|CUDAGraph Stats|PG_CG_PROFILE=' \
  "$PGCG_LOG_DIR/smoke-server.log"

pgcg_stop_server
pgcg_wait_cool 55
```

通过条件：

- 返回文本非空；
- 日志包含 `Using V2 Model Runner`；
- 日志包含 `Graph capturing finished`；
- 日志包含非空 `CUDAGraph Stats` 和 `PG_CG_PROFILE=`；
- 没有 OOM、PTX、driver、illegal memory access 错误。

单条请求可能在 10 秒日志周期边界上没有留下 profile 行。如果只有 `PG_CG_PROFILE=` 缺失，额外发送 20 条请求并等待 12 秒；不要伪造画像数据。

## 12. 运行默认配置画像

```bash
pgcg_start_server profile "" 1 0

"$PGCG_REPO/.venv/bin/vllm" bench serve \
  --backend vllm \
  --model pg-cg-qwen \
  --tokenizer "$PGCG_MODEL" \
  --host 127.0.0.1 \
  --port "$PGCG_PORT" \
  --endpoint /v1/completions \
  --dataset-name random \
  --input-len 512 \
  --output-len 128 \
  --num-prompts 300 \
  --request-rate inf \
  --max-concurrency 16 \
  --seed 2026 \
  --temperature 0 \
  --ignore-eos

sleep 12
grep 'PG_CG_PROFILE=' "$PGCG_LOG_DIR/profile-server.log" | \
  tee "$PGCG_LOG_DIR/profile-lines.txt"

pgcg_stop_server
pgcg_wait_cool 55
```

检查日志中没有严重错误：

```bash
if rg -i 'out of memory|unsupported ptx|driver version is insufficient|illegal memory|traceback' \
  "$PGCG_LOG_DIR/profile-server.log"; then
  echo '画像运行存在严重错误，停止实验'
  false
fi
```

## 13. 从真实日志生成 K=8 计划

```bash
cd "$PGCG_REPO"

.venv/bin/python -m vllm.benchmarks.pg_cg_lite \
  --log "$PGCG_LOG_DIR/profile-server.log" \
  --max-sizes 8 \
  --output "$PGCG_LOG_DIR/plan.json"

.venv/bin/python -m json.tool "$PGCG_LOG_DIR/plan.json"

for budget in 4 8 16; do
  .venv/bin/python -m vllm.benchmarks.pg_cg_lite \
    --log "$PGCG_LOG_DIR/profile-server.log" \
    --max-sizes "$budget" \
    --output "$PGCG_LOG_DIR/sensitivity-k${budget}.json"
done

.venv/bin/python - "$PGCG_LOG_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for budget in (4, 8, 16):
    plan = json.loads((root / f"sensitivity-k{budget}.json").read_text())
    print(
        f"K={budget}: sizes={plan['selected_capture_size_count']}, "
        f"padding={plan['selected_predicted_padding_tokens']}"
    )
PY

export PGCG_LITE_CONFIG="$(
  .venv/bin/python - "$PGCG_LOG_DIR/plan.json" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps(plan["compilation_config"], separators=(",", ":")))
PY
)"

export PGCG_EQUAL_CONFIG="$(
  .venv/bin/python - "$PGCG_LOG_DIR/plan.json" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps(plan["equal_budget_compilation_config"], separators=(",", ":")))
PY
)"

{
  printf 'PGCG_EQUAL_CONFIG=%s\n' "$PGCG_EQUAL_CONFIG"
  printf 'PGCG_LITE_CONFIG=%s\n' "$PGCG_LITE_CONFIG"
} | tee "$PGCG_LOG_DIR/candidate-configs.txt"
```

自动门禁：

```bash
.venv/bin/python - "$PGCG_LOG_DIR/plan.json" <<'PY' | \
  tee "$PGCG_LOG_DIR/plan-validation.txt"
import json
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
sizes = plan["selected_capture_sizes"]
equal_sizes = plan["equal_budget_capture_sizes"]
source_sizes = plan["source_capture_sizes"]
assert plan["selection_policy"] == "default_capture_size_subset_dp"
assert plan["equal_budget_selection_policy"] == "uniform_rank_default_subset"
assert plan["max_capture_sizes"] == 8
assert 1 <= len(sizes) <= 8
assert sizes == sorted(set(sizes))
assert len(equal_sizes) == len(sizes)
assert equal_sizes == sorted(set(equal_sizes))
assert set(sizes) <= set(source_sizes)
assert set(equal_sizes) <= set(source_sizes)
assert sizes[-1] == source_sizes[-1]
assert equal_sizes[-1] == source_sizes[-1]
assert sizes == plan["compilation_config"]["cudagraph_capture_sizes"]
assert equal_sizes == plan["equal_budget_compilation_config"][
    "cudagraph_capture_sizes"
]
assert plan["baseline_capture_size_count"] > 8, plan
assert plan["baseline_capture_size_count"] == len(source_sizes), plan
assert plan["selected_capture_size_count"] == len(sizes), plan
assert plan["equal_budget_capture_size_count"] == len(equal_sizes), plan
assert plan["baseline_predicted_padding_tokens"] >= 0
assert (
    plan["baseline_predicted_padding_tokens"]
    <= plan["selected_predicted_padding_tokens"]
    <= plan["equal_budget_predicted_padding_tokens"]
)
print("计划门禁通过：", plan["baseline_capture_size_count"], "->", len(sizes))
PY
```

`max_num_seqs=128` 时默认尺寸通常是 35 个，OOM 回退到 64 时通常约 19 个。以日志中的实际值为准；若默认数量已经不大于 8，则该环境没有可剪枝空间，停止正式 A/B/C，不要声称项目有效。

## 14. 确定性模式下的 20 条请求功能等价性 A/B/C

本节只验证“改变 capture-size 配置不改变模型功能”。三组都启用
`VLLM_BATCH_INVARIANT=1`，使用相同随机请求、贪心采样和固定 seed，再逐字比较输出。该环境变量会改变执行约束，因此本节结果不能作为真实性能数字；下一节会在显式关闭它的状态下重新预热并测量性能。

默认组：

```bash
pgcg_start_server correctness-A "" 0 1
pgcg_correctness_bench A
pgcg_stop_server
pgcg_wait_cool 55
```

同预算等秩组：

```bash
pgcg_start_server correctness-B "$PGCG_EQUAL_CONFIG" 0 1
pgcg_correctness_bench B
pgcg_stop_server
pgcg_wait_cool 55
```

PG-CG Lite 组：

```bash
pgcg_start_server correctness-C "$PGCG_LITE_CONFIG" 0 1
pgcg_correctness_bench C
pgcg_stop_server
pgcg_wait_cool 55
```

比较结果：

```bash
.venv/bin/python - "$PGCG_LOG_DIR" <<'PY' | \
  tee "$PGCG_LOG_DIR/correctness-validation.txt"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
results = {
    group: json.loads((root / f"correctness-{group}.json").read_text())
    for group in "ABC"
}
assert all(result["completed"] == 20 for result in results.values())
assert all(result["failed"] == 0 for result in results.values())
assert all(not any(result["errors"]) for result in results.values())
assert all(
    all(length == 512 for length in result["input_lens"])
    for result in results.values()
)
assert all(
    all(length == 64 for length in result["output_lens"])
    for result in results.values()
)
assert results["A"]["generated_texts"] == results["B"]["generated_texts"]
assert results["A"]["generated_texts"] == results["C"]["generated_texts"]
print("correctness_validation_exit=0")
print("正确性门禁通过：20/20 输出完全一致")
PY
```

再确认三组实际模式：

```bash
for group in A B C; do
  grep -qx 'batch_invariant=1' \
    "$PGCG_LOG_DIR/correctness-${group}-mode.txt"
done
```

如果输出不一致，停止性能结论，保留三个 JSON、服务日志和模式文件并排查；不能只比较“看起来相似”，也不能退回到只比较输出数量。

## 15. 按固定次序完成 9 次 A/B/C 性能实验

正确性阶段使用了 batch-invariant 模式，不能承担性能模式预热。先在显式关闭该模式的状态下依次预热 A/B/C；预热结果只证明请求链路有效，不进入正式结果。三组使用同一个依赖与编译缓存，之后不清理缓存。正式性能组都关闭 `--cudagraph-metrics`，避免画像日志影响组间比较。A 是默认全集，B 是同预算等秩子集，C 是 PG-CG Lite 子集。

```bash
pgcg_warm_performance_group() {
  local group="$1"
  local compilation_config="${2:-}"
  local label="warm-${group}"

  pgcg_start_server "$label" "$compilation_config" 0 0
  pgcg_perf_bench "$label" 16 20
  pgcg_stop_server
  pgcg_wait_cool 55
}

pgcg_warm_performance_group A
pgcg_warm_performance_group B "$PGCG_EQUAL_CONFIG"
pgcg_warm_performance_group C "$PGCG_LITE_CONFIG"

pgcg_perf_run() {
  local label="$1"
  local compilation_config="${2:-}"
  local max_concurrency="${3:-16}"
  local num_prompts="${4:-500}"

  pgcg_start_server "$label" "$compilation_config" 0 0
  pgcg_perf_bench "$label" "$max_concurrency" "$num_prompts"
  pgcg_stop_server
  pgcg_wait_cool 55
}

# 第一轮：A → B → C
pgcg_perf_run A1
pgcg_perf_run B1 "$PGCG_EQUAL_CONFIG"
pgcg_perf_run C1 "$PGCG_LITE_CONFIG"

# 第二轮：C → A → B
pgcg_perf_run C2 "$PGCG_LITE_CONFIG"
pgcg_perf_run A2
pgcg_perf_run B2 "$PGCG_EQUAL_CONFIG"

# 第三轮：B → C → A
pgcg_perf_run B3 "$PGCG_EQUAL_CONFIG"
pgcg_perf_run C3 "$PGCG_LITE_CONFIG"
pgcg_perf_run A3
```

完整性检查：

```bash
for label in A1 B1 C1 C2 A2 B2 B3 C3 A3; do
  test -s "$PGCG_LOG_DIR/${label}.json"
  test -s "$PGCG_LOG_DIR/${label}-server.log"
  test -s "$PGCG_LOG_DIR/${label}-mode.txt"
  test -s "$PGCG_LOG_DIR/${label}-ready-seconds.txt"
  grep -qx 'batch_invariant=0' "$PGCG_LOG_DIR/${label}-mode.txt"
  grep -qx 'cudagraph_metrics=0' "$PGCG_LOG_DIR/${label}-mode.txt"
  grep 'Graph capturing finished' "$PGCG_LOG_DIR/${label}-server.log"
done
```

不要删除某个差结果，也不要增加“最好的一次”替换它。这里的 time-to-ready 是共享依赖与编译缓存已预热后的服务启动时间，不是首次安装、首次 JIT 或冷缓存部署时间；面试和简历中必须使用同一口径。

## 16. 轻量 workload-shift 检查

主实验完成后，把并发从 16 改为 4，每组只运行 200 条请求。该检查不重复三次、不进入主结论，只用于观察画像选择在一个简单分布变化下是否出现明显退化。

```bash
pgcg_perf_run shift-A "" 4 200
pgcg_perf_run shift-B "$PGCG_EQUAL_CONFIG" 4 200
pgcg_perf_run shift-C "$PGCG_LITE_CONFIG" 4 200
```

## 17. 自动汇总结果并生成一张图

```bash
.venv/bin/python - "$PGCG_LOG_DIR" <<'PY'
import json
import re
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(sys.argv[1])
labels = ["A1", "B1", "C1", "C2", "A2", "B2", "B3", "C3", "A3"]
capture_pattern = re.compile(
    r"Graph capturing finished in ([0-9.]+) secs, took ([0-9.]+) GiB"
)

rows = {}
for label in labels:
    result = json.loads((root / f"{label}.json").read_text())
    matches = capture_pattern.findall((root / f"{label}-server.log").read_text())
    if len(matches) != 1:
        raise RuntimeError(f"{label} 应有且仅有一条 capture 日志，实际为 {matches}")
    capture_time, capture_memory = map(float, matches[0])
    rows[label] = {
        "ready_time_s": float(
            (root / f"{label}-ready-seconds.txt").read_text().strip()
        ),
        "capture_time_s": capture_time,
        "capture_memory_gib": capture_memory,
        "request_throughput": float(result["request_throughput"]),
        "median_ttft_ms": float(result["median_ttft_ms"]),
        "p95_ttft_ms": float(result["p95_ttft_ms"]),
        "p99_ttft_ms": float(result["p99_ttft_ms"]),
        "median_tpot_ms": float(result["median_tpot_ms"]),
        "p95_tpot_ms": float(result["p95_tpot_ms"]),
        "p99_tpot_ms": float(result["p99_tpot_ms"]),
        "median_itl_ms": float(result["median_itl_ms"]),
        "p95_itl_ms": float(result["p95_itl_ms"]),
        "p99_itl_ms": float(result["p99_itl_ms"]),
        "completed": int(result["completed"]),
        "failed": int(result["failed"]),
        "valid": (
            int(result["completed"]) == 500
            and int(result["failed"]) == 0
            and not any(result["errors"])
            and all(length == 512 for length in result["input_lens"])
            and all(length == 128 for length in result["output_lens"])
        ),
    }

plan = json.loads((root / "plan.json").read_text())
correctness = {
    group: json.loads((root / f"correctness-{group}.json").read_text())
    for group in "ABC"
}
outputs_match = all(
    correctness["A"]["generated_texts"] == correctness[group]["generated_texts"]
    for group in "BC"
)
correctness_runs_valid = all(
    int(result["completed"]) == 20
    and int(result["failed"]) == 0
    and not any(result["errors"])
    and all(length == 512 for length in result["input_lens"])
    and all(length == 64 for length in result["output_lens"])
    for result in correctness.values()
)
shift = {}
for group in "ABC":
    result = json.loads((root / f"shift-{group}.json").read_text())
    shift[group] = {
        key: result[key]
        for key in (
            "request_throughput",
            "median_ttft_ms",
            "p95_ttft_ms",
            "p99_ttft_ms",
            "median_tpot_ms",
            "p95_tpot_ms",
            "p99_tpot_ms",
            "median_itl_ms",
            "p95_itl_ms",
            "p99_itl_ms",
        )
    }
    shift[group]["valid"] = (
        int(result["completed"]) == 200
        and int(result["failed"]) == 0
        and not any(result["errors"])
        and all(length == 512 for length in result["input_lens"])
        and all(length == 128 for length in result["output_lens"])
    )

def median(group, metric):
    return statistics.median(rows[f"{group}{index}"][metric] for index in (1, 2, 3))

def change(a, b):
    return (b - a) / a * 100.0

metrics = {
    metric: tuple(median(group, metric) for group in "ABC")
    for metric in (
        "ready_time_s",
        "capture_time_s",
        "capture_memory_gib",
        "request_throughput",
        "median_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "median_tpot_ms",
        "p95_tpot_ms",
        "p99_tpot_ms",
        "median_itl_ms",
        "p95_itl_ms",
        "p99_itl_ms",
    )
}

summary = {
    "raw_runs": rows,
    "medians": {
        name: {
            "A": values[0],
            "B": values[1],
            "C": values[2],
            "B_vs_A_percent": change(values[0], values[1]),
            "C_vs_A_percent": change(values[0], values[2]),
            "C_vs_B_percent": change(values[1], values[2]),
        }
        for name, values in metrics.items()
    },
    "capture_size_count": {
        "A": plan["baseline_capture_size_count"],
        "B": plan["equal_budget_capture_size_count"],
        "C": plan["selected_capture_size_count"],
    },
    "predicted_padding_tokens": {
        "A": plan["baseline_predicted_padding_tokens"],
        "B": plan["equal_budget_predicted_padding_tokens"],
        "C": plan["selected_predicted_padding_tokens"],
    },
    "actual_cudagraph_count": None,
    "actual_cudagraph_count_note": (
        "未从稳定指标观测；capture-size 数量不等于实际 graph 数量"
    ),
    "workload_shift_concurrency_4": shift,
    "functional_equivalence": {
        "batch_invariant": True,
        "completed_per_group": 20,
        "outputs_match_20_of_20": outputs_match,
        "all_groups_valid": correctness_runs_valid,
        "passed": correctness_runs_valid and outputs_match,
    },
    "performance_run_validity": {
        "batch_invariant": False,
        "all_nine_runs_valid": all(row["valid"] for row in rows.values()),
        "all_shift_runs_valid": all(row["valid"] for row in shift.values()),
        "expected_input_tokens_per_request": 512,
        "expected_output_tokens_per_request": 128,
        "cross_run_text_equality_required": False,
    },
    "throughput_non_regression": (
        change(metrics["request_throughput"][0], metrics["request_throughput"][2])
        >= -5.0
    ),
    "tpot_non_regression": (
        change(metrics["median_tpot_ms"][0], metrics["median_tpot_ms"][2])
        <= 5.0
    ),
}
(root / "summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)

table = [
    "| 指标 | 默认 A | 等秩 B | PG-CG C | C 相对 A | C 相对 B | 判定 |",
    "|---|---:|---:|---:|---:|---:|---|",
    f"| capture-size 数量 | {plan['baseline_capture_size_count']} | "
    f"{plan['equal_budget_capture_size_count']} | "
    f"{plan['selected_capture_size_count']} | - | - | B 与 C 同预算 |",
    "| 实际 CUDA Graph 数量 | 未观测 | 未观测 | 未观测 | - | - | "
    "不以 size 数量代替 |",
    f"| 画像预测 padding / token | "
    f"{plan['baseline_predicted_padding_tokens']} | "
    f"{plan['equal_budget_predicted_padding_tokens']} | "
    f"{plan['selected_predicted_padding_tokens']} | - | "
    f"{plan['selected_predicted_padding_tokens'] - plan['equal_budget_predicted_padding_tokens']:+d} | 越低越好 |",
]
display = [
    ("Time-to-ready / s", "ready_time_s", "越低越好"),
    ("Graph capture 时间 / s", "capture_time_s", "越低越好"),
    ("Graph capture 显存 / GiB", "capture_memory_gib", "越低越好"),
    ("Request throughput / req/s", "request_throughput", "下降不超过 5%"),
    ("Median TTFT / ms", "median_ttft_ms", "越低越好"),
    ("P95 TTFT / ms", "p95_ttft_ms", "越低越好"),
    ("P99 TTFT / ms", "p99_ttft_ms", "诊断项"),
    ("Median TPOT / ms", "median_tpot_ms", "上升不超过 5%"),
    ("P95 TPOT / ms", "p95_tpot_ms", "越低越好"),
    ("P99 TPOT / ms", "p99_tpot_ms", "诊断项"),
    ("Median ITL / ms", "median_itl_ms", "越低越好"),
    ("P95 ITL / ms", "p95_itl_ms", "越低越好"),
    ("P99 ITL / ms", "p99_itl_ms", "诊断项"),
]
for title, key, rule in display:
    a, b, c = metrics[key]
    table.append(
        f"| {title} | {a:.4f} | {b:.4f} | {c:.4f} | "
        f"{change(a, c):+.2f}% | {change(b, c):+.2f}% | {rule} |"
    )
table.append(
    f"| 确定性模式 20 条输出一致性 | - | - | - | - | - | "
    f"{'通过' if correctness_runs_valid and outputs_match else '失败'} |"
)
table.append(
    f"| 真实性能模式九轮有效性 | - | - | - | - | - | "
    f"{'全部有效' if all(row['valid'] for row in rows.values()) else '失败'} |"
)
(root / "results.md").write_text("\n".join(table) + "\n", encoding="utf-8")

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
for axis, key, title, unit in (
    (axes[0], "capture_time_s", "CUDA Graph 捕获时间", "秒"),
    (axes[1], "capture_memory_gib", "CUDA Graph 捕获显存", "GiB"),
):
    values = metrics[key]
    bars = axis.bar(
        ["默认 A", "等秩 B", "PG-CG C"],
        values,
        color=["#6b7280", "#d97706", "#2563eb"],
    )
    axis.set_title(title)
    axis.set_ylabel(unit)
    axis.bar_label(bars, fmt="%.2f")
fig.tight_layout()
fig.savefig(root / "capture-comparison.png", dpi=180)

print("\n".join(table))
print("\n汇总：", root / "summary.json")
print("表格：", root / "results.md")
print("图：", root / "capture-comparison.png")
PY
```

查看最终文件：

```bash
cat "$PGCG_LOG_DIR/results.md"
.venv/bin/python -m json.tool "$PGCG_LOG_DIR/summary.json"
ls -lh "$PGCG_LOG_DIR/capture-comparison.png"
```

## 18. 结果判定规则

主结论按以下优先级写：

1. Linux 测试与小规模穷举对拍证明配置满足默认集合子集、预算、最大覆盖和确定性最优规则；
2. batch-invariant 正确性模式下 A/B/C 均无错误、token 数符合协议且 20/20 输出逐字一致；
3. batch-invariant 关闭的九轮性能运行与三轮 shift 均完成、无请求错误且 token 数符合协议；性能轮之间不要求生成文本逐字一致；
4. B 与 C 的 capture-size 数量相同、都不超过 8，且明显小于默认数量；
5. 画像预测 padding 满足 `A <= C <= B`；
6. A 对 C 的 capture 时间、capture 显存和预热缓存下 time-to-ready 是否改善；
7. B 对 C 在同预算下是否出现额外收益；
8. C 相对 A 的请求吞吐下降不超过 5%、median TPOT 上升不超过 5%；
9. P95/P99 TTFT、TPOT 和 ITL 全量报告，用于识别尾部退化；未预注册额外通过阈值，不用尾部分位数事后改写主判定。

以下结果都可以作为项目结论：

- C 优于 A 且优于同预算 B：画像指导的子集在该固定 workload 上显示了额外价值；
- B 与 C 接近：减少 capture-size 数量有效，但当前画像目标没有显示超越简单等秩剪枝的额外收益；
- C 的预测 padding 低于 B、真实性能却不优于 B：说明 padding 代理不足以单独预测 GPU 性能；
- capture 开销下降但稳态退化超过 5%：展示初始化成本与运行时 padding 的明确权衡；
- capture 开销没有下降：保留负结果，说明该模型/版本的 capture-size 数量并非主要启动瓶颈。

不要使用“通用最优”“生产提升已证明”或“统计显著”等表述。只有 1 张卡、1 个模型、1 个主 workload、每组 3 次；并发 4 的 shift 只运行 1 次，只能作为边界提示。任何正向数字都必须排在配置正确性、功能等价性和性能运行有效性之后。

## 19. 唯一允许的 OOM 回退

若默认配置在启动 capture 阶段 OOM，统一把服务函数中的两项改为：

```bash
export PGCG_MAX_NUM_SEQS=64
export PGCG_GPU_MEMORY_UTILIZATION=0.80
export PGCG_PREVIOUS_LOG_DIR="$PGCG_LOG_DIR"
export PGCG_LOG_DIR="$PGCG_ROOT/logs-oom64"
mkdir -p "$PGCG_LOG_DIR"
cp "$PGCG_PREVIOUS_LOG_DIR"/{00-host-before.txt,01-repository.txt,02-python-cuda.txt,03-torch-compile.txt,04-linux-verification.txt,05-model-manifest.txt} \
  "$PGCG_LOG_DIR"/
```

保留原日志作为失败证据，不在其中续跑；上面的复制只复用提交、主机、环境门禁和模型哈希这六项未变化的前置证据。随后重新执行第 10 节，使新日志目录中的 `06-experiment-manifest.txt` 与服务函数同时记录回退值，再从第 11 节重新执行画像、计划、正确性、性能模式预热、全部 9 次 A/B/C 和 shift 检查。不能让三组使用不同的 `max_num_seqs`，也不能沿用旧 `plan.json`。回退后默认 capture sizes 通常约 19 个，仍可与 8 个形成清晰对比。

如果 64 仍 OOM，不继续降低到使默认尺寸数接近 8；改用更小模型并把“模型变化”写入实验条件，或先解决环境问题。

## 20. 常见故障定位

| 现象 | 原因与动作 |
|---|---|
| `torch.version.cuda` 是 13.x | 安装选错轮子；将当前 `.venv` 重命名留档，按第 7 节强制 `cu129` 重建 |
| 普通 matmul 成功、`torch.compile` 失败并报告 PTX | R535 的 JIT 限制；升级到 `575.57.08` 或更新驱动 |
| 服务日志没有 `Using V2 Model Runner` | 环境变量未生效；确认 `export VLLM_USE_V2_MODEL_RUNNER=1` 后重启 |
| 有表格但无 `PG_CG_PROFILE=` | 确认第 2 个补丁已应用，并等待完整 10 秒日志周期 |
| profile 全是 `NONE` | CUDA Graph 未真正覆盖该 workload；检查模式与严重错误，不伪造 FULL 数据 |
| plan 报不同 capture config | 日志混入了不同服务运行；只给脚本传单次 profile server 日志 |
| Lite 启动时报配置不一致 | 确认计划来自同一 vLLM commit、同一 max tokens/seqs 配置 |
| 正确性不一致 | 停止性能结论，保留详细 JSON 与服务日志 |
| 某轮 `completed` 不等于该命令的请求数或 `failed != 0` | 该轮无效；先定位错误，再按预注册 A/B/C 顺序从该轮开始重做，并注明 |
| 服务停止后仍有本人的 vLLM 进程 | 不启动下一组；根据 PID/进程组优雅终止并确认显存释放 |

## 21. 校验证据、制作归档并带回本地

不要打包整个工作目录、模型、`.venv` 或编译缓存。正式证据只来自
`$PGCG_LOG_DIR`；若缓存被误写入该目录，先停止并查明原因，不要把几十 MiB 的二进制缓存伪装成实验材料。

### 21.1 在远端 Linux 服务器整理证据

```bash
cd "$PGCG_REPO"
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = \
  "$(sed -n 's/^commit=//p' "$PGCG_LOG_DIR/01-repository.txt")"

for required in \
  00-host-before.txt \
  01-repository.txt \
  02-python-cuda.txt \
  03-torch-compile.txt \
  04-linux-verification.txt \
  05-model-manifest.txt \
  06-experiment-manifest.txt \
  07-smoke-response.json \
  profile-server.log \
  profile-lines.txt \
  plan.json \
  plan-validation.txt \
  sensitivity-k4.json \
  sensitivity-k8.json \
  sensitivity-k16.json \
  candidate-configs.txt \
  correctness-validation.txt \
  summary.json \
  results.md \
  capture-comparison.png; do
  test -s "$PGCG_LOG_DIR/$required"
done

for label in \
  smoke profile \
  correctness-A correctness-B correctness-C \
  warm-A warm-B warm-C \
  A1 B1 C1 C2 A2 B2 B3 C3 A3 \
  shift-A shift-B shift-C; do
  test -s "$PGCG_LOG_DIR/${label}-server.log"
  test -s "$PGCG_LOG_DIR/${label}-mode.txt"
  test -s "$PGCG_LOG_DIR/${label}-ready-seconds.txt"
done

for label in \
  correctness-A correctness-B correctness-C \
  warm-A warm-B warm-C \
  A1 B1 C1 C2 A2 B2 B3 C3 A3 \
  shift-A shift-B shift-C; do
  test -s "$PGCG_LOG_DIR/${label}.json"
done

for label in \
  warm-A warm-B warm-C \
  A1 B1 C1 C2 A2 B2 B3 C3 A3 \
  shift-A shift-B shift-C; do
  test -s "$PGCG_LOG_DIR/${label}-validation.txt"
done

unexpected="$(
  find "$PGCG_LOG_DIR" -type f \( \
    -path '*/.cache/*' -o \
    -path '*/__pycache__/*' -o \
    -path '*/vllm-cache/*' -o \
    -path '*/flashinfer-workspace/*' -o \
    -name '*.pyc' -o -name '*.so' -o -name '*.cubin' -o -name '*.ptx' \
  \) -print -quit
)"
if [[ -n "$unexpected" ]]; then
  echo "日志目录混入缓存或二进制产物：$unexpected"
  false
fi

(
  cd "$PGCG_LOG_DIR"
  find . -type f ! -name SHA256SUMS.txt -print0 | \
    LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS.txt
  sha256sum --check SHA256SUMS.txt
)

export PGCG_EVIDENCE_NAME="pg-cg-lite-evidence-$(date +%Y%m%d-%H%M%S).tar.gz"
export PGCG_LOG_BASENAME="$(basename "$PGCG_LOG_DIR")"
tar -C "$PGCG_ROOT" -czf "$PGCG_ROOT/$PGCG_EVIDENCE_NAME" \
  "$PGCG_LOG_BASENAME"
sha256sum "$PGCG_ROOT/$PGCG_EVIDENCE_NAME" | \
  tee "$PGCG_ROOT/$PGCG_EVIDENCE_NAME.sha256"
du -sh "$PGCG_LOG_DIR" "$PGCG_ROOT/$PGCG_EVIDENCE_NAME"
```

上面的完整性循环会拒绝缺失核心文件；`SHA256SUMS.txt` 固定归档内每个证据文件，归档外的 `.sha256` 固定传输对象。归档应包含原始 JSON、服务日志、实际模式/配置、验证输出、自动摘要和图，不应包含模型权重、Python 环境或编译缓存。

### 21.2 在本地 Windows 校验归档

先在服务器执行 `ls -1 "$PGCG_ROOT"/pg-cg-lite-evidence-*`，记下刚生成的精确文件名。然后在本地仓库根目录的 PowerShell 中执行；将示例时间戳和服务器绝对路径替换为实际值：

```powershell
$PGCG_EVIDENCE_NAME = "pg-cg-lite-evidence-YYYYMMDD-HHMMSS.tar.gz"
$PGCG_LOCAL_RESULTS_DIR = Join-Path (Get-Location) "pg-cg-lite-server-results"
New-Item -ItemType Directory -Force -Path "$PGCG_LOCAL_RESULTS_DIR" | Out-Null

scp "你的用户名@服务器地址:服务器绝对路径/pg-cg-lite-work/$PGCG_EVIDENCE_NAME" `
  "$PGCG_LOCAL_RESULTS_DIR"
scp "你的用户名@服务器地址:服务器绝对路径/pg-cg-lite-work/$PGCG_EVIDENCE_NAME.sha256" `
  "$PGCG_LOCAL_RESULTS_DIR"

$PGCG_ARCHIVE_PATH = Join-Path "$PGCG_LOCAL_RESULTS_DIR" "$PGCG_EVIDENCE_NAME"
$PGCG_SHA_PATH = "${PGCG_ARCHIVE_PATH}.sha256"
$PGCG_ACTUAL_SHA = (Get-FileHash -Algorithm SHA256 "$PGCG_ARCHIVE_PATH").Hash.ToLowerInvariant()
$PGCG_EXPECTED_SHA = ((Get-Content "$PGCG_SHA_PATH" -Raw).Trim() -split '\s+')[0].ToLowerInvariant()
if ($PGCG_ACTUAL_SHA -ne $PGCG_EXPECTED_SHA) {
  throw "证据归档 SHA256 不一致"
}
Write-Host "证据归档 SHA256 校验通过：$PGCG_ACTUAL_SHA"
tar -tzf "$PGCG_ARCHIVE_PATH"
```

只有出现“证据归档 SHA256 校验通过”才能继续。需要展开审阅时，再执行：

```powershell
$PGCG_EXTRACT_DIR = Join-Path "$PGCG_LOCAL_RESULTS_DIR" "extracted"
New-Item -ItemType Directory -Force -Path "$PGCG_EXTRACT_DIR" | Out-Null
tar -xzf (Join-Path "$PGCG_LOCAL_RESULTS_DIR" "$PGCG_EVIDENCE_NAME") `
  -C "$PGCG_EXTRACT_DIR"
```

这些材料构成“代码提交 → 环境与模型 → A/B/C 实际配置 → 配置正确性/功能等价性/性能运行有效性三道门禁 → 原始指标 → 自动结论”的可追溯证据链。正式结果发布时，只提交体积合理的摘要、配置和哈希清单；大日志归档放在 release artifact 或外部存储，并在仓库中保留索引。

## 22. 面试时用 60 秒讲清

1. vLLM 默认按规则捕获一组 CUDA Graph sizes，但不知道业务的真实 scheduled-token 分布。
2. 我补齐了 Model Runner V2 的指标传播，并让现有低频日志输出机器可读直方图。
3. 离线规划器只在默认 capture-size 集合中做精确子集搜索；当前 DP 为 `O(Km² log n)`，保留原最大覆盖上界并最小化画像上的预测 padding。
4. 在线热路径没有增加搜索，也没有修改 scheduler、kernel 或配置协议。
5. 我把正确性与性能分开验证：确定性模式逐字比较 A/B/C 的 20 条输出，真实性能模式关闭该约束，对三组各做 3 次交叉运行并检查启动、TTFT、TPOT、ITL、吞吐、错误和 token 数。
6. A 对 C 说明整体剪枝权衡，B 对 C 才说明画像的额外价值；结论只对该画像 workload 负责，流量分布变化后需要重新画像。
