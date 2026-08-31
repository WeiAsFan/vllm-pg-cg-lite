# PG-CG Lite：A6000 完整操作手册

> 目标：你拿到服务器后，从空目录开始，按本文顺序完成环境门禁、补丁应用、单元测试、画像、计划生成、正确性检查、9 次 A/B/C 主实验、轻量 workload-shift 检查和结果汇总。除用户名、服务器地址和工作目录外，不需要临时决定实验参数。

## 0. 本手册对应的固定版本

| 项目 | 固定值 |
|---|---|
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

登录服务器后先执行：

```bash
set -euo pipefail

export PGCG_ROOT="$PWD/pg-cg-lite-work"
export PGCG_REPO="$PGCG_ROOT/vllm"
export PGCG_PATCH_DIR="$PGCG_ROOT/patches"
export PGCG_LOG_DIR="$PGCG_ROOT/logs"
export PGCG_MODEL_DIR="$PGCG_ROOT/models/Qwen2.5-7B-Instruct"
export PGCG_PORT=8000

mkdir -p "$PGCG_ROOT" "$PGCG_LOG_DIR" "$PGCG_ROOT/models" "$PGCG_ROOT/wheels"
```

建议至少预留：

- 80 GiB 可用磁盘；
- 32 GiB 系统内存，推荐 64 GiB；
- GPU 上没有其他计算进程。

## 3. 从当前电脑把补丁复制到服务器

当前仓库的 `pg-cg-lite-project/patches` 目录中包含交付补丁。先在 Windows PowerShell 中切换到仓库根目录，再执行下面命令；只替换 `你的用户名` 和 `服务器地址`：

```powershell
$PGCG_LOCAL_PATCH_DIR = (Resolve-Path ".\pg-cg-lite-project\patches").Path
scp -r "$PGCG_LOCAL_PATCH_DIR" `
  你的用户名@服务器地址:~/pg-cg-lite-patches
```

回到服务器，执行：

```bash
export PGCG_UPLOADED_PATCH_DIR="$PWD/pg-cg-lite-patches"
test -d "$PGCG_UPLOADED_PATCH_DIR"
mkdir -p "$PGCG_PATCH_DIR"
cp -a "$PGCG_UPLOADED_PATCH_DIR"/. "$PGCG_PATCH_DIR"/
find "$PGCG_PATCH_DIR" -maxdepth 1 -type f -name '*.patch' -print | sort
(cd "$PGCG_PATCH_DIR" && sha256sum --check SHA256SUMS.txt)
```

通过条件：输出恰好有 5 个按 `0001` 到 `0005` 排序的补丁，随后 5 行校验均为 `OK`。若你的上传位置不同，只调整复制来源路径；不要改补丁内容。

## 4. 记录服务器原始状态

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

## 5. 安装基础工具

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

## 6. 获取 vLLM v0.27.1 并应用补丁

```bash
cd "$PGCG_ROOT"
git clone --branch v0.27.1 --depth 1 \
  https://github.com/vllm-project/vllm.git "$PGCG_REPO"
cd "$PGCG_REPO"
git switch -c feature/pg-cg-lite
git am "$PGCG_PATCH_DIR"/*.patch

git log --oneline --decorate -5
git diff --name-only v0.27.1..HEAD
git status --short
```

通过条件：

- 最上方 5 个提交依次覆盖同预算基线、默认集合子集修复、planner、profile metrics 和 metrics propagation；
- `git diff --name-only` 只显示以下 5 个路径：

```text
tests/benchmarks/test_pg_cg_lite.py
tests/v1/cudagraph/test_cudagraph_logging.py
vllm/benchmarks/pg_cg_lite.py
vllm/compilation/cuda_graph.py
vllm/v1/worker/gpu/model_runner.py
```

- `git status --short` 没有输出。

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
uv pip install pytest==9.1.1 ruff==0.16.4 matplotlib==3.9.2

unset VLLM_USE_PRECOMPILED
unset VLLM_MAIN_CUDA_VERSION
unset VLLM_PRECOMPILED_WHEEL_LOCATION
```

不要把 `--torch-backend=cu129` 改成 `auto`。本项目不修改 C++/CUDA kernel，所以不需要本机编译 vLLM。

## 8. 环境三层门禁

### 8.1 门禁一：普通 CUDA

```bash
cd "$PGCG_REPO"
source .venv/bin/activate

.venv/bin/python - <<'PY' | tee "$PGCG_LOG_DIR/01-python-cuda.txt"
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
.venv/bin/python - <<'PY' | tee "$PGCG_LOG_DIR/02-torch-compile.txt"
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
  |& tee "$PGCG_LOG_DIR/03-linux-verification.txt"
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

重启后重新登录，从第 4 节重新记录环境，再重跑第 8 节。不要混用 apt 驱动和 NVIDIA `.run` 安装器。若你无驱动升级权限，把 `00-host-before.txt`、`01-python-cuda.txt`、`02-torch-compile.txt` 交给管理员即可；在门禁通过前不生成实验结论。

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

把完整模型目录复制到 `$PGCG_MODEL_DIR`。至少应包含 `config.json`、tokenizer 文件和全部 safetensors 分片。

### 9.3 校验模型目录

```bash
test -f "$PGCG_MODEL_DIR/config.json"
find "$PGCG_MODEL_DIR" -maxdepth 1 -type f -name '*.safetensors' -print | sort
du -sh "$PGCG_MODEL_DIR"
export PGCG_MODEL="$PGCG_MODEL_DIR"
```

后续所有组都使用这个绝对路径，不再使用会变化的 Hub 名称。

## 10. 定义唯一的一组服务与压测函数

整段复制到同一个 Bash 终端。关闭终端后，需要重新执行第 2 节的环境变量和本节函数定义。

```bash
cd "$PGCG_REPO"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V2_MODEL_RUNNER=1
export PGCG_MODEL="$PGCG_MODEL_DIR"
export PGCG_PID_FILE="$PGCG_ROOT/vllm-server.pid"

pgcg_start_server() {
  local run_name="$1"
  local compilation_config="${2:-}"
  local enable_metrics="${3:-0}"
  local log_file="$PGCG_LOG_DIR/${run_name}-server.log"
  local start_ms
  start_ms="$(date +%s%3N)"
  local cmd=(
    "$PGCG_REPO/.venv/bin/vllm" serve "$PGCG_MODEL"
    --served-model-name pg-cg-qwen
    --dtype bfloat16
    --seed 2026
    --max-model-len 4096
    --max-num-seqs 128
    --max-num-batched-tokens 4096
    --gpu-memory-utilization 0.85
    --port "$PGCG_PORT"
  )

  if [[ -n "$compilation_config" ]]; then
    cmd+=(--compilation-config "$compilation_config")
  fi
  if [[ "$enable_metrics" == "1" ]]; then
    cmd+=(--cudagraph-metrics)
  fi

  printf '启动命令：'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  setsid "${cmd[@]}" >"$log_file" 2>&1 < /dev/null &
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
    --metric-percentiles 95 \
    --seed 2026 \
    --temperature 0 \
    --ignore-eos \
    --save-result \
    --result-dir "$PGCG_LOG_DIR" \
    --result-filename "${label}.json"

  "$PGCG_REPO/.venv/bin/python" - \
    "$PGCG_LOG_DIR/${label}.json" "$num_prompts" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
assert result["completed"] == int(sys.argv[2]), result
assert result["failed"] == 0, result
print("本轮通过：", sys.argv[1])
PY
}
```

## 11. 门禁三的 GPU 部分：真实服务冒烟

```bash
pgcg_start_server smoke "" 1

curl -fsS "http://127.0.0.1:$PGCG_PORT/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"pg-cg-qwen","prompt":"CUDA Graph is","max_tokens":32,"temperature":0}' | \
  jq . | tee "$PGCG_LOG_DIR/03-smoke-response.json"

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
pgcg_start_server profile "" 1

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
.venv/bin/python - "$PGCG_LOG_DIR/plan.json" <<'PY'
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

`max_num_seqs=128` 时默认尺寸通常是 35 个。以日志中的实际值为准；若默认数量已经不大于 8，则该环境没有可剪枝空间，停止正式 A/B/C，不要声称项目有效。

## 14. 20 条请求正确性 A/B/C

默认组：

```bash
pgcg_start_server correctness-A
pgcg_correctness_bench A
pgcg_stop_server
pgcg_wait_cool 55
```

同预算等秩组：

```bash
pgcg_start_server correctness-B "$PGCG_EQUAL_CONFIG"
pgcg_correctness_bench B
pgcg_stop_server
pgcg_wait_cool 55
```

PG-CG Lite 组：

```bash
pgcg_start_server correctness-C "$PGCG_LITE_CONFIG"
pgcg_correctness_bench C
pgcg_stop_server
pgcg_wait_cool 55
```

比较结果：

```bash
.venv/bin/python - "$PGCG_LOG_DIR" <<'PY'
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
assert results["A"]["generated_texts"] == results["B"]["generated_texts"]
assert results["A"]["generated_texts"] == results["C"]["generated_texts"]
print("正确性门禁通过：20/20 输出完全一致")
PY
```

如果输出不一致，停止性能结论，保留三个 JSON 并排查；不能只比较“看起来相似”。

## 15. 按固定次序完成 9 次 A/B/C 性能实验

画像和正确性阶段已完成编译缓存预热。正式性能组都关闭 `--cudagraph-metrics`，避免画像日志影响组间比较。A 是默认全集，B 是同预算等秩子集，C 是 PG-CG Lite 子集。

```bash
pgcg_perf_run() {
  local label="$1"
  local compilation_config="${2:-}"
  local max_concurrency="${3:-16}"
  local num_prompts="${4:-500}"

  pgcg_start_server "$label" "$compilation_config"
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
  test -s "$PGCG_LOG_DIR/${label}-ready-seconds.txt"
  grep 'Graph capturing finished' "$PGCG_LOG_DIR/${label}-server.log"
done
```

不要删除某个差结果，也不要增加“最好的一次”替换它。

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
        "median_tpot_ms": float(result["median_tpot_ms"]),
        "p95_tpot_ms": float(result["p95_tpot_ms"]),
        "completed": int(result["completed"]),
        "failed": int(result["failed"]),
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
shift = {}
for group in "ABC":
    result = json.loads((root / f"shift-{group}.json").read_text())
    shift[group] = {
        key: result[key]
        for key in (
            "request_throughput",
            "median_ttft_ms",
            "p95_ttft_ms",
            "median_tpot_ms",
            "p95_tpot_ms",
        )
    }

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
        "median_tpot_ms",
        "p95_tpot_ms",
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
    "outputs_match_20_of_20": outputs_match,
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
    ("Median TPOT / ms", "median_tpot_ms", "上升不超过 5%"),
    ("P95 TPOT / ms", "p95_tpot_ms", "越低越好"),
]
for title, key, rule in display:
    a, b, c = metrics[key]
    table.append(
        f"| {title} | {a:.4f} | {b:.4f} | {c:.4f} | "
        f"{change(a, c):+.2f}% | {change(b, c):+.2f}% | {rule} |"
    )
table.append(
    f"| 20 条输出一致性 | - | - | - | - | - | "
    f"{'完全一致' if outputs_match else '失败'} |"
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

1. B 与 C 的 capture-size 数量相同、都不超过 8，且明显小于默认数量；
2. 画像预测 padding 满足 `A <= C <= B`；
3. A 对 C 的 capture 时间、capture 显存和 time-to-ready 是否改善；
4. B 对 C 在同预算下是否出现额外收益；
5. C 相对 A 的请求吞吐下降不超过 5%、median TPOT 上升不超过 5%；
6. A/B/C 的 20/20 输出完全一致。

三种结果都可以作为项目结论：

- C 优于 A 且优于同预算 B：画像指导的子集在该固定 workload 上显示了额外价值；
- B 与 C 接近：减少 capture-size 数量有效，但当前画像目标没有显示超越简单等秩剪枝的额外收益；
- C 的预测 padding 低于 B、真实性能却不优于 B：说明 padding 代理不足以单独预测 GPU 性能；
- capture 开销下降但稳态退化超过 5%：展示初始化成本与运行时 padding 的明确权衡；
- capture 开销没有下降：保留负结果，说明该模型/版本的 capture-size 数量并非主要启动瓶颈。

不要使用“通用最优”“生产提升已证明”或“统计显著”等表述。只有 1 张卡、1 个模型、1 个主 workload、每组 3 次；并发 4 的 shift 只运行 1 次，只能作为边界提示。

## 19. 唯一允许的 OOM 回退

若默认配置在启动 capture 阶段 OOM，统一把服务函数中的两项改为：

```text
--max-num-seqs 64
--gpu-memory-utilization 0.80
```

然后删除本轮未完成的结论，从第 11 节开始重新执行画像、计划、正确性、全部 9 次 A/B/C 和 shift 检查。不能让三组使用不同的 `max_num_seqs`，也不能沿用旧 `plan.json`。回退后默认 capture sizes 通常约 19 个，仍可与 8 个形成清晰对比。

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

## 21. 服务器实验完成后带回的文件

把整个日志目录复制回当前电脑：

```powershell
$PGCG_LOCAL_RESULTS_DIR = Join-Path (Get-Location) "pg-cg-lite-server-results"
New-Item -ItemType Directory -Force -Path "$PGCG_LOCAL_RESULTS_DIR" | Out-Null
scp -r 你的用户名@服务器地址:服务器上的绝对路径/pg-cg-lite-work/logs/. `
  "$PGCG_LOCAL_RESULTS_DIR"
```

至少应包含：

```text
00-host-before.txt
01-python-cuda.txt
02-torch-compile.txt
profile-server.log
profile-lines.txt
plan.json
sensitivity-k4.json sensitivity-k8.json sensitivity-k16.json
candidate-configs.txt
correctness-A.json
correctness-B.json
correctness-C.json
A1.json B1.json C1.json C2.json A2.json B2.json B3.json C3.json A3.json
A1-server.log B1-server.log C1-server.log ... B3-server.log C3-server.log A3-server.log
A1-ready-seconds.txt B1-ready-seconds.txt ... C3-ready-seconds.txt A3-ready-seconds.txt
shift-A.json shift-B.json shift-C.json
shift-A-server.log shift-B-server.log shift-C-server.log
shift-A-ready-seconds.txt shift-B-ready-seconds.txt shift-C-ready-seconds.txt
summary.json
results.md
capture-comparison.png
```

这些原始文件足以复核全部结论，也是面试时最有价值的证据链。

## 22. 面试时用 60 秒讲清

1. vLLM 默认按规则捕获一组 CUDA Graph sizes，但不知道业务的真实 scheduled-token 分布。
2. 我补齐了 Model Runner V2 的指标传播，并让现有低频日志输出机器可读直方图。
3. 离线规划器只在默认 capture-size 集合中做精确子集搜索；当前 DP 为 `O(Km² log n)`，保留原最大覆盖上界并最小化画像上的预测 padding。
4. 在线热路径没有增加搜索，也没有修改 scheduler、kernel 或配置协议。
5. 我用单卡 A6000、同一模型快照和同一 workload 比较默认全集、同预算等秩子集与画像子集，每组 3 次交叉运行，并检查启动、TTFT、TPOT、吞吐和 20 条输出一致性。
6. 结论只对该画像 workload 负责；流量分布变化后需要重新画像。
