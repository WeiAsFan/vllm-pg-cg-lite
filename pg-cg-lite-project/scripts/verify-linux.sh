#!/usr/bin/env bash

set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "错误：该验证入口只能在 Linux 服务器执行。" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
python_bin="$repo_root/.venv/bin/python"
ruff_bin="$repo_root/.venv/bin/ruff"

if [[ ! -x "$python_bin" ]]; then
  echo "错误：未找到 $python_bin；请先按 A6000 运行手册创建项目环境。" >&2
  exit 2
fi

if [[ ! -x "$ruff_bin" ]]; then
  echo "错误：未找到 $ruff_bin；请先按 A6000 运行手册安装固定版本的 Ruff。" >&2
  exit 2
fi

cd -- "$repo_root"

echo "[1/5] 校验交付补丁 SHA256"
(
  cd pg-cg-lite-project/patches
  sha256sum --check SHA256SUMS.txt
)

echo "[2/5] 运行画像日志单元测试（预期 2 项）"
"$python_bin" -m pytest \
  --confcutdir=tests/v1/cudagraph \
  tests/v1/cudagraph/test_cudagraph_logging.py -q

echo "[3/5] 运行 planner 单元测试（预期 20 项）"
"$python_bin" -m pytest \
  --confcutdir=tests/benchmarks \
  tests/benchmarks/test_pg_cg_lite.py -q

check_paths=(
  vllm/v1/worker/gpu/model_runner.py
  vllm/compilation/cuda_graph.py
  vllm/benchmarks/pg_cg_lite.py
  tests/v1/cudagraph/test_cudagraph_logging.py
  tests/benchmarks/test_pg_cg_lite.py
)

echo "[4/5] 运行 Ruff 静态检查"
"$ruff_bin" check "${check_paths[@]}"

echo "[5/5] 运行 Ruff 格式检查"
"$ruff_bin" format --check "${check_paths[@]}"

echo "Linux 固定验证入口全部通过。"
