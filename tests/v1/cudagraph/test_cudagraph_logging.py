# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

from vllm.compilation.cuda_graph import (
    PROFILE_PREFIX,
    CUDAGraphLogging,
    CUDAGraphStat,
)
from vllm.config import CUDAGraphMode


def make_logging() -> CUDAGraphLogging:
    logging = CUDAGraphLogging(
        CUDAGraphMode.FULL_AND_PIECEWISE,
        [1, 2, 4, 8],
    )
    logging.observe(CUDAGraphStat(3, 4, 1, "PIECEWISE"))
    logging.observe(CUDAGraphStat(1, 1, 0, "FULL"))
    logging.observe(CUDAGraphStat(3, 4, 1, "PIECEWISE"))
    return logging


def test_profile_line_is_deterministic_and_aggregated() -> None:
    logging = make_logging()

    line = logging.generate_profile_line()

    assert line.startswith(PROFILE_PREFIX)
    payload = json.loads(line.removeprefix(PROFILE_PREFIX))
    assert payload == {
        "bins": [
            {
                "count": 1,
                "num_paddings": 0,
                "num_padded_tokens": 1,
                "num_unpadded_tokens": 1,
                "runtime_mode": "FULL",
            },
            {
                "count": 2,
                "num_paddings": 1,
                "num_padded_tokens": 4,
                "num_unpadded_tokens": 3,
                "runtime_mode": "PIECEWISE",
            },
        ],
        "capture_sizes": [1, 2, 4, 8],
        "cudagraph_mode": str(CUDAGraphMode.FULL_AND_PIECEWISE),
        "schema_version": 1,
    }


def test_log_emits_table_and_profile_then_resets() -> None:
    logging = make_logging()
    messages: list[str] = []

    logging.log(messages.append)

    assert len(messages) == 2
    assert "CUDAGraph Stats" in messages[0]
    assert messages[1].startswith(PROFILE_PREFIX)
    assert logging.stats == []
