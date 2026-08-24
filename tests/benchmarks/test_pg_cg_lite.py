# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from itertools import combinations
from pathlib import Path
from random import Random

import pytest

from vllm.benchmarks.pg_cg_lite import (
    PROFILE_PREFIX,
    build_plan,
    main,
    parse_profile_lines,
    predict_padding,
    select_capture_sizes,
)

SIZES = [1, 2, 4, 8]


def event(token_count: int, count: int = 1, mode: str = "FULL") -> dict[str, object]:
    padded = next((size for size in SIZES if size >= token_count), token_count)
    return {
        "num_unpadded_tokens": token_count,
        "num_padded_tokens": padded,
        "num_paddings": padded - token_count,
        "runtime_mode": mode,
        "count": count,
    }


def profile_line(*bins: dict[str, object], sizes: list[int] = SIZES) -> str:
    payload = {
        "schema_version": 1,
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "capture_sizes": sizes,
        "bins": bins,
    }
    return "log-prefix " + PROFILE_PREFIX + json.dumps(payload)


def test_parse_profile_lines_merges_intervals() -> None:
    profile = parse_profile_lines(
        [
            "ordinary log",
            profile_line(event(3, 2)),
            profile_line(event(3, 5, "CUDAGraphMode.PIECEWISE"), event(9, 4, "NONE")),
        ]
    )
    assert profile.histogram == {3: 7}
    assert profile.none_event_count == 4
    assert profile.source_capture_sizes == tuple(SIZES)


def test_two_sizes_choose_minimum_padding_partition() -> None:
    histogram = {1: 5, 3: 3, 8: 2}
    selected = select_capture_sizes(histogram, max_graphs=2, max_capture_size=8)
    assert selected == (3, 8)
    assert predict_padding(histogram, selected) == 10


def brute_force(
    histogram: dict[int, int], max_graphs: int, maximum: int
) -> tuple[int, ...]:
    points = sorted(set(histogram) | {maximum})
    candidates = (
        chosen
        for chosen in combinations(points, min(max_graphs, len(points)))
        if chosen[-1] == maximum
    )
    return min(
        candidates, key=lambda chosen: (predict_padding(histogram, chosen), chosen)
    )


def test_dynamic_programming_matches_brute_force() -> None:
    random = Random(2026)
    for _ in range(50):
        points = sorted(random.sample(range(1, 31), random.randint(2, 8)))
        histogram = {point: random.randint(1, 9) for point in points}
        maximum = points[-1] + random.randint(0, 3)
        max_graphs = random.randint(1, min(4, len(points)))
        assert select_capture_sizes(histogram, max_graphs, maximum) == brute_force(
            histogram, max_graphs, maximum
        )


def test_parse_rejects_empty_profile() -> None:
    with pytest.raises(ValueError, match="no PG_CG_PROFILE"):
        parse_profile_lines(["ordinary log only"])


def test_parse_rejects_mixed_capture_configs() -> None:
    with pytest.raises(ValueError, match="different capture-size"):
        parse_profile_lines(
            [profile_line(event(1)), profile_line(event(1), sizes=[1, 2, 4, 16])]
        )


def test_build_plan_is_directly_applicable() -> None:
    plan = build_plan(parse_profile_lines([profile_line(event(3, 5), event(8))]), 2)
    assert plan["selected_capture_size_count"] == 2
    assert plan["compilation_config"] == {
        "cudagraph_capture_sizes": plan["selected_capture_sizes"]
    }


def test_main_writes_plan_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_path, output_path = tmp_path / "server.log", tmp_path / "plan.json"
    log_path.write_text(profile_line(event(3, 5), event(8)), encoding="utf-8")
    assert (
        main(
            ["--log", str(log_path), "--max-graphs", "2", "--output", str(output_path)]
        )
        == 0
    )
    assert json.loads(output_path.read_text())["selected_capture_sizes"] == [3, 8]
    assert '"cudagraph_capture_sizes":[3,8]' in capsys.readouterr().out
