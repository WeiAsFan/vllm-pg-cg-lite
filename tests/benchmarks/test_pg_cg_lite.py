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
    select_uniform_rank_subset,
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


def test_two_sizes_choose_minimum_padding_default_subset() -> None:
    histogram = {1: 5, 3: 3, 8: 2}
    selected = select_capture_sizes(histogram, max_sizes=2, source_capture_sizes=SIZES)
    assert selected == (1, 8)
    assert predict_padding(histogram, selected) == 15


def brute_force(
    histogram: dict[int, int], max_sizes: int, source_sizes: list[int]
) -> tuple[int, ...]:
    candidates = (
        chosen
        for size_count in range(1, min(max_sizes, len(source_sizes)) + 1)
        for chosen in combinations(source_sizes, size_count)
        if chosen[-1] == source_sizes[-1]
    )
    return min(
        candidates,
        key=lambda chosen: (predict_padding(histogram, chosen), len(chosen), chosen),
    )


def test_dynamic_programming_matches_brute_force() -> None:
    random = Random(2026)
    for _ in range(50):
        source_sizes = sorted(random.sample(range(1, 31), random.randint(2, 8)))
        points = sorted(
            random.sample(
                range(1, source_sizes[-1] + 1),
                random.randint(1, min(8, source_sizes[-1])),
            )
        )
        histogram = {point: random.randint(1, 9) for point in points}
        max_sizes = random.randint(1, min(4, len(source_sizes)))
        assert select_capture_sizes(histogram, max_sizes, source_sizes) == brute_force(
            histogram, max_sizes, source_sizes
        )


def test_result_is_a_default_subset_and_preserves_maximum() -> None:
    selected = select_capture_sizes({3: 7, 7: 2}, 3, SIZES)
    assert set(selected) <= set(SIZES)
    assert selected[-1] == SIZES[-1]
    assert len(selected) <= 3


def test_equal_padding_prefers_fewer_sizes() -> None:
    assert select_capture_sizes({8: 4}, 4, SIZES) == (8,)


def test_budget_covering_all_used_default_sizes_keeps_them() -> None:
    assert select_capture_sizes({size: 1 for size in SIZES}, 8, SIZES) == tuple(SIZES)


@pytest.mark.parametrize(
    ("size_count", "expected"),
    [(1, (8,)), (2, (1, 8)), (3, (1, 4, 8)), (8, (1, 2, 4, 8))],
)
def test_uniform_rank_subset_is_deterministic_and_preserves_maximum(
    size_count: int, expected: tuple[int, ...]
) -> None:
    assert select_uniform_rank_subset(SIZES, size_count) == expected


def test_uniform_rank_subset_rejects_zero_budget() -> None:
    with pytest.raises(ValueError, match="size_count"):
        select_uniform_rank_subset(SIZES, 0)


@pytest.mark.parametrize(
    "source_sizes",
    [[], [1, 4, 2, 8], [1, 2, 2, 8], [0, 1, 2, 8]],
)
def test_select_rejects_invalid_source_sizes(source_sizes: list[int]) -> None:
    with pytest.raises(ValueError, match="source_capture_sizes"):
        select_capture_sizes({1: 1}, 2, source_sizes)


def test_select_rejects_uncovered_histogram() -> None:
    with pytest.raises(ValueError, match="must cover"):
        select_capture_sizes({9: 1}, 2, SIZES)


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
    assert plan["selection_policy"] == "default_capture_size_subset_dp"
    assert plan["max_capture_sizes"] == 2
    assert plan["source_capture_sizes"] == SIZES
    assert plan["equal_budget_selection_policy"] == "uniform_rank_default_subset"
    assert plan["equal_budget_capture_sizes"] == [1, 8]
    assert (
        plan["equal_budget_capture_size_count"]
        == plan["selected_capture_size_count"]
    )
    assert plan["equal_budget_compilation_config"] == {
        "cudagraph_capture_sizes": plan["equal_budget_capture_sizes"]
    }
    assert plan["compilation_config"] == {
        "cudagraph_capture_sizes": plan["selected_capture_sizes"]
    }


def test_main_writes_plan_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_path, output_path = tmp_path / "server.log", tmp_path / "plan.json"
    log_path.write_text(profile_line(event(3, 5), event(8)), encoding="utf-8")
    assert (
        main(["--log", str(log_path), "--max-sizes", "2", "--output", str(output_path)])
        == 0
    )
    assert json.loads(output_path.read_text())["selected_capture_sizes"] == [4, 8]
    assert '"cudagraph_capture_sizes":[4,8]' in capsys.readouterr().out
