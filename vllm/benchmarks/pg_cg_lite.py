# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import argparse
import json
from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

PROFILE_PREFIX = "PG_CG_PROFILE="
ELIGIBLE_MODES = {"FULL", "PIECEWISE"}


@dataclass(frozen=True)
class Profile:
    histogram: dict[int, int]
    none_event_count: int
    source_capture_sizes: tuple[int, ...]


def parse_profile_lines(lines: Iterable[str]) -> Profile:
    histogram: Counter[int] = Counter()
    none_event_count = 0
    source_sizes: tuple[int, ...] | None = None
    records = 0

    for line_number, line in enumerate(lines, start=1):
        _, marker, raw = line.partition(PROFILE_PREFIX)
        if not marker:
            continue
        records += 1
        try:
            payload = json.loads(raw)
            if payload.get("schema_version") != 1:
                raise ValueError("unsupported schema")
            sizes = tuple(map(int, payload["capture_sizes"]))
            if not sizes or sizes != tuple(sorted(set(sizes))) or sizes[0] <= 0:
                raise ValueError("capture_sizes must be sorted positive integers")
            if source_sizes is not None and source_sizes != sizes:
                raise ValueError("profile contains different capture-size configs")
            source_sizes = sizes
            for item in payload["bins"]:
                token_count, count = map(
                    int, (item["num_unpadded_tokens"], item["count"])
                )
                if token_count <= 0 or count <= 0:
                    raise ValueError("profile bins must be positive")
                mode = str(item["runtime_mode"]).rsplit(".", maxsplit=1)[-1]
                if mode in ELIGIBLE_MODES:
                    histogram[token_count] += count
                elif mode == "NONE":
                    none_event_count += count
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid PG-CG record at line {line_number}") from error

    if not records:
        raise ValueError("log contains no PG_CG_PROFILE records")
    if source_sizes is None or not histogram:
        raise ValueError("profile contains no FULL or PIECEWISE events")
    if max(histogram) > source_sizes[-1]:
        raise ValueError("eligible event exceeds source capture-size maximum")
    return Profile(dict(sorted(histogram.items())), none_event_count, source_sizes)


def parse_profile_log(path: Path) -> Profile:
    with path.open(encoding="utf-8", errors="replace") as log_file:
        return parse_profile_lines(log_file)


def predict_padding(histogram: Mapping[int, int], capture_sizes: Sequence[int]) -> int:
    sizes = tuple(sorted(capture_sizes))
    if not sizes:
        raise ValueError("capture_sizes must not be empty")
    total = 0
    for token_count, count in histogram.items():
        index = bisect_left(sizes, token_count)
        if index == len(sizes):
            raise ValueError(f"token count {token_count} is not covered")
        total += count * (sizes[index] - token_count)
    return total


def select_capture_sizes(
    histogram: Mapping[int, int],
    max_sizes: int,
    source_capture_sizes: Sequence[int],
) -> tuple[int, ...]:
    if max_sizes < 1 or not histogram:
        raise ValueError("max_sizes and histogram must be non-zero")
    if any(token_count <= 0 or count <= 0 for token_count, count in histogram.items()):
        raise ValueError("histogram keys and counts must be positive")

    source_sizes = tuple(source_capture_sizes)
    if (
        not source_sizes
        or source_sizes != tuple(sorted(set(source_sizes)))
        or source_sizes[0] <= 0
    ):
        raise ValueError("source_capture_sizes must be sorted positive integers")
    if max(histogram) > source_sizes[-1]:
        raise ValueError("source capture-size maximum must cover the histogram")

    points = sorted(histogram)
    prefix_count = [0]
    prefix_tokens = [0]
    for point in points:
        prefix_count.append(prefix_count[-1] + histogram[point])
        prefix_tokens.append(prefix_tokens[-1] + point * histogram[point])

    def interval_cost(previous: int | None, endpoint: int) -> int:
        start = 0 if previous is None else bisect_right(points, previous)
        stop = bisect_right(points, endpoint)
        count = prefix_count[stop] - prefix_count[start]
        tokens = prefix_tokens[stop] - prefix_tokens[start]
        return endpoint * count - tokens

    groups = min(max_sizes, len(source_sizes))
    dp: list[list[tuple[int, tuple[int, ...]] | None]] = [
        [None] * len(source_sizes) for _ in range(groups + 1)
    ]
    for stop, endpoint in enumerate(source_sizes):
        dp[1][stop] = (interval_cost(None, endpoint), (endpoint,))

    for group in range(2, groups + 1):
        for stop in range(group - 1, len(source_sizes)):
            endpoint = source_sizes[stop]
            candidates = []
            for previous_index in range(group - 2, stop):
                previous = dp[group - 1][previous_index]
                if previous is None:
                    continue
                candidates.append(
                    (
                        previous[0]
                        + interval_cost(source_sizes[previous_index], endpoint),
                        previous[1] + (endpoint,),
                    )
                )
            if candidates:
                dp[group][stop] = min(candidates)

    final_states = [
        state for group in range(1, groups + 1) if (state := dp[group][-1]) is not None
    ]
    result = min(final_states, key=lambda state: (state[0], len(state[1]), state[1]))
    return result[1]


def build_plan(profile: Profile, max_sizes: int) -> dict[str, object]:
    selected = select_capture_sizes(
        profile.histogram, max_sizes, profile.source_capture_sizes
    )
    return {
        "selection_policy": "default_capture_size_subset_dp",
        "max_capture_sizes": max_sizes,
        "source_capture_sizes": list(profile.source_capture_sizes),
        "selected_capture_sizes": list(selected),
        "profile_event_count": sum(profile.histogram.values()),
        "none_event_count": profile.none_event_count,
        "baseline_capture_size_count": len(profile.source_capture_sizes),
        "selected_capture_size_count": len(selected),
        "baseline_predicted_padding_tokens": predict_padding(
            profile.histogram, profile.source_capture_sizes
        ),
        "selected_predicted_padding_tokens": predict_padding(
            profile.histogram, selected
        ),
        "compilation_config": {"cudagraph_capture_sizes": list(selected)},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a small profile-guided CUDA Graph capture plan."
    )
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--max-sizes", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    plan = build_plan(parse_profile_log(args.log), args.max_sizes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(
        f"capture sizes: {plan['baseline_capture_size_count']} -> "
        f"{plan['selected_capture_size_count']}"
    )
    print(
        f"predicted padding tokens: {plan['baseline_predicted_padding_tokens']} -> "
        f"{plan['selected_predicted_padding_tokens']}"
    )
    print(json.dumps(plan["compilation_config"], separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
