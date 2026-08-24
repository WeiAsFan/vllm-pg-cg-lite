# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import argparse
import json
from bisect import bisect_left
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
    histogram: Mapping[int, int], max_graphs: int, max_capture_size: int
) -> tuple[int, ...]:
    if max_graphs < 1 or not histogram:
        raise ValueError("max_graphs and histogram must be non-zero")
    if max(histogram) > max_capture_size:
        raise ValueError("max_capture_size must cover the histogram")

    weighted = Counter(histogram)
    weighted.setdefault(max_capture_size, 0)
    points = sorted(weighted)
    groups = min(max_graphs, len(points))
    prefix_count = [0]
    prefix_tokens = [0]
    for point in points:
        prefix_count.append(prefix_count[-1] + weighted[point])
        prefix_tokens.append(prefix_tokens[-1] + point * weighted[point])

    # dp[g][j] = minimum padding and endpoints for points[:j] in g groups.
    dp: list[list[tuple[int, tuple[int, ...]] | None]] = [
        [None] * (len(points) + 1) for _ in range(groups + 1)
    ]
    dp[0][0] = (0, ())
    for group in range(1, groups + 1):
        for stop in range(group, len(points) + 1):
            candidates = []
            for start in range(group - 1, stop):
                previous = dp[group - 1][start]
                if previous is None:
                    continue
                count = prefix_count[stop] - prefix_count[start]
                tokens = prefix_tokens[stop] - prefix_tokens[start]
                endpoint = points[stop - 1]
                candidates.append(
                    (previous[0] + endpoint * count - tokens, previous[1] + (endpoint,))
                )
            dp[group][stop] = min(candidates)

    result = dp[groups][-1]
    assert result is not None
    return result[1]


def build_plan(profile: Profile, max_graphs: int) -> dict[str, object]:
    selected = select_capture_sizes(
        profile.histogram, max_graphs, profile.source_capture_sizes[-1]
    )
    return {
        "max_graphs": max_graphs,
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
    parser.add_argument("--max-graphs", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    plan = build_plan(parse_profile_log(args.log), args.max_graphs)
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
