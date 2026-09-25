"""Keep generated and imported training commands away from the held-out benchmark.

A command is too close when its token set overlaps a ``benchmark_v2`` case at Jaccard >= 0.8.
Digits are collapsed to one token, so "move standup to 10" and "move standup to 11" collide.
"""

from __future__ import annotations

import re

from .benchmark_v2 import CASES

TOKEN = re.compile(r"[a-z0-9']+")
THRESHOLD = 0.8


def tokens(text: str) -> frozenset[str]:
    return frozenset("#" if token.isdigit() else token for token in TOKEN.findall(text.casefold()))


BENCHMARK = tuple(tokens(case.command) for case in CASES)


def closest_case(command: str) -> int | None:
    """Index of the first benchmark case within the threshold, else None."""
    mine = tokens(command)
    if not mine:
        return None
    for index, theirs in enumerate(BENCHMARK):
        shared = len(mine & theirs)
        if shared and shared / len(mine | theirs) >= THRESHOLD:
            return index
    return None
