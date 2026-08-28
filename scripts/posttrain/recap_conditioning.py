"""Policy-specific language conditioning for RECAP datasets."""

from __future__ import annotations

import hashlib
import re

_CONDITION = re.compile(r"\nAdvantage: (?:positive|negative)\s*$")


def strip_condition(task: str) -> str:
    """Remove a condition previously written by this pipeline."""

    return _CONDITION.sub("", str(task).rstrip())


def training_prompt(
    policy: str,
    task: str,
    positive: bool,
    *,
    unconditional_probability: float,
    seed: int,
    episode: int,
    frame: int,
) -> str:
    """Return the prompt consumed by a policy's unmodified training loader."""

    base = strip_condition(task)
    if policy == "pi05":
        return f"{base}\nAdvantage: {'positive' if positive else 'negative'}"
    if policy != "g05":
        raise ValueError(f"Unsupported RECAP policy: {policy}")
    if not positive:
        return base
    key = f"{seed}:{episode}:{frame}".encode()
    sample = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    return base if sample < unconditional_probability else f"{base}\nAdvantage: positive"
