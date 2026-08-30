"""Strict WCM checkpoint compatibility across Transformers ViT key renames."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_MODERN_TO_LEGACY_VIT_PARTS = {
    "attention.q_proj": "attention.attention.query",
    "attention.k_proj": "attention.attention.key",
    "attention.v_proj": "attention.attention.value",
    "attention.o_proj": "attention.output.dense",
    "layernorm_before": "layernorm_before",
    "layernorm_after": "layernorm_after",
    "mlp.fc1": "intermediate.dense",
    "mlp.fc2": "output.dense",
}


def _alternate_vit_state_key(key: str) -> str | None:
    modern_prefix = "vision_encoder.backbone.layers."
    legacy_prefix = "vision_encoder.backbone.encoder.layer."
    if key.startswith(modern_prefix):
        source_prefix = modern_prefix
        target_prefix = legacy_prefix
        translations = _MODERN_TO_LEGACY_VIT_PARTS
    elif key.startswith(legacy_prefix):
        source_prefix = legacy_prefix
        target_prefix = modern_prefix
        translations = {
            value: name for name, value in _MODERN_TO_LEGACY_VIT_PARTS.items()
        }
    else:
        return None

    layer, separator, tail = key[len(source_prefix) :].partition(".")
    if not separator or not layer.isdigit():
        return None
    for source, target in translations.items():
        parameter_prefix = f"{source}."
        if tail.startswith(parameter_prefix):
            return (
                f"{target_prefix}{layer}.{target}."
                f"{tail[len(parameter_prefix):]}"
            )
    return None


def _summarize_keys(keys: list[str], *, limit: int = 8) -> str:
    shown = ", ".join(repr(key) for key in keys[:limit])
    if len(keys) > limit:
        shown += f", ... ({len(keys)} total)"
    return f"[{shown}]"


def adapt_wcm_state_dict(
    checkpoint_state: Mapping[str, Any],
    target_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize known ViT key renames, then require exact compatibility."""

    target_keys = set(target_state)
    adapted: dict[str, Any] = {}
    for source_key, value in checkpoint_state.items():
        if not isinstance(source_key, str):
            raise TypeError("WCM checkpoint state_dict keys must be strings.")
        target_key = source_key
        if source_key not in target_keys:
            alternate = _alternate_vit_state_key(source_key)
            if alternate in target_keys:
                target_key = alternate
        if target_key in adapted:
            raise RuntimeError(
                "WCM checkpoint key normalization produced duplicate target key "
                f"{target_key!r}."
            )
        adapted[target_key] = value

    missing = sorted(target_keys - set(adapted))
    unexpected = sorted(set(adapted) - target_keys)
    shape_mismatches = sorted(
        key
        for key in target_keys & set(adapted)
        if getattr(adapted[key], "shape", None)
        != getattr(target_state[key], "shape", None)
    )
    if missing or unexpected or shape_mismatches:
        details = []
        if missing:
            details.append(f"missing={_summarize_keys(missing)}")
        if unexpected:
            details.append(f"unexpected={_summarize_keys(unexpected)}")
        if shape_mismatches:
            details.append(f"shape_mismatches={_summarize_keys(shape_mismatches)}")
        raise RuntimeError(
            "WCM checkpoint is incompatible with the configured model after known "
            f"Transformers ViT key normalization: {'; '.join(details)}"
        )
    return adapted
