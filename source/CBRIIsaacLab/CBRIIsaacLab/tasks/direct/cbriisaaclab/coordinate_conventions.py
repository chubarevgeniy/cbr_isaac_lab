"""Canonical joint-coordinate conventions used by the CBR-I task."""

from __future__ import annotations

import torch


def raw_actuated_to_canonical(raw: torch.Tensor, hip_down_angle: float) -> torch.Tensor:
    """Convert [right hip, left hip, right knee, left knee] from USD signs."""

    canonical = raw.clone()
    canonical[..., 0] = hip_down_angle - raw[..., 0]
    canonical[..., 1] = raw[..., 1] + hip_down_angle
    canonical[..., 2] = -raw[..., 2]
    canonical[..., 3] = raw[..., 3]
    return canonical


def canonical_actuated_to_raw(canonical: torch.Tensor, hip_down_angle: float) -> torch.Tensor:
    """Convert [right hip, left hip, right knee, left knee] to USD signs."""

    raw = canonical.clone()
    raw[..., 0] = hip_down_angle - canonical[..., 0]
    raw[..., 1] = canonical[..., 1] - hip_down_angle
    raw[..., 2] = -canonical[..., 2]
    raw[..., 3] = canonical[..., 3]
    return raw

