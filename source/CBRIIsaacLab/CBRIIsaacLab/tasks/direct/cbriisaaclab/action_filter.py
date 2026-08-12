"""Torch action filters shared by training and Isaac Sim policy playback."""

from __future__ import annotations

import torch


def second_order_action_filter_step(
    position: torch.Tensor,
    velocity: torch.Tensor,
    target: torch.Tensor,
    *,
    dt: float,
    response_time: float,
    max_velocity: torch.Tensor,
    max_acceleration: torch.Tensor,
    output_min: torch.Tensor,
    output_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance an acceleration- and velocity-limited action trajectory.

    ``position`` is the action sent to the position target controller and
    ``velocity`` is its internal action-space velocity.  The implementation
    mirrors ``SecondOrderActionFilter`` in ``scripts/skrl/numpy_policy.py``.
    All limits are per-action vectors and are broadcast over environments.
    """

    target = torch.minimum(torch.maximum(target, output_min), output_max)
    error = target - position
    desired_velocity = torch.clamp(
        error / float(response_time),
        min=-max_velocity,
        max=max_velocity,
    )
    max_velocity_step = max_acceleration * float(dt)
    new_velocity = velocity + torch.clamp(
        desired_velocity - velocity,
        min=-max_velocity_step,
        max=max_velocity_step,
    )
    new_position = position + 0.5 * (velocity + new_velocity) * float(dt)

    crossed_target = ((error > 0.0) & (new_position > target)) | (
        (error < 0.0) & (new_position < target)
    )
    new_position = torch.where(crossed_target, target, new_position)
    new_velocity = torch.where(crossed_target, torch.zeros_like(new_velocity), new_velocity)

    below_min = new_position < output_min
    new_position = torch.maximum(new_position, output_min)
    new_velocity = torch.where(
        below_min & (new_velocity < 0.0),
        torch.zeros_like(new_velocity),
        new_velocity,
    )

    above_max = new_position > output_max
    new_position = torch.minimum(new_position, output_max)
    new_velocity = torch.where(
        above_max & (new_velocity > 0.0),
        torch.zeros_like(new_velocity),
        new_velocity,
    )
    return new_position, new_velocity


__all__ = ["second_order_action_filter_step"]
