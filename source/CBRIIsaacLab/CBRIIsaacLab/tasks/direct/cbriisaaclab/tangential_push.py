# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tangential disturbances for the fixed-base CBR-I articulation."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply


def push_body_tangential_impulse(
    env,
    env_ids: torch.Tensor | None,
    velocity_change_magnitude_range: tuple[float, float],
    disturbed_mass_kg: float,
    rotor_body_name: str,
    rotor_axis_local: tuple[float, float, float],
    rotor_pivot_offset_local: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["body"]),
) -> None:
    """Apply a one-physics-step impulse to the moving body along the rotor tangent.

    ``push_by_setting_velocity`` is not suitable here: the articulation root is
    ``Rock`` and is fixed to the world.  This event instead applies a force only
    to the selected ``body`` link.  The force is computed from a Unitree-style
    instantaneous velocity kick for the approximately 5 kg moving assembly.

    The tangent is recomputed in the world frame from the current body COM and
    the ``Rock_Revolute_1`` pivot.  Consequently the force has no radial or
    axial component, even when the body is tilted or the environment is cloned
    at a different world position.
    """

    asset = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(asset.num_instances, device=asset.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=asset.device, dtype=torch.long)

    if env_ids.numel() == 0:
        return

    if asset_cfg.body_ids is None or isinstance(asset_cfg.body_ids, slice):
        raise ValueError("Tangential push requires an explicit single body in asset_cfg.body_names")
    body_ids = torch.as_tensor(asset_cfg.body_ids, device=asset.device, dtype=torch.long).flatten()
    if body_ids.numel() != 1:
        raise ValueError("Tangential push currently supports exactly one disturbed body")

    rotor_body_ids, _ = asset.find_bodies(rotor_body_name, preserve_order=True)
    if len(rotor_body_ids) != 1:
        raise ValueError(f"Expected exactly one rotor body matching {rotor_body_name!r}")
    rotor_body_id = rotor_body_ids[0]

    body_com_pos_w = asset.data.body_com_pos_w.torch[env_ids][:, body_ids[0], :]
    rotor_link_pos_w = asset.data.body_link_pos_w.torch[env_ids][:, rotor_body_id, :]
    rotor_link_quat_w = asset.data.body_link_quat_w.torch[env_ids][:, rotor_body_id, :]

    pivot_offset_local = torch.tensor(
        rotor_pivot_offset_local,
        device=asset.device,
        dtype=body_com_pos_w.dtype,
    ).expand(env_ids.numel(), -1)
    axis_local = torch.tensor(
        rotor_axis_local,
        device=asset.device,
        dtype=body_com_pos_w.dtype,
    ).expand(env_ids.numel(), -1)

    rotor_pivot_w = rotor_link_pos_w + quat_apply(rotor_link_quat_w, pivot_offset_local)
    rotor_axis_w = quat_apply(rotor_link_quat_w, axis_local)
    rotor_axis_w = rotor_axis_w / torch.linalg.vector_norm(rotor_axis_w, dim=-1, keepdim=True).clamp_min(1.0e-6)

    radius_w = body_com_pos_w - rotor_pivot_w
    radius_tangent_plane_w = radius_w - rotor_axis_w * torch.sum(
        radius_w * rotor_axis_w,
        dim=-1,
        keepdim=True,
    )
    tangent_w = torch.cross(rotor_axis_w, radius_tangent_plane_w, dim=-1)
    tangent_w = tangent_w / torch.linalg.vector_norm(tangent_w, dim=-1, keepdim=True).clamp_min(1.0e-6)

    velocity_change_magnitude = torch.empty(
        (env_ids.numel(), 1),
        device=asset.device,
        dtype=body_com_pos_w.dtype,
    ).uniform_(*velocity_change_magnitude_range)
    velocity_change_sign = torch.where(
        torch.rand_like(velocity_change_magnitude) < 0.5,
        -torch.ones_like(velocity_change_magnitude),
        torch.ones_like(velocity_change_magnitude),
    )
    velocity_change = velocity_change_magnitude * velocity_change_sign
    impulse = velocity_change * float(disturbed_mass_kg)
    force_w = tangent_w * (impulse / float(env.physics_dt)).unsqueeze(-1)

    forces = force_w.unsqueeze(1)
    torques = torch.zeros_like(forces)
    asset.instantaneous_wrench_composer.add_forces_and_torques_index(
        forces=forces,
        torques=torques,
        body_ids=body_ids,
        env_ids=env_ids,
        is_global=True,
    )
