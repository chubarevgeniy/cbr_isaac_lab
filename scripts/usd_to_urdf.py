#!/usr/bin/env python3
"""Convert a USD robot articulation (e.g. CBR-I.usda) to URDF + OBJ meshes.

Runs standalone — only needs the pip USD package, no Isaac Sim:

    pip install usd-core numpy

Usage (from the repo root, on the machine that has the .usda file):

    python scripts/usd_to_urdf.py \
        source/CBRIIsaacLab/CBRIIsaacLab/robots/CBR-I.usda \
        --out ../cbr_ros/src/cbr/urdf --name cbr

    # inspect what would be converted without writing anything:
    python scripts/usd_to_urdf.py path/to/CBR-I.usda --list

Output layout (what the cbr ROS package expects):

    <out>/cbr.urdf
    <out>/meshes/<link>.obj

The converter reads UsdPhysics Revolute/Prismatic/Fixed joints, uses each
joint's frame on the child body as that link's URDF frame (so URDF q=0 ==
USD joint q=0), and bakes every UsdGeomMesh under a rigid body into a
single OBJ per link, expressed in the link frame. Collision geometry and
accurate inertia are intentionally omitted: the output is meant for
visualization (RViz digital twin), not dynamics.
"""

import argparse
import math
import os
import sys

import numpy as np

try:
    from pxr import Usd, UsdGeom, UsdPhysics, Gf  # noqa: F401
except ImportError:
    sys.exit("The 'pxr' module is missing. Install it with: pip install usd-core")


# ---------------------------------------------------------------------------
# Math helpers. Gf matrices use the row-vector convention (p' = p * M);
# everything below converts to the standard column-vector convention first.
# ---------------------------------------------------------------------------

def gf_to_np(m) -> np.ndarray:
    return np.array(m, dtype=float).T


def quat_to_np(q) -> np.ndarray:
    """Gf.Quatf/Quatd -> 3x3 rotation matrix (column-vector convention)."""
    return gf_to_np(Gf.Matrix3d(Gf.Quatd(q)))


def frame_from_pos_rot(pos, rot) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = quat_to_np(rot if rot is not None else Gf.Quatf(1.0))
    if pos is not None:
        t[:3, 3] = np.array(pos, dtype=float)
    return t


def strip_scale(world_mat) -> np.ndarray:
    """World transform of a prim with any scale removed (for link/joint frames)."""
    tr = Gf.Transform(world_mat)
    t = np.eye(4)
    t[:3, :3] = quat_to_np(tr.GetRotation().GetQuat())
    t[:3, 3] = np.array(tr.GetTranslation(), dtype=float)
    return t


def rpy_from_mat(t: np.ndarray):
    """URDF fixed-axis roll/pitch/yaw from a 4x4 (R = Rz(y) @ Ry(p) @ Rx(r))."""
    r_mat = t[:3, :3]
    pitch = math.asin(max(-1.0, min(1.0, -r_mat[2, 0])))
    if abs(math.cos(pitch)) > 1e-9:
        roll = math.atan2(r_mat[2, 1], r_mat[2, 2])
        yaw = math.atan2(r_mat[1, 0], r_mat[0, 0])
    else:  # gimbal lock
        roll = math.atan2(-r_mat[1, 2], r_mat[1, 1])
        yaw = 0.0
    return roll, pitch, yaw


def origin_xml(t: np.ndarray, scale: float) -> str:
    x, y, z = t[:3, 3] * scale
    r, p, yw = rpy_from_mat(t)
    return f'<origin xyz="{x:.6f} {y:.6f} {z:.6f}" rpy="{r:.6f} {p:.6f} {yw:.6f}"/>'


# ---------------------------------------------------------------------------
# USD parsing
# ---------------------------------------------------------------------------

class JointInfo:
    def __init__(self, prim, xf_cache):
        self.name = prim.GetName()
        joint = UsdPhysics.Joint(prim)

        b0 = joint.GetBody0Rel().GetTargets()
        b1 = joint.GetBody1Rel().GetTargets()
        self.body0 = str(b0[0]) if b0 else None  # None -> anchored to world
        if not b1:
            raise ValueError(f"Joint {prim.GetPath()} has no body1 target")
        self.body1 = str(b1[0])

        self.local0 = frame_from_pos_rot(
            joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
        self.local1 = frame_from_pos_rot(
            joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())

        if prim.IsA(UsdPhysics.RevoluteJoint):
            self.type = "revolute"
            j = UsdPhysics.RevoluteJoint(prim)
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            self.type = "prismatic"
            j = UsdPhysics.PrismaticJoint(prim)
        else:
            self.type = "fixed"
            j = None

        self.axis = np.array([1.0, 0.0, 0.0])
        self.lower = self.upper = None
        if j is not None:
            token = j.GetAxisAttr().Get() or "X"
            self.axis = {"X": np.array([1.0, 0.0, 0.0]),
                         "Y": np.array([0.0, 1.0, 0.0]),
                         "Z": np.array([0.0, 0.0, 1.0])}[str(token)]
            lo, hi = j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()
            # USD limits are in degrees (revolute) / stage units (prismatic)
            if lo is not None and hi is not None and lo > -1e6 and hi < 1e6 and lo < hi:
                if self.type == "revolute":
                    self.lower, self.upper = math.radians(lo), math.radians(hi)
                else:
                    self.lower, self.upper = float(lo), float(hi)


def collect_meshes(body_prim, bodies_paths):
    """All renderable UsdGeomMesh prims under a body, not crossing into other bodies."""
    meshes = []
    it = iter(Usd.PrimRange(body_prim))
    for prim in it:
        path = str(prim.GetPath())
        if path != str(body_prim.GetPath()) and path in bodies_paths:
            it.PruneChildren()
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        img = UsdGeom.Imageable(prim)
        if img.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if img.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
            continue
        meshes.append(prim)
    return meshes


def write_link_obj(path, mesh_prims, x_link_world_inv, xf_cache, scale):
    """Merge all meshes of one link into a single OBJ, in the link frame."""
    v_lines, f_lines = [], []
    offset = 0
    for prim in mesh_prims:
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts or not indices:
            continue
        x_world_mesh = gf_to_np(xf_cache.GetLocalToWorldTransform(prim))
        t = x_link_world_inv @ x_world_mesh  # full matrix: keeps mesh scale baked in
        pts = np.asarray(points, dtype=float)
        pts = (t[:3, :3] @ pts.T).T + t[:3, 3]
        pts *= scale
        left_handed = mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded
        for p in pts:
            v_lines.append(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
        i = 0
        for c in counts:
            face = [int(indices[i + k]) + 1 + offset for k in range(c)]
            if left_handed:
                face.reverse()
            for k in range(1, c - 1):  # triangle fan
                f_lines.append(f"f {face[0]} {face[k]} {face[k + 1]}")
            i += c
        offset += len(pts)
    if not v_lines:
        return False
    with open(path, "w") as f:
        f.write("\n".join(v_lines) + "\n" + "\n".join(f_lines) + "\n")
    return True


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert(usd_path, out_dir, robot_name, mesh_uri_prefix, list_only=False):
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        sys.exit(f"Could not open {usd_path}")
    scale = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        print("WARNING: stage is not Z-up; URDF will keep the USD axes as-is.")

    xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    bodies = {}   # path -> prim
    joints = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies[str(prim.GetPath())] = prim
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint) \
                or prim.IsA(UsdPhysics.FixedJoint):
            try:
                joints.append(JointInfo(prim, xf_cache))
            except ValueError as e:
                print(f"WARNING: skipping joint: {e}")

    if not bodies:
        sys.exit("No prims with UsdPhysics.RigidBodyAPI found — is this the physics asset?")

    print(f"Stage: {usd_path} (metersPerUnit={scale})")
    print(f"Rigid bodies ({len(bodies)}):")
    for p in bodies:
        print(f"  {p}")
    print(f"Joints ({len(joints)}):")
    for j in joints:
        lim = f" limits=[{j.lower:.3f}, {j.upper:.3f}]" if j.lower is not None else ""
        print(f"  {j.name}: {j.type} {j.body0 or '<world>'} -> {j.body1}"
              f" axis={j.axis.tolist()}{lim}")
    if list_only:
        return

    link_name = {p: bodies[p].GetName() for p in bodies}

    # Link frames (world, scale-stripped). Roots use the body frame; every
    # joint child uses the joint frame on the child body, so URDF q=0 matches
    # USD joint q=0 exactly.
    body_world = {p: strip_scale(xf_cache.GetLocalToWorldTransform(bodies[p]))
                  for p in bodies}
    link_frame = dict(body_world)
    child_paths = set()
    for j in joints:
        link_frame[j.body1] = body_world[j.body1] @ j.local1
        child_paths.add(j.body1)

    urdf_links = {}   # name -> list of xml chunks (visuals)
    urdf_joints = []  # xml chunks

    mesh_dir = os.path.join(out_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    for p, prim in bodies.items():
        name = link_name[p]
        chunks = [f'  <link name="{name}">',
                  "    <inertial>",
                  '      <mass value="0.1"/>',
                  '      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>',
                  "    </inertial>"]
        mesh_prims = collect_meshes(prim, set(bodies))
        obj_name = f"{name}.obj"
        if mesh_prims and write_link_obj(os.path.join(mesh_dir, obj_name), mesh_prims,
                                         np.linalg.inv(link_frame[p]), xf_cache, scale):
            chunks += ["    <visual>",
                       '      <origin xyz="0 0 0" rpy="0 0 0"/>',
                       "      <geometry>",
                       f'        <mesh filename="{mesh_uri_prefix}/meshes/{obj_name}"/>',
                       "      </geometry>",
                       "    </visual>"]
        else:
            print(f"NOTE: no visual mesh found for link '{name}'")
        chunks.append("  </link>")
        urdf_links[name] = "\n".join(chunks)

    # Fixed world root so RViz always has a stable fixed frame named 'world'.
    world_link = '  <link name="world"/>'
    roots = [p for p in bodies if p not in child_paths]
    for p in roots:
        urdf_joints.append(
            f'  <joint name="world_to_{link_name[p]}" type="fixed">\n'
            f"    {origin_xml(link_frame[p], scale)}\n"
            f'    <parent link="world"/>\n'
            f'    <child link="{link_name[p]}"/>\n'
            f"  </joint>")

    for j in joints:
        if j.body0 is None:
            parent_name, x_parent = "world", np.eye(4)
        else:
            parent_name = link_name[j.body0]
            x_parent = link_frame[j.body0]
        x_world_j0 = body_world[j.body0] @ j.local0 if j.body0 else j.local0
        origin = np.linalg.inv(x_parent) @ x_world_j0
        jtype = j.type
        limit = ""
        if jtype == "revolute":
            if j.lower is None:
                jtype = "continuous"
            else:
                limit = (f'\n    <limit lower="{j.lower:.6f}" upper="{j.upper:.6f}"'
                         f' effort="100" velocity="100"/>')
        elif jtype == "prismatic":
            lo = j.lower if j.lower is not None else -1.0
            hi = j.upper if j.upper is not None else 1.0
            limit = (f'\n    <limit lower="{lo * scale:.6f}" upper="{hi * scale:.6f}"'
                     f' effort="100" velocity="100"/>')
        axis = ""
        if jtype != "fixed":
            axis = f'\n    <axis xyz="{j.axis[0]:.0f} {j.axis[1]:.0f} {j.axis[2]:.0f}"/>'
        urdf_joints.append(
            f'  <joint name="{j.name}" type="{jtype}">\n'
            f"    {origin_xml(origin, scale)}\n"
            f'    <parent link="{parent_name}"/>\n'
            f'    <child link="{link_name[j.body1]}"/>{axis}{limit}\n'
            f"  </joint>")

    urdf = "\n".join(
        ['<?xml version="1.0"?>',
         f'<robot name="{robot_name}">',
         world_link,
         *urdf_links.values(),
         *urdf_joints,
         "</robot>", ""])

    urdf_path = os.path.join(out_dir, f"{robot_name}.urdf")
    with open(urdf_path, "w") as f:
        f.write(urdf)
    print(f"\nWrote {urdf_path}")
    print(f"Meshes in {mesh_dir}")
    print("Copy the whole output directory into cbr_ros/src/cbr/urdf/ "
          "and rebuild the cbr package (colcon build --packages-select cbr).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("usd", help="Path to the robot .usd/.usda file")
    ap.add_argument("--out", default="urdf_out", help="Output directory")
    ap.add_argument("--name", default="cbr", help="Robot name (also URDF file name)")
    ap.add_argument("--mesh-uri-prefix", default="package://cbr/urdf",
                    help="Prefix for mesh filenames in the URDF")
    ap.add_argument("--list", action="store_true",
                    help="Only print bodies/joints found, write nothing")
    args = ap.parse_args()
    convert(args.usd, args.out, args.name, args.mesh_uri_prefix, list_only=args.list)


if __name__ == "__main__":
    main()
