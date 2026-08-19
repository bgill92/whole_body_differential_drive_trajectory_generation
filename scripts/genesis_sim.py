#!/usr/bin/env python3
"""Play the project's solved whole-body trajectory in the Genesis simulator.

Runs the Rust pipeline with WBDD_RRD_PATH set — the same save hook
scripts/render_demo.py uses — reads the animated joint transforms back out of
the Rerun recording, and plays them through the diff-drive base + UR5e arm
loaded in Genesis (https://github.com/Genesis-Embodied-AI/Genesis). Playback
is kinematic: qpos is driven along the solved knots; there is no dynamics or
controller tracking (a follow-up PR adds ROS control).

Requires `cargo` on PATH (and `ffmpeg` for the GIF capture) plus a Python
environment with `genesis-world` and `rerun-sdk` — the rerun-sdk version must
match the crate's rerun dependency in Cargo.toml (0.34.1). Genesis is a heavy
pip dependency and is intentionally not part of CI.

Usage:
  python3 scripts/genesis_sim.py              # headless, writes assets/genesis_demo.gif
  python3 scripts/genesis_sim.py --viewer     # interactive viewer, no capture
  python3 scripts/genesis_sim.py --rrd f.rrd  # reuse an existing recording
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_PATH = os.path.join(ROOT, "assets", "rox_diff_ur5e.urdf")
DEFAULT_OUT = os.path.join(ROOT, "assets", "genesis_demo.gif")


def require(name):
    path = shutil.which(name)
    if not path:
        sys.exit(f"genesis_sim: {name} not found on PATH")
    return path


def patch_urdf(src, dst_dir):
    """Copy the URDF for Genesis: the virtual planar base joints carry no
    <limit> (Genesis requires one on every prismatic/revolute joint), and
    mesh paths must resolve from the copy's new location."""
    tree = ET.parse(src)
    root = tree.getroot()
    for joint in root.iter("joint"):
        if joint.get("type") in ("revolute", "prismatic") and joint.find("limit") is None:
            limit = ET.SubElement(joint, "limit")
            limit.set("lower", "-1e+16")
            limit.set("upper", "1e+16")
            limit.set("velocity", "100")
            limit.set("effort", "1000")
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if filename and not os.path.isabs(filename):
            mesh.set("filename", os.path.abspath(os.path.join(os.path.dirname(src), filename)))
    dst = os.path.join(dst_dir, os.path.basename(src))
    tree.write(dst)
    return dst


def parse_urdf_joints(src):
    """joint name -> (type, axis, origin_xyz, origin_rpy)."""
    joints = {}
    for joint in ET.parse(src).getroot().iter("joint"):
        name = joint.get("name")
        jtype = joint.get("type")
        if jtype not in ("revolute", "prismatic"):
            continue
        axis_el = joint.find("axis")
        axis = [float(v) for v in axis_el.get("xyz", "0 0 1").split()] if axis_el is not None else [0, 0, 1]
        origin = joint.find("origin")
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()] if origin is not None else [0, 0, 0]
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()] if origin is not None else [0, 0, 0]
        joints[name] = (jtype, np.array(axis), np.array(xyz), np.array(rpy))
    return joints


def rpy_to_mat(rpy):
    r, p, y = rpy

    def rot(axis, a):
        c, s = np.cos(a), np.sin(a)
        if axis == 0:
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        if axis == 1:
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    return rot(2, y) @ rot(1, p) @ rot(0, r)


def quat_to_mat(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def joint_position(jtype, axis, origin_xyz, origin_rpy, translation, quaternion):
    """Invert the joint transform logged by the Rust pipeline: the viewer
    animates each URDF joint with origin * motion(q), so motion is recovered
    by stripping the origin and projecting onto the joint axis."""
    r_origin = rpy_to_mat(origin_rpy)
    r_motion = r_origin.T @ quat_to_mat(quaternion)
    t_motion = r_origin.T @ (np.array(translation) - origin_xyz)
    if jtype == "prismatic":
        return float(t_motion @ axis)
    # Rotation about `axis` by theta: R - R.T = 2 sin(theta) [axis]x.
    sin_t = 0.5 * (
        axis[0] * (r_motion[2, 1] - r_motion[1, 2])
        + axis[1] * (r_motion[0, 2] - r_motion[2, 0])
        + axis[2] * (r_motion[1, 0] - r_motion[0, 1])
    )
    cos_t = 0.5 * (np.trace(r_motion) - 1)
    return float(np.arctan2(sin_t, cos_t))


def load_trajectory(rrd_path, urdf_joints):
    """Read the /transforms entity back out of the recording: one row per
    (step, joint), labelled by the joint's child link. Returns the knots as a
    list of {joint name: position} in step order."""
    import rerun as rr

    reader = rr.experimental.RrdReader(rrd_path)
    store = reader.store()

    # child link -> joint name
    link_to_joint = {}
    for joint in ET.parse(URDF_PATH).getroot().iter("joint"):
        child = joint.find("child")
        if child is not None:
            link_to_joint[child.get("link")] = joint.get("name")

    import pyarrow as pa

    tables = [
        pa.Table.from_batches([chunk.to_record_batch()])
        for chunk in store.stream()
        if str(chunk.entity_path) == "/transforms"
    ]
    table = pa.concat_tables(tables)
    steps = table.column("step").to_pylist()
    frames = table.column("Transform3D:child_frame").to_pylist()
    translations = table.column("Transform3D:translation").to_pylist()
    quaternions = table.column("Transform3D:quaternion").to_pylist()

    knots = {}
    for step, frame, translation, quaternion in zip(steps, frames, translations, quaternions):
        joint_name = link_to_joint.get(frame[0])
        if joint_name is None:
            continue
        jtype, axis, origin_xyz, origin_rpy = urdf_joints[joint_name]
        knots.setdefault(step, {})[joint_name] = joint_position(
            jtype, axis, origin_xyz, origin_rpy, translation[0], quaternion[0]
        )
    return [knots[step] for step in sorted(knots)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rrd", help="reuse an existing recording instead of solving")
    parser.add_argument("--viewer", action="store_true", help="interactive viewer instead of headless capture")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"GIF output path (default {DEFAULT_OUT})")
    parser.add_argument("--fps", type=int, default=10, help="GIF frame rate (default 10)")
    args = parser.parse_args()

    if not args.rrd:
        require("cargo")
    if not args.viewer:
        require("ffmpeg")

    work = tempfile.mkdtemp(prefix="wbdd-genesis-")
    rrd = args.rrd or os.path.join(work, "demo.rrd")
    if not args.rrd:
        print("solving trajectory and recording to", rrd)
        subprocess.run(
            ["cargo", "run", "--release"],
            cwd=ROOT,
            env={**os.environ, "WBDD_RRD_PATH": rrd},
            check=True,
        )

    urdf_joints = parse_urdf_joints(URDF_PATH)
    knots = load_trajectory(rrd, urdf_joints)
    print(f"loaded {len(knots)} trajectory knots")

    import yaml

    with open(os.path.join(ROOT, "assets", "config.yaml")) as f:
        dt = float(yaml.safe_load(f).get("trajectory", {}).get("dt", 0.5))

    import genesis as gs

    gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(show_viewer=args.viewer)
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.URDF(file=patch_urdf(URDF_PATH, work), fixed=False))
    camera = None
    if not args.viewer:
        camera = scene.add_camera(res=(640, 480), pos=(3.6, -3.6, 2.6), lookat=(0, 0, 0.5), fov=45)
    scene.build()

    dof_of = {}
    for joint in robot.joints:
        if joint.name in urdf_joints:
            idx = joint.dofs_idx_local
            dof_of[joint.name] = idx[0] if isinstance(idx, (list, tuple)) else idx

    frames = os.path.join(work, "frames")
    os.makedirs(frames, exist_ok=True)
    for k, knot in enumerate(knots):
        qpos = np.zeros(robot.n_dofs)
        for name, position in knot.items():
            qpos[dof_of[name]] = position
        robot.set_qpos(qpos)
        scene.visualizer.update(force=True)
        if camera is not None:
            rgb = camera.render()[0]
            import PIL.Image

            PIL.Image.fromarray(np.asarray(rgb)).save(f"{frames}/frame_{k:04d}.png")
            print(f"rendered knot {k}/{len(knots) - 1}")
        else:
            time.sleep(dt)

    if camera is None:
        print("playback finished")
        return

    subprocess.run(
        [
            require("ffmpeg"),
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(args.fps),
            "-pattern_type",
            "glob",
            "-i",
            f"{frames}/frame_*.png",
            "-vf",
            "split[s0][s1];"
            "[s0]palettegen=max_colors=256:stats_mode=full[p];"
            "[s1][p]paletteuse=dither=sierra2_4a",
            "-loop",
            "0",
            args.out,
        ],
        check=True,
    )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
