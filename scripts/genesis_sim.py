#!/usr/bin/env python3
"""Play the project's solved whole-body trajectory in the Genesis simulator.

Runs the Rust pipeline with WBDD_RRD_PATH set — the same save hook
scripts/render_demo.py uses — reads the animated joint transforms back out of
the Rerun recording, and plays them through the diff-drive base + UR5e arm
loaded in Genesis (https://github.com/Genesis-Embodied-AI/Genesis). Playback
is kinematic: qpos is driven along the solved knots; there is no dynamics or
controller tracking (scripts/genesis_ros_sim.py closes the loop with ROS-based
control through genesis_ros).

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
import subprocess
import sys
import tempfile
import time

import numpy as np

import genesis_common as gc

ROOT = gc.ROOT
URDF_PATH = gc.URDF_PATH
DEFAULT_OUT = os.path.join(ROOT, "assets", "genesis_demo.gif")

require = gc.require
patch_urdf = gc.patch_urdf
parse_urdf_joints = gc.parse_urdf_joints
load_trajectory = gc.load_trajectory


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

    dt = gc.load_solver_dt()

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
