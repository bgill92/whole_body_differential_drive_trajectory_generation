#!/usr/bin/env python3
"""Run the Genesis robot under genesis_ros, driven by ROS joint commands.

The simulation half of the genesis_ros control demo: starts the
genesis_ros bridge (https://github.com/vybhav-ibr/genesis_ros, pinned commit
documented in the README) with the scene/robot config in
assets/genesis_ros.yaml, which loads the same patched URDF and plane as
scripts/genesis_sim.py. The solved trajectory is NOT played back kinematically
here — it arrives over ROS as joint commands published by
scripts/wbdd_trajectory_publisher.py, and genesis_ros turns them into Genesis
motor commands (velocity on the planar base joints, position on the UR5e arm).

Headless mode captures a camera frame per trajectory knot and encodes
assets/genesis_ros_demo.gif (needs ffmpeg); the step loop is paced to real
time so the command stream and the simulation stay in sync. --viewer shows an
interactive window instead.

Requires a ROS 2 environment with genesis_ros (and its Genesis dependency)
installed — see the README's "ROS Control Demo (genesis_ros)" section — plus
`rerun-sdk` matching the crate's rerun dependency (0.34.1) and `pyarrow` to
load the recording. Run the Rust pipeline first with WBDD_RRD_PATH (or pass
--rrd).

Usage:
  python3 scripts/genesis_ros_sim.py --rrd demo.rrd            # headless GIF
  python3 scripts/genesis_ros_sim.py --rrd demo.rrd --viewer   # interactive
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import yaml

import genesis_common as gc

ROOT = gc.ROOT
DEFAULT_CONFIG = os.path.join(ROOT, "assets", "genesis_ros.yaml")
DEFAULT_OUT = os.path.join(ROOT, "assets", "genesis_ros_demo.gif")
SETTLE_S = 2.0  # must match wbdd_trajectory_publisher.py
TAIL_S = 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rrd", help="recording of the solved trajectory (required headless)")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help=f"genesis_ros scene config (default {DEFAULT_CONFIG})")
    parser.add_argument("--viewer", action="store_true", help="interactive viewer instead of headless capture")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"GIF output path (default {DEFAULT_OUT})")
    parser.add_argument("--fps", type=int, default=10, help="GIF frame rate (default 10)")
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu", help="Genesis backend (default cpu)")
    args = parser.parse_args()

    duration = None
    if args.rrd:
        knots = gc.load_trajectory(args.rrd, gc.parse_urdf_joints(gc.URDF_PATH))
        duration = (len(knots) - 1) * gc.load_solver_dt()
        print(f"loaded {len(knots)} trajectory knots, duration {duration:.1f}s")
    elif not args.viewer:
        sys.exit("genesis_ros_sim: --rrd is required for headless capture")
    if not args.viewer:
        gc.require("ffmpeg")

    work = tempfile.mkdtemp(prefix="wbdd-genesis-ros-")
    with open(args.config) as f:
        scene_config = yaml.safe_load(f)
    robot_name, robot_config = next(iter(scene_config["robots"].items()))
    robot_config["morph"]["path"] = gc.patch_urdf(gc.URDF_PATH, work)
    if args.viewer:
        scene_config["scene"]["show_Viewer"] = True
    patched_config = os.path.join(work, "scene.yaml")
    with open(patched_config, "w") as f:
        yaml.safe_dump(scene_config, f)

    import genesis as gs
    import rclpy
    from gs_ros import GsRosBridge
    from rclpy.node import Node

    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.gpu, logging_level="warning")
    rclpy.init()
    node = Node("gs_ros_bridge_node")
    bridge = GsRosBridge(node, patched_config)
    camera = None
    if not args.viewer:
        camera = bridge.scene.add_camera(res=(640, 480), pos=(3.6, -3.6, 2.6), lookat=(0, 0, 0.5), fov=45)
    bridge.build()
    control = robot_config["control"]
    namespace = control.get("namespace", robot_config.get("namespace", "robot"))
    print(f"bridge built: robot '{robot_name}' listening on "
          f"/{namespace}/{control.get('joint_commands_topic', 'joint_commands')}")

    frames = os.path.join(work, "frames")
    os.makedirs(frames, exist_ok=True)
    dt_knot = gc.load_solver_dt()
    t0 = bridge.scene.cur_t
    wall_start = time.monotonic()
    next_capture = 0.0
    frame_idx = 0
    try:
        while rclpy.ok():
            bridge.step()
            sim_t = bridge.scene.cur_t - t0

            if camera is not None:
                target = wall_start + sim_t
                ahead = target - time.monotonic()
                if ahead > 0:
                    time.sleep(ahead)
                if sim_t >= next_capture:
                    rgb = camera.render()[0]
                    import PIL.Image

                    PIL.Image.fromarray(np.asarray(rgb)).save(f"{frames}/frame_{frame_idx:04d}.png")
                    print(f"captured frame {frame_idx} at sim time {sim_t:.2f}s")
                    frame_idx += 1
                    next_capture += dt_knot

            if duration is not None and sim_t > duration + SETTLE_S + TAIL_S:
                print(f"reached sim time {sim_t:.2f}s, stopping")
                break
    except KeyboardInterrupt:
        print("interrupted")

    if camera is not None and frame_idx > 0:
        subprocess.run(
            [
                gc.require("ffmpeg"),
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

    if rclpy.ok():
        rclpy.shutdown()
    del bridge
    gs.destroy()


if __name__ == "__main__":
    main()
