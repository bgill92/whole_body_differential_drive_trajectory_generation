#!/usr/bin/env python3
"""Publish the solved whole-body trajectory as ROS 2 joint commands.

The ROS half of the genesis_ros control demo: reads the solved trajectory
back out of the Rerun recording (the same WBDD_RRD_PATH hook
scripts/genesis_sim.py uses) and republishes it as sensor_msgs/JointState
commands on {namespace}/joint_commands, which the genesis_ros bridge
(scripts/genesis_ros_sim.py) subscribes to and turns into Genesis motor
commands. The virtual planar base joints are velocity-commanded and the UR5e
arm joints position-commanded, matching the joint_properties in
assets/genesis_ros.yaml.

Timing follows the simulator's /clock topic (published by the bridge), so the
commands stay synchronized with simulation time no matter how fast or slow
the simulation steps.

Requires a ROS 2 environment (rclpy) plus `rerun-sdk` matching the crate's
rerun dependency in Cargo.toml (0.34.1) and `pyarrow`; run the Rust pipeline
first with WBDD_RRD_PATH (or pass --rrd). See the README's "ROS Control Demo
(genesis_ros)" section.

Usage:
  ros2 run-style standalone:  python3 scripts/wbdd_trajectory_publisher.py --rrd demo.rrd
"""

import argparse
import os
import subprocess
import sys

import yaml

import genesis_common as gc

ROOT = gc.ROOT
DEFAULT_CONFIG = os.path.join(ROOT, "assets", "genesis_ros.yaml")
SETTLE_S = 2.0  # keep commanding zero base velocity after the last knot


def load_knots(rrd):
    if not rrd:
        gc.require("cargo")
        rrd = os.path.join(os.getcwd(), "wbdd_trajectory.rrd")
        print("solving trajectory and recording to", rrd)
        subprocess.run(
            ["cargo", "run", "--release"],
            cwd=ROOT,
            env={**os.environ, "WBDD_RRD_PATH": rrd},
            check=True,
        )
    knots = gc.load_trajectory(rrd, gc.parse_urdf_joints(gc.URDF_PATH))
    print(f"loaded {len(knots)} trajectory knots")
    return knots


def command_groups(control_config):
    """Split joint_properties into position/velocity groups.

    genesis_ros consumes a JointState command's position array in the
    alphabetical joint_properties order of the position-commanded joints (and
    likewise velocity), so both groups are returned sorted by joint name to
    match that dispatch (gs_ros/gs_ros/gs_ros_robot_control.py).
    """
    props = control_config["joint_properties"]
    all_joints = sorted(props)
    pos_joints = [j for j in all_joints if props[j].get("command", "").lower() == "position"]
    vel_joints = [j for j in all_joints if props[j].get("command", "").lower() == "velocity"]
    unknown = [j for j in all_joints if j not in pos_joints and j not in vel_joints]
    if unknown:
        sys.exit(f"publisher: unsupported command type for joints: {unknown}")
    return all_joints, pos_joints, vel_joints


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rrd", help="reuse an existing recording instead of solving")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help=f"genesis_ros scene config (default {DEFAULT_CONFIG})")
    parser.add_argument("--rate", type=float, default=50.0, help="command publish rate in Hz (default 50)")
    args = parser.parse_args()

    knots = load_knots(args.rrd)
    dt = gc.load_solver_dt()
    duration = (len(knots) - 1) * dt

    with open(args.config) as f:
        scene_config = yaml.safe_load(f)
    robot_config = next(iter(scene_config["robots"].values()))
    control = robot_config["control"]
    # genesis_ros resolves the control-topic namespace from the control block
    # (GsRosRobotControl), falling back to "robot".
    namespace = control.get("namespace", robot_config.get("namespace", "robot"))
    all_joints, pos_joints, vel_joints = command_groups(control)

    import rclpy
    from rosgraph_msgs.msg import Clock
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = Node("wbdd_trajectory_publisher")
    commands = node.create_publisher(JointState, f"{namespace}/{control.get('joint_commands_topic', 'joint_commands')}", 10)
    latest = {"t": None}

    def on_clock(msg):
        latest["t"] = msg.clock.sec + msg.clock.nanosec * 1e-9

    node.create_subscription(Clock, "/clock", on_clock, 10)

    def trajectory_command(t):
        """(positions for pos_joints, velocities for vel_joints) at sim time t."""
        if t <= 0.0:
            k, alpha = 0, 0.0
        elif t >= duration:
            k, alpha = len(knots) - 2, 1.0
        else:
            k = int(t // dt)
            alpha = (t - k * dt) / dt
        q_now, q_next = knots[k], knots[min(k + 1, len(knots) - 1)]
        positions = [q_now[j] + alpha * (q_next[j] - q_now[j]) for j in pos_joints]
        if t < duration:
            velocities = [(q_next[j] - q_now[j]) / dt for j in vel_joints]
        else:
            velocities = [0.0] * len(vel_joints)
        return positions, velocities

    print(f"waiting for /clock (is the bridge running?)")
    while rclpy.ok() and latest["t"] is None:
        rclpy.spin_once(node, timeout_sec=0.5)
    if not rclpy.ok():
        return
    print(f"trajectory duration {duration:.1f}s ({len(knots)} knots at dt={dt}s), commands indexed by /clock")

    sent_first = False
    last_t = None
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=1.0 / args.rate)
        t = latest["t"]
        if t is None or t == last_t:
            continue  # sim time has not advanced; nothing new to command
        last_t = t
        positions, velocities = trajectory_command(t)
        msg = JointState()
        msg.name = all_joints
        msg.position = positions
        msg.velocity = velocities
        msg.effort = []
        commands.publish(msg)
        if not sent_first:
            print(f"first command at sim time {t:.2f}s")
            sent_first = True
        if t > duration + SETTLE_S:
            print(f"trajectory complete + {SETTLE_S:.0f}s settle at sim time {t:.2f}s")
            break

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
