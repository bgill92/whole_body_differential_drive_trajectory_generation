# Whole-Body Differential-Drive Trajectory Generation

Whole-body trajectory generation for a mobile manipulator: a differential-drive
base (Neobotix ROX) carrying a UR5e arm with a Robotiq gripper. Given a set of
end-effector waypoint poses in the world frame, the program produces a joint
trajectory for the whole robot — base and arm together — that tracks the
end-effector path while respecting the base's no-lateral-slip (nonholonomic)
constraint, and streams the result to the [Rerun](https://rerun.io) viewer for
3D playback.

## TL;DR

```bash
# 1. Install Rust (stable): https://rustup.rs
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 2. Install the Rerun viewer (must be on PATH)
cargo install rerun-cli   # or: pip install rerun-sdk

# 3. Clone and run
git clone <this-repo-url>
cd whole_body_differential_drive_trajectory_generation
cargo run --release       # uses assets/config.yaml
```

A Rerun window opens with the robot, the end-effector path, and the solved
whole-body trajectory playing back. Edit `assets/config.yaml` (or pass your own
config: `cargo run --release -- path/to/config.yaml`) to change waypoints and
solver settings.

## How It Works

The pipeline has three stages:

1. **Path interpolation** — Waypoints from `assets/config.yaml` are densified
   into `poses_per_segment` poses per segment (translation lerp, rotation
   slerp). Interior corners can be rounded with tangent circular arcs
   (`corner_radius`); see
   [docs/superpowers/specs/2026-08-14-corner-rounding-design.md](docs/superpowers/specs/2026-08-14-corner-rounding-design.md).

2. **Sequential differential IK** — Each interpolated pose is solved with a
   damped differential IK loop, one dense QP per step
   ([Clarabel](https://clarabel.org) interior-point solver). Each solve seeds
   from the previous solution so the trajectory stays continuous. The pose
   error is the SE(3) log of the goal-relative transform, and joint-space
   equality constraints (e.g. pinning the shoulder or elbow to a preferred
   configuration) can be applied to the first pose to pick a good starting
   posture. Optionally the base yaw is pinned for the first pose so the base
   x-axis points along the path.

3. **Whole-trajectory SQP** — A Gauss-Newton SQP pass over all knot
   configurations at once, warm-started from the sequential IK result, one
   dense Clarabel QP per iteration. It enforces:
   - the differential-drive **no-lateral-slip constraint**, linearized at the
     midpoint heading of each knot interval,
   - **joint velocity limits** via finite differences over `dt`,
   - **smoothness** (quadratic penalty on velocity),
   - an optional one-sided **backward-motion penalty** so the base prefers
     driving forward,
   - end-effector tracking, either **soft** (weighted cost) or **hard** (exact
     equality constraints, which may be infeasible).

   Design notes:
   [docs/superpowers/specs/2026-08-14-sqp-trajectory-optimization-design.md](docs/superpowers/specs/2026-08-14-sqp-trajectory-optimization-design.md).

After solving, the program reports **diagnostics** for both the sequential-IK
and SQP trajectories: per-knot end-effector position/orientation error and the
per-interval lateral-slip residual of the base, logged as Rerun time series
(`diagnostics/{ik,sqp}_...` on the `step` timeline) with a console summary of
the worst violations.

## Requirements

- Rust (edition 2024 — a recent stable toolchain)
- The [Rerun viewer](https://rerun.io) — the program spawns it automatically
  via `rerun::RecordingStreamBuilder::spawn()`, so the `rerun` binary must be
  on your `PATH` (e.g. `cargo install rerun-cli` or `pip install rerun-sdk`)

## Running

```bash
cargo run --release                       # uses assets/config.yaml
cargo run --release -- path/to/config.yaml
```

A Rerun viewer window opens showing the robot model, the interpolated
end-effector path with goal axes, and the solved whole-body trajectory played
back over time. The program blocks on exit until all data (including ~50 MiB
of URDF meshes) reaches the viewer.

## Configuration

Everything is driven by a YAML config (see `assets/config.yaml` for the
commented reference). Summary of the sections:

### Top level

| Key | Meaning |
|-----|---------|
| `urdf_path` | Robot URDF (default `assets/rox_diff_ur5e.urdf`) |
| `end_joint` | Joint whose child frame is the IK target (default `grasp_link_joint`) |

### `path`

| Key | Meaning |
|-----|---------|
| `waypoints` | List of `{xyz, rpy}` end-effector poses in the world frame (rpy in radians) |
| `poses_per_segment` | Interpolated poses per waypoint segment |
| `corner_radius` | Radius (m) of tangent arcs rounding interior corners; `0` leaves corners sharp; clamped per corner to fit |
| `solve_first_pose_only` | Debug: solve only the first pose (skips the SQP pass) |
| `align_first_pose_base_yaw` | Pin base yaw for the first pose so the base faces along the path |

### `differential_ik`

| Key | Meaning |
|-----|---------|
| `num_steps` | Max IK iterations per pose |
| `damping_factor` | Damping on the QP step |
| `convergence_threshold` | Pose-error norm at which the loop stops |
| `equality_constraints` | `{joint_name, target_value}` pins, applied to the **first pose only** |

### `trajectory`

| Key | Meaning |
|-----|---------|
| `enabled` | Toggle the SQP pass |
| `dt` | Time between knots (s), used for velocity limits |
| `ee_tracking` | `soft` (weighted cost) or `hard` (exact, may be infeasible) |
| `ee_weight` | End-effector tracking weight (soft mode) |
| `smoothness_weight` | Quadratic velocity penalty weight |
| `backward_weight` | One-sided penalty on backward base motion; `0` allows reversing |
| `max_joint_velocity` | Per-joint velocity limit (rad/s or m/s) |
| `sqp_max_iterations` | Max Gauss-Newton iterations |
| `trust_region` | Per-iteration step bound |
| `convergence_step_norm` | Step norm at which SQP stops |
| `damping` | Levenberg-style regularization on the QP Hessian |
| `base_joint_names` | Names of the planar base joints `[x, y, yaw]` |

## Project Layout

```
src/
  main.rs           # binary: load config, run the pipeline, log to Rerun
  lib.rs            # library crate `wbdd` (flat public API via re-exports)
  configs.rs        # config structs + YAML parsing + path interpolation
  kinematics.rs     # k-based FK/Jacobians, SE(3) log, differential IK loop
  qp.rs             # generic dense convex QP wrapper over Clarabel
  diagnostics.rs    # pose-tracking error + nonholonomic slip residuals
  trajectory.rs     # Gauss-Newton SQP whole-trajectory optimizer
  visualization.rs  # Rerun logging (URDF, path, goal axes, trajectory)
assets/
  config.yaml       # reference configuration
  rox_diff_ur5e.urdf# ROX base + UR5e + Robotiq gripper, with a planar
                    # (x, y, yaw) joint stack connecting world to base
  meshes/           # visual + collision meshes
docs/
  qp-sqp-solver-crates.md        # solver crate survey
  rust-best-practices.md         # Rust conventions for this repo
  superpowers/specs/             # design docs (corner rounding, SQP)
```

The mobile base is modeled with three planar joints
(`world_base_link_planar_prismatic_x`, `world_base_link_planar_prismatic_y`,
`world_base_link_planar_yaw`) so the whole robot is a single kinematic chain;
the differential-drive constraint is imposed by the optimizer rather than the
model.

## Development

```bash
cargo fmt
cargo check --all-targets   # plain `check` skips #[cfg(test)] code
cargo test
```

The crate builds warning-clean. Contributor conventions live in
[AGENTS.md](AGENTS.md).

## Dependencies

| Crate | Role |
|-------|------|
| [`k`](https://crates.io/crates/k) | Kinematics: URDF chain, FK, Jacobians (nalgebra-based) |
| [`clarabel`](https://crates.io/crates/clarabel) | Interior-point convex QP solver |
| [`rerun`](https://crates.io/crates/rerun) | 3D visualization and trajectory playback |
| `serde` / `serde_yaml_ng` | YAML config parsing |
