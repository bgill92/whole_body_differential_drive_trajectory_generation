# Trajectory Diagnostics: Pose Error and Nonholonomic Slip

**Date:** 2026-08-18
**Status:** Approved

## Goal

After the pipeline solves a trajectory, quantify how well it did:

1. **Pose tracking error** — per knot, the position and orientation error
   between the desired end-effector pose (interpolated path) and the actual
   pose (forward kinematics of the solved joint configuration).
2. **Nonholonomic slip** — per knot interval, the lateral-slip residual of the
   differential-drive base, and whether it exceeds tolerance.

Both are logged to the existing Rerun stream as time series and summarized on
the console. Diagnostics are computed for **both** the sequential-IK result and
the SQP-optimized result, so the optimizer's trade-offs (soft end-effector
tracking vs. slip elimination) are visible side by side.

## Non-Goals

- No static image output (PNG/CSV); Rerun is the single visualization sink.
- No per-axis error breakdown — scalar norms only.
- No enforcement or feedback into the solvers; diagnostics are read-only.

## Design

### New module: `src/diagnostics.rs` (library, re-exported from `wbdd`)

Pure functions, no Rerun dependency, unit-testable.

```rust
pub struct PoseError {
    pub position: f64,    // meters: ||t_actual − t_desired||
    pub orientation: f64, // radians: angle of R_desiredᵀ · R_actual (rotation log norm)
}

pub fn pose_errors(
    goal_poses: &[Isometry3<f64>],
    joint_positions: &[Vec<f64>],
    kinematics: &mut Kinematics,
) -> Vec<PoseError>;

pub fn slip_residuals(
    joint_positions: &[Vec<f64>],
    base: &BaseIndices,
) -> Vec<f64>; // length N−1, signed: sin(θ̄)·Δx − cos(θ̄)·Δy, θ̄ = midpoint yaw

pub struct SlipSummary {
    pub max_abs: f64,
    pub max_index: Option<usize>, // interval of the worst violation; None if all zero
    pub count_above_tol: usize,
}

pub const SLIP_TOLERANCE: f64 = 1e-6; // meters of lateral motion per interval

pub fn summarize_slip(residuals: &[f64]) -> SlipSummary;
```

- `pose_errors` sets joint positions on the chain, runs FK to the end-effector
  frame, and compares against the goal. Position error is the translation
  difference norm; orientation error is the rotation angle of the relative
  rotation (well-conditioned at π via the existing quaternion-based log).
- The slip residual uses the **same formula** as the SQP constraint
  (`trajectory::nonholonomic_linearization`). The residual computation is
  extracted into a shared helper so diagnostics and optimizer cannot drift
  apart. `BaseIndices` moves to (or is re-exported for) shared use; base joint
  indices are resolved from `trajectory.base_joint_names` in the config, with
  a fallback to the known planar joint names when the `trajectory` section is
  absent.

### Rerun logging: `src/visualization.rs` (binary)

New function logging scalar series on the existing `step` sequence timeline
(the one the trajectory playback animates on), so error plots scrub in sync
with the 3D view:

```
diagnostics/ik_position_error       (m)
diagnostics/ik_orientation_error    (rad)
diagnostics/ik_slip_residual        (m, signed)
diagnostics/sqp_position_error
diagnostics/sqp_orientation_error
diagnostics/sqp_slip_residual
```

Slip residual for interval k is logged at knot k+1 (the knot the motion ends
at).

### `src/main.rs` wiring

1. After sequential IK: snapshot `joint_positions` as the IK trajectory.
2. After the SQP pass (when enabled): the optimized trajectory.
3. For each available trajectory: compute pose errors + slip residuals, log to
   Rerun, print a console summary:

```
[ik]  pos err max 1.23e-2 m (knot 42), ori err max 4.5e-2 rad (knot 42)
[ik]  slip max 3.2e-3 m (interval 17), 40/74 intervals above 1e-6 m
[sqp] ...
```

(Scientific notation throughout so sub-1e-4 errors stay visible; the worst
slip interval prints `-` when every residual is exactly zero.)

- SQP disabled or `solve_first_pose_only` set → only the IK series (slip needs
  ≥ 2 knots; skipped with a note otherwise).

### Testing

Unit tests in `#[cfg(test)] mod tests` inside `diagnostics.rs`:

- Straight-line base motion along its heading → all slip residuals ≈ 0.
- Pure lateral base step (Δy with yaw = 0) → residual = −Δy exactly.
- `summarize_slip` picks the right max/index/count.
- `pose_errors` returns zeros when goals are taken from FK of the same
  configurations; known offset yields the offset norm.

## Alternatives Considered

- **All logic in the binary's `visualization.rs`:** fewer files, but the math
  becomes untestable and unavailable to library users. Rejected.
- **Folding diagnostics into `trajectory.rs`:** reuses the linearization
  directly but couples reporting to the optimizer; diagnostics must also run
  when the SQP pass is disabled. Rejected — instead the residual formula is
  shared via a helper.
