# SQP Whole-Body Trajectory Optimization — Design

Date: 2026-08-14. Status: approved approach A (dense Gauss-Newton SQP over the
existing `qp::solve`); this document is the spec for the implementation plan.

## Problem

The current pipeline solves each interpolated pose with an independent
differential-IK QP. Nothing couples consecutive knots: the planar base can slide
sideways (violating differential-drive kinematics), joint velocities are
unbounded, and there is no smoothness. This feature optimizes the whole
trajectory at once, subject to the diff-drive nonholonomic constraint, with the
sequential-IK result as warm start.

## Decisions (from brainstorming)

- **EE tracking**: config switch `ee_tracking: soft | hard`. Soft = weighted
  quadratic cost per knot. Hard = linearized EE pose equality rows per knot
  (may be infeasible; surfaced as an error, documented as expected when the
  path fights base kinematics).
- **Timing**: fixed uniform `dt` from config. Velocities are finite
  differences. Free/total-time optimization is out of scope.
- **Base model**: keep planar x/y/yaw joints as variables; nonholonomic
  no-lateral-slip equality per interval, linearized each SQP iteration.
- **v1 scope**: joint position limits, joint velocity limits, smoothness cost.
  No wheel-speed limits, no obstacle avoidance, no torque/dynamics.
- **Solver**: dense assembly, one QP per SQP iteration through the existing
  `qp::solve` (extended with general inequality rows). Sparse block-banded
  assembly is deliberately deferred until a path size demands it.

## Formulation

Variables: `Q = [q_0, …, q_{N-1}]`, stacked length `N·n` (N = interpolated
pose count, n = serial-chain joint count, ~10). Knots coincide with the poses
from `path.interpolate()`. Each SQP iteration solves for the step `ΔQ`.

### Cost (quadratic model in ΔQ)

- **EE tracking (soft mode)**: per knot, residual `r_k = se3_log(T_goal_k ·
  fk(q_k)^{-1})` (the existing pose-error twist) with Gauss-Newton model
  `r_k − J_k Δq_k`; contributes `w_ee · (J_kᵀJ_k)` to P's k-th diagonal block
  and `−w_ee · J_kᵀ r_k` to q. `J_k` is the existing `Kinematics::jacobian()`
  evaluated at `q_k`. One scalar weight `w_ee`; per-axis weighting is YAGNI.
- **Smoothness**: `w_s · Σ_k ‖q_{k+1} − q_k‖²` over the full (current + step)
  values — exact quadratic, assembled from the first-difference operator D as
  `w_s·DᵀD` into P and `w_s·DᵀD·Q_current` into q.
- **Damping**: `λ²·I` added to P (config `damping`), keeping P positive
  definite as `qp::solve` requires.

### Constraints

- **Nonholonomic (equality, relinearized each iteration)**: per interval k,
  with midpoint heading `θ̄_k = (θ_k + θ_{k+1})/2`:
  `c_k = sin(θ̄_k)·(x_{k+1} − x_k) − cos(θ̄_k)·(y_{k+1} − y_k) = 0`.
  Linearization rows (all other partials zero):
  - `∂c/∂x_k = −sin θ̄_k`, `∂c/∂x_{k+1} = +sin θ̄_k`
  - `∂c/∂y_k = +cos θ̄_k`, `∂c/∂y_{k+1} = −cos θ̄_k`
  - `∂c/∂θ_k = ∂c/∂θ_{k+1} = ½·(cos θ̄_k·Δx_k + sin θ̄_k·Δy_k)`
  QP rows: `∇c_k · ΔQ = −c_k(Q_current)`.
  Correctness gate: linearization verified against finite differences in a
  unit test.
- **EE tracking (hard mode)**: instead of the tracking cost, equality rows
  `J_k Δq_k = r_k` for every knot (6 rows per knot). Cost then contains only
  smoothness + damping.
- **Joint limits (box)**: `lower − q_k ≤ Δq_k ≤ upper − q_k` for every knot,
  same re-anchoring as the IK loop.
- **Velocity limits (inequality)**: `−v_max·dt ≤ (q_{k+1} + Δq_{k+1}) −
  (q_k + Δq_k) ≤ v_max·dt`, i.e. two general inequality rows per joint per
  interval on ΔQ with rhs re-anchored by the current difference. Single scalar
  `max_joint_velocity` applied to all joints (the `k` crate does not expose
  URDF velocity limits; per-joint limits are YAGNI until it does).
- **Trust region**: `|ΔQ_i| ≤ trust_region` intersected into the box bounds.
  Steps are accepted or rejected against an ℓ1 exact-penalty merit function
  (tracking + smoothness + penalty·|violations|); rejection halves the trust
  region, acceptance regrows it toward the configured radius. Added after
  plain acceptance limit-cycled on L-shaped paths — the originally-documented
  ceiling.
- **Not constrained**: the config `equality_constraints` (pinned joints) apply
  only to the warm-start IK, as today; the trajectory optimizer is free to
  move every joint. Pinning knots inside the trajectory is YAGNI.

### SQP loop

```
Q ← warm start (sequential IK results, one row per knot)
repeat up to sqp_max_iterations:
    assemble P, q, A_eq, b_eq, A_in, b_in, lb, ub at Q   (relinearize)
    ΔQ ← qp::solve(...)                                   (one dense QP)
    Q ← clamp(Q + ΔQ, joint limits)
    stop when ‖ΔQ‖∞ < convergence_step_norm
return Q as Vec<Vec<f64>> (one row per knot)
```

Between assembly steps, per-knot FK/Jacobians come from the existing
`Kinematics` by `set_positions(q_k)` per knot (mutable borrow, sequential —
no parallelism needed at this size).

## Interfaces

### `qp::solve` extension

Add general inequality rows to the existing signature:

```rust
pub(crate) fn solve(p, q, a_eq, b_eq, a_in, b_in, lb, ub) -> Result<DVector<f64>, String>
// new: a_in: &DMatrix<f64>, b_in: &DVector<f64>, rows meaning A_in·x ≤ b_in
```

Implementation: the `a_in` rows are appended into the existing
`NonnegativeConeT` block ahead of the finite-bound rows. The IK call site
passes zero-row `a_in`. Existing qp tests unchanged plus new inequality-row
tests.

### New module `src/trajectory.rs`

```rust
pub fn optimize(
    goal_poses: &[k::Isometry3<f64>],
    kinematics: &mut Kinematics,
    config: &TrajectoryConfig,
    warm_start: &[Vec<f64>],   // one row per knot, from sequential IK
) -> Result<Vec<Vec<f64>>, String>
```

One responsibility: SQP loop + subproblem assembly. Internals: an
`assemble_subproblem` helper and a `nonholonomic_rows` helper (unit-testable
against finite differences without a URDF). Validation at entry: N ≥ 2,
`warm_start.len() == goal_poses.len()`, base joint names resolve to chain
indices, `dt > 0`, `trust_region > 0`, mode-specific weights present.

### Config (`assets/config.yaml` + `src/configs.rs`)

```yaml
trajectory:
  enabled: true               # false = current behavior, IK only
  dt: 0.5                     # seconds per interval
  ee_tracking: soft           # soft | hard
  ee_weight: 100.0            # soft mode only
  smoothness_weight: 1.0
  max_joint_velocity: 1.0     # rad/s or m/s, all joints
  sqp_max_iterations: 20
  trust_region: 0.2           # per-variable step bound per SQP iteration
  convergence_step_norm: 1.0e-4
  damping: 1.0e-3
  base_joint_names: [world_base_link_planar_prismatic_x, world_base_link_planar_prismatic_y, world_base_link_planar_yaw]
```

`base_joint_names` are the planar x, y, yaw joints in order; validated against
the serial chain at startup (names verified against `assets/rox_diff_ur5e.urdf`).
`ee_tracking` deserializes into a two-variant enum.

### `main.rs` flow

Sequential IK loop (unchanged, provides warm start) → if `trajectory.enabled`,
run `trajectory::optimize` and visualize its result; otherwise visualize the
IK result as today. The optimized trajectory feeds the existing
`log_trajectory` unchanged (same `Vec<Vec<f64>>` shape).

## Error handling

- Hard mode infeasibility: `qp_not_solved: PrimalInfeasible` from `qp::solve`
  is wrapped with context: which SQP iteration, and a hint that `ee_tracking:
  hard` may conflict with base kinematics (try `soft`). Before erroring, hard
  mode retries the iteration with the linearized EE targets scaled by α =
  0.5…0.0625; relaxation applies only to the EE rows, never the no-slip rows.
- All config validation errors name the offending field/joint.
- Non-convergence within `sqp_max_iterations` is NOT an error: return the last
  iterate with a stderr log line (matches IK loop behavior, which also returns
  best-effort after `num_steps`).

## Testing

- `qp`: inequality-row tests (active inequality clamps the optimum; inactive
  inequality leaves it; combined with equalities).
- `trajectory` unit: `nonholonomic_rows` vs finite differences on random-ish
  fixed configurations; velocity-limit row assembly rhs re-anchoring.
- `trajectory` integration (uses the real URDF, like existing kinematics
  tests): 3–5 knot straight-line path; assert after optimization
  (a) `|c_k| < 1e-6` every interval (no lateral slip),
  (b) `|Δq|/dt ≤ max_joint_velocity + 1e-9` every interval,
  (c) mean EE tracking error stays bounded (< 0.25 combined twist norm, soft
      mode) — it cannot beat the unconstrained warm start, which the original
      draft of this line wrongly claimed,
  (d) hard mode on a base-reachable path solves; hard mode on a deliberately
  side-stepping path returns the infeasibility error.
- Existing 16 tests unchanged (the `qp::solve` signature change updates call
  sites mechanically, not behavior).

## Out of scope (recorded so they stay out)

Sparse assembly (revisit when N·n ≳ 1000 — also reevaluate POUNCE then, per
`docs/qp-sqp-solver-crates.md`), free time, wheel-speed limits, obstacle
avoidance, torque/dynamics, merit-function line search, per-joint velocity
limits.
