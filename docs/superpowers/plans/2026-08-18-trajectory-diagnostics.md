# Trajectory Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute and visualize per-knot end-effector pose error and per-interval nonholonomic slip for both the sequential-IK and SQP trajectories.

**Architecture:** A new pure `diagnostics` module in the `wbdd` library owns the math (`BaseIndices` and the slip-residual formula move there from `trajectory.rs` so optimizer and diagnostics share one implementation). The binary logs the series to Rerun on the existing `step` timeline and prints a console summary.

**Tech Stack:** Rust edition 2024, `k` (nalgebra) for kinematics, `rerun` 0.34 `Scalars` archetype for time-series plots.

**Spec:** `docs/superpowers/specs/2026-08-18-trajectory-diagnostics-design.md`

**Conventions:** AGENTS.md applies — run `cargo fmt && cargo check --all-targets && cargo test` before every commit. Never change matrix-multiplication order.

---

### Task 1: `diagnostics` module — `BaseIndices` + slip residual (moved from `trajectory.rs`)

**Files:**
- Create: `src/diagnostics.rs`
- Modify: `src/lib.rs`
- Modify: `src/trajectory.rs` (delete moved code, import instead)

- [ ] **Step 1: Create `src/diagnostics.rs` with the failing tests**

```rust
//! Read-only trajectory diagnostics: end-effector pose-tracking error and
//! differential-drive lateral-slip residuals. Never feeds back into the
//! solvers. See docs/superpowers/specs/2026-08-18-trajectory-diagnostics-design.md.

/// Chain indices of the planar base joints.
pub struct BaseIndices {
    pub x: usize,
    pub y: usize,
    pub yaw: usize,
}

/// Find the [x, y, yaw] planar base joints in the serial chain.
pub fn resolve_base_indices(
    joint_names: &[String],
    base_joint_names: &[String],
) -> Result<BaseIndices, String> {
    if base_joint_names.len() != 3 {
        return Err(format!(
            "base_joint_names must list exactly [x, y, yaw], got {} entries",
            base_joint_names.len()
        ));
    }
    let mut indices = [0usize; 3];
    for (slot, name) in base_joint_names.iter().enumerate() {
        indices[slot] = joint_names
            .iter()
            .position(|joint| joint == name)
            .ok_or_else(|| format!("base joint '{name}' not in serial chain"))?;
    }
    Ok(BaseIndices {
        x: indices[0],
        y: indices[1],
        yaw: indices[2],
    })
}

/// Lateral slip of the base over one knot interval:
/// sin(θ̄)·Δx − cos(θ̄)·Δy with θ̄ the midpoint heading. Zero iff the base
/// motion satisfies the differential-drive no-lateral-slip constraint.
/// Signed: positive is slip to the base's left.
pub fn slip_residual(q_k: &[f64], q_k1: &[f64], base: &BaseIndices) -> f64 {
    let dx = q_k1[base.x] - q_k[base.x];
    let dy = q_k1[base.y] - q_k[base.y];
    let mid_yaw = 0.5 * (q_k[base.yaw] + q_k1[base.yaw]);
    let (sin_mid, cos_mid) = mid_yaw.sin_cos();
    sin_mid * dx - cos_mid * dy
}

#[cfg(test)]
mod tests {
    use super::*;

    // Base joints at chain indices 0..2, one arm joint behind them.
    fn base() -> BaseIndices {
        BaseIndices { x: 0, y: 1, yaw: 2 }
    }

    #[test]
    fn slip_zero_when_driving_along_heading() {
        // Heading 45°, motion along [cos45, sin45]: pure forward roll.
        let yaw = std::f64::consts::FRAC_PI_4;
        let q0 = vec![0.0, 0.0, yaw, 0.3];
        let q1 = vec![yaw.cos(), yaw.sin(), yaw, -0.2];
        assert!(slip_residual(&q0, &q1, &base()).abs() < 1e-12);
    }

    #[test]
    fn slip_equals_negative_dy_at_zero_yaw() {
        // Pure sideways step with yaw = 0: residual = −Δy exactly.
        let q0 = vec![0.0, 0.0, 0.0, 0.0];
        let q1 = vec![0.0, 0.25, 0.0, 0.0];
        assert!((slip_residual(&q0, &q1, &base()) - (-0.25)).abs() < 1e-12);
    }

    #[test]
    fn resolve_base_indices_finds_joints() {
        let names: Vec<String> = ["px", "py", "yaw", "arm"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let base_names: Vec<String> =
            ["px", "py", "yaw"].iter().map(|s| s.to_string()).collect();
        let base = resolve_base_indices(&names, &base_names).unwrap();
        assert_eq!((base.x, base.y, base.yaw), (0, 1, 2));
    }

    #[test]
    fn resolve_base_indices_rejects_missing_joint() {
        let names: Vec<String> = ["px"].iter().map(|s| s.to_string()).collect();
        let base_names: Vec<String> =
            ["px", "py", "yaw"].iter().map(|s| s.to_string()).collect();
        assert!(resolve_base_indices(&names, &base_names).is_err());
    }
}
```

- [ ] **Step 2: Register the module and re-exports in `src/lib.rs`**

Replace the whole file with:

```rust
mod configs;
mod diagnostics;
mod kinematics;
mod qp;
pub mod trajectory;

pub use crate::configs::{
    Config, DifferentialIkConfig, EeTracking, EqualityConstraint, TrajectoryConfig,
};
pub use crate::diagnostics::{BaseIndices, resolve_base_indices, slip_residual};
pub use crate::kinematics::{Kinematics, differential_ik};
```

(Task 2 and Task 3 extend this `pub use` list.)

- [ ] **Step 3: Run tests, verify the new ones pass**

Run: `cargo test diagnostics`
Expected: 4 passed.

- [ ] **Step 4: Delete the moved code from `src/trajectory.rs` and import it**

In `src/trajectory.rs`:

1. Add to the imports at the top:

```rust
use crate::diagnostics::{BaseIndices, resolve_base_indices, slip_residual};
```

2. Delete the local `struct BaseIndices { ... }` definition (around line 13) and the entire local `fn resolve_base_indices(...)` (around line 94).

3. In `nonholonomic_linearization`, replace the residual computation with a call to the shared helper — the partials still need `dx`/`dy`/`sin_mid`/`cos_mid`, so only the `residual` line changes:

```rust
    let residual = slip_residual(q_k, q_k1, base);
```

(the old line was `let residual = sin_mid * dx - cos_mid * dy;`).

- [ ] **Step 5: Full check and test**

Run: `cargo fmt && cargo check --all-targets && cargo test`
Expected: warning-clean, all tests pass (35 existing + 4 new).

- [ ] **Step 6: Commit**

```bash
git add src/diagnostics.rs src/lib.rs src/trajectory.rs
git commit -m "refactor: extract shared base-slip residual into diagnostics module"
```

---

### Task 2: Slip series and summary

**Files:**
- Modify: `src/diagnostics.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Add failing tests to `src/diagnostics.rs` `mod tests`**

```rust
    #[test]
    fn slip_residuals_returns_one_per_interval() {
        let traj = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.5, 0.0, 0.0],
        ];
        let residuals = slip_residuals(&traj, &base());
        assert_eq!(residuals.len(), 2);
        assert!(residuals[0].abs() < 1e-12); // forward along +x
        assert!((residuals[1] - (-0.5)).abs() < 1e-12); // lateral step
    }

    #[test]
    fn summarize_slip_finds_max_and_counts() {
        let summary = summarize_slip(&[1e-9, -3e-3, 2e-4]);
        assert!((summary.max_abs - 3e-3).abs() < 1e-15);
        assert_eq!(summary.max_index, 1);
        assert_eq!(summary.count_above_tol, 2);
    }

    #[test]
    fn summarize_slip_empty_is_clean() {
        let summary = summarize_slip(&[]);
        assert_eq!(summary.max_abs, 0.0);
        assert_eq!(summary.count_above_tol, 0);
    }
```

- [ ] **Step 2: Run tests, verify they fail to compile (functions missing)**

Run: `cargo test diagnostics`
Expected: compile error, `slip_residuals` / `summarize_slip` not found.

- [ ] **Step 3: Implement in `src/diagnostics.rs`**

```rust
/// Slip tolerance in meters of lateral motion per knot interval. Residuals
/// below this are solver noise, not violations.
// ponytail: constant, promote to config if a use case needs tuning it.
pub const SLIP_TOLERANCE: f64 = 1e-6;

/// Lateral-slip residual for every knot interval; length is one less than
/// the trajectory. Empty for trajectories with fewer than two knots.
pub fn slip_residuals(joint_positions: &[Vec<f64>], base: &BaseIndices) -> Vec<f64> {
    joint_positions
        .windows(2)
        .map(|pair| slip_residual(&pair[0], &pair[1], base))
        .collect()
}

/// Worst and out-of-tolerance slip over a residual series.
pub struct SlipSummary {
    pub max_abs: f64,
    /// Knot interval of the worst violation; 0 for an empty series.
    pub max_index: usize,
    pub count_above_tol: usize,
}

pub fn summarize_slip(residuals: &[f64]) -> SlipSummary {
    let mut max_abs = 0.0;
    let mut max_index = 0;
    let mut count_above_tol = 0;
    for (i, r) in residuals.iter().enumerate() {
        if r.abs() > max_abs {
            max_abs = r.abs();
            max_index = i;
        }
        if r.abs() > SLIP_TOLERANCE {
            count_above_tol += 1;
        }
    }
    SlipSummary {
        max_abs,
        max_index,
        count_above_tol,
    }
}
```

- [ ] **Step 4: Extend the diagnostics re-export in `src/lib.rs`**

```rust
pub use crate::diagnostics::{
    BaseIndices, SLIP_TOLERANCE, SlipSummary, resolve_base_indices, slip_residual,
    slip_residuals, summarize_slip,
};
```

(Exact wrapping is rustfmt's call — run `cargo fmt` and keep its layout.)

- [ ] **Step 5: Run tests**

Run: `cargo fmt && cargo check --all-targets && cargo test`
Expected: all pass, warning-clean.

- [ ] **Step 6: Commit**

```bash
git add src/diagnostics.rs src/lib.rs
git commit -m "feat: add slip-residual series and violation summary"
```

---

### Task 3: Pose-tracking error

**Files:**
- Modify: `src/diagnostics.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Add failing tests to `src/diagnostics.rs` `mod tests`**

These build the real URDF chain (same pattern as existing config/kinematics tests; `assets/` is checked in).

```rust
    use crate::kinematics::Kinematics;

    #[test]
    fn pose_errors_zero_against_own_fk() {
        let mut kinematics =
            Kinematics::build("assets/rox_diff_ur5e.urdf", "grasp_link_joint").unwrap();
        let q = kinematics.positions();
        kinematics.set_positions(&q);
        let goals = vec![kinematics.end_pose()];
        let errors = pose_errors(&goals, &[q], &mut kinematics);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].position < 1e-12);
        assert!(errors[0].orientation < 1e-12);
    }

    #[test]
    fn pose_errors_reports_translation_offset() {
        let mut kinematics =
            Kinematics::build("assets/rox_diff_ur5e.urdf", "grasp_link_joint").unwrap();
        let q = kinematics.positions();
        kinematics.set_positions(&q);
        let mut goal = kinematics.end_pose();
        goal.translation.vector.x += 0.1;
        let errors = pose_errors(&[goal], &[q], &mut kinematics);
        assert!((errors[0].position - 0.1).abs() < 1e-12);
        assert!(errors[0].orientation < 1e-12);
    }

    #[test]
    fn pose_errors_restores_kinematics_state() {
        let mut kinematics =
            Kinematics::build("assets/rox_diff_ur5e.urdf", "grasp_link_joint").unwrap();
        let before = kinematics.positions();
        let mut q_other = before.clone();
        q_other[0] += 1.0;
        let goals = vec![kinematics.end_pose()];
        let _ = pose_errors(&goals, &[q_other], &mut kinematics);
        assert_eq!(kinematics.positions(), before);
    }
```

- [ ] **Step 2: Run tests, verify compile failure**

Run: `cargo test diagnostics`
Expected: compile error, `pose_errors` / `PoseError` not found.

- [ ] **Step 3: Implement in `src/diagnostics.rs`**

Add to the top of the file:

```rust
use crate::kinematics::Kinematics;
```

Then:

```rust
/// End-effector tracking error at one knot.
pub struct PoseError {
    /// Meters: ‖t_actual − t_desired‖.
    pub position: f64,
    /// Radians: rotation angle between desired and actual orientation.
    pub orientation: f64,
}

/// Pose-tracking error per knot: forward kinematics of each configuration
/// against the corresponding goal pose. Pairs by index and stops at the
/// shorter of the two slices (first-pose debug mode solves one knot against
/// a full goal path). Restores the chain's joint positions on return.
pub fn pose_errors(
    goal_poses: &[k::Isometry3<f64>],
    joint_positions: &[Vec<f64>],
    kinematics: &mut Kinematics,
) -> Vec<PoseError> {
    let saved = kinematics.positions();
    let errors = goal_poses
        .iter()
        .zip(joint_positions)
        .map(|(goal, q)| {
            kinematics.set_positions(q);
            let actual = kinematics.end_pose();
            PoseError {
                position: (actual.translation.vector - goal.translation.vector).norm(),
                // Quaternion angle_to is well-conditioned at all angles
                // including π, unlike the matrix-trace acos route.
                orientation: goal.rotation.angle_to(&actual.rotation),
            }
        })
        .collect();
    kinematics.set_positions(&saved);
    errors
}
```

- [ ] **Step 4: Extend the diagnostics re-export in `src/lib.rs`**

Add `PoseError` and `pose_errors` to the existing `pub use crate::diagnostics::{...}` list, then `cargo fmt`.

- [ ] **Step 5: Run tests**

Run: `cargo fmt && cargo check --all-targets && cargo test`
Expected: all pass, warning-clean.

- [ ] **Step 6: Commit**

```bash
git add src/diagnostics.rs src/lib.rs
git commit -m "feat: add per-knot end-effector pose-error diagnostics"
```

---

### Task 4: Rerun logging

**Files:**
- Modify: `src/visualization.rs` (binary-only module — no unit tests here, math is already covered in the library)

- [ ] **Step 1: Add the logging function to `src/visualization.rs`**

```rust
/// Log diagnostics series on the "step" timeline so they scrub in sync with
/// the trajectory playback. `prefix` distinguishes trajectories ("ik", "sqp").
/// The slip residual for interval k is logged at knot k+1 (the knot the
/// motion ends at), so `slip[k]` pairs with `errors[k + 1]`.
pub fn log_diagnostics(
    rec: &rerun::RecordingStream,
    prefix: &str,
    pose_errors: &[wbdd::PoseError],
    slip_residuals: &[f64],
) -> Result<()> {
    for (idx, error) in pose_errors.iter().enumerate() {
        rec.set_time_sequence("step", idx as i64);
        rec.log(
            format!("diagnostics/{prefix}/position_error"),
            &rerun::Scalars::single(error.position),
        )?;
        rec.log(
            format!("diagnostics/{prefix}/orientation_error"),
            &rerun::Scalars::single(error.orientation),
        )?;
    }
    for (idx, residual) in slip_residuals.iter().enumerate() {
        rec.set_time_sequence("step", (idx + 1) as i64);
        rec.log(
            format!("diagnostics/{prefix}/slip_residual"),
            &rerun::Scalars::single(*residual),
        )?;
    }
    Ok(())
}
```

- [ ] **Step 2: Check it compiles**

Run: `cargo fmt && cargo check --all-targets`
Expected: one `dead_code` warning for `log_diagnostics` (wired up in Task 5 — acceptable only at this intermediate step).

- [ ] **Step 3: Do NOT commit yet** — Task 5 wires it up; committing a dead-code warning would break the warning-clean convention. Continue directly to Task 5.

---

### Task 5: Wire into `main.rs` + console summary

**Files:**
- Modify: `src/main.rs`

- [ ] **Step 1: Update imports in `src/main.rs`**

```rust
use wbdd::{
    Config, EqualityConstraint, Kinematics, SLIP_TOLERANCE, differential_ik, pose_errors,
    resolve_base_indices, slip_residuals, summarize_slip, trajectory,
};
```

- [ ] **Step 2: Snapshot the IK trajectory before the SQP pass**

In `main`, immediately before the `if let Some(trajectory_config) = &config.trajectory` block, add:

```rust
    let ik_joint_positions = joint_positions.clone();
```

- [ ] **Step 3: Add the diagnostics helper at the bottom of `src/main.rs`**

```rust
/// Fallback planar base joints when the `trajectory` config section (which
/// names them) is absent.
const DEFAULT_BASE_JOINT_NAMES: [&str; 3] = [
    "world_base_link_planar_prismatic_x",
    "world_base_link_planar_prismatic_y",
    "world_base_link_planar_yaw",
];

/// Compute, log, and print diagnostics for one solved trajectory.
fn report_diagnostics(
    rec: &rerun::RecordingStream,
    label: &str,
    goal_poses: &[k::Isometry3<f64>],
    joint_positions: &[Vec<f64>],
    kinematics: &mut Kinematics,
    base_joint_names: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let errors = pose_errors(goal_poses, joint_positions, kinematics);
    let (max_pos, pos_knot) = errors
        .iter()
        .enumerate()
        .map(|(i, e)| (e.position, i))
        .fold((0.0, 0), |acc, cur| if cur.0 > acc.0 { cur } else { acc });
    let (max_ori, ori_knot) = errors
        .iter()
        .enumerate()
        .map(|(i, e)| (e.orientation, i))
        .fold((0.0, 0), |acc, cur| if cur.0 > acc.0 { cur } else { acc });
    println!(
        "[{label}] pos err max {max_pos:.4} m (knot {pos_knot}), \
         ori err max {max_ori:.4} rad (knot {ori_knot})"
    );

    let slips = if joint_positions.len() < 2 {
        println!("[{label}] slip check skipped: fewer than two knots");
        Vec::new()
    } else {
        let base = resolve_base_indices(&kinematics.joint_names(), base_joint_names)?;
        let slips = slip_residuals(joint_positions, &base);
        let summary = summarize_slip(&slips);
        println!(
            "[{label}] slip max {:.3e} m (interval {}), {}/{} intervals above {SLIP_TOLERANCE:.0e} m",
            summary.max_abs,
            summary.max_index,
            summary.count_above_tol,
            slips.len(),
        );
        slips
    };

    visualization::log_diagnostics(rec, label, &errors, &slips)?;
    Ok(())
}
```

- [ ] **Step 4: Call it from `main`**

After the `visualization::log_trajectory(...)` call and before `rec.flush_blocking()?`, add:

```rust
    let base_joint_names: Vec<String> = config
        .trajectory
        .as_ref()
        .map(|t| t.base_joint_names.clone())
        .unwrap_or_else(|| DEFAULT_BASE_JOINT_NAMES.map(String::from).to_vec());

    report_diagnostics(
        &rec,
        "ik",
        &goal_poses,
        &ik_joint_positions,
        &mut kinematics,
        &base_joint_names,
    )?;
    // The SQP pass replaced joint_positions only when it actually ran; equal
    // trajectories mean it was skipped, so a second report would duplicate.
    if joint_positions != ik_joint_positions {
        report_diagnostics(
            &rec,
            "sqp",
            &goal_poses,
            &joint_positions,
            &mut kinematics,
            &base_joint_names,
        )?;
    }
```

- [ ] **Step 5: Full check, test, run**

Run: `cargo fmt && cargo check --all-targets && cargo test`
Expected: warning-clean (the Task 4 `dead_code` warning is gone), all tests pass.

Then a live smoke test (requires the Rerun viewer; skip if headless and note it):

Run: `cargo run --release`
Expected console output shape:

```
[ik]  pos err max 0.0xxx m (knot NN), ori err max 0.0xxx rad (knot NN)
[ik]  slip max x.xe-xx m (interval NN), NN/74 intervals above 1e-06 m
[sqp] pos err max ...
[sqp] slip max ...
```

and six `diagnostics/...` scalar entities in the viewer that scrub with the 3D playback.

- [ ] **Step 6: Commit (Tasks 4 + 5 together — warning-clean unit)**

```bash
git add src/visualization.rs src/main.rs
git commit -m "feat: plot pose error and slip diagnostics for ik and sqp"
```

---

### Task 6: README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the diagnostics**

In README.md, after the "How It Works" pipeline list, add:

```markdown
After solving, the program reports **diagnostics** for both the sequential-IK
and SQP trajectories: per-knot end-effector position/orientation error and the
per-interval lateral-slip residual of the base, logged as Rerun time series
(`diagnostics/{ik,sqp}/...` on the `step` timeline) with a console summary of
the worst violations.
```

In the Project Layout tree, add under `src/`:

```
  diagnostics.rs    # pose-tracking error + nonholonomic slip residuals
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: describe trajectory diagnostics in README"
```

---

## Verification

- `cargo fmt && cargo check --all-targets && cargo test` — warning-clean, all green (35 pre-existing + 10 new tests).
- `cargo run --release` shows six `diagnostics/*` series in Rerun and the four-line console summary.
- The pre-existing known-failing expectation mismatch (`configs::tests::config_parses` vs `assets/config.yaml`, see AGENTS.md) is out of scope — do not touch it.
