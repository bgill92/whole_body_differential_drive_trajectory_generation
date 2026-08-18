# SQP Whole-Body Trajectory Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize the whole joint trajectory at once — Gauss-Newton SQP with the diff-drive no-lateral-slip constraint, joint velocity limits, and smoothness — warm-started from the existing sequential IK, solving one dense Clarabel QP per SQP iteration.

**Architecture:** New `src/trajectory.rs` owns the SQP loop and subproblem assembly over stacked knot configurations `Q = [q_0…q_{N-1}]` (dim = N·n ≈ 110 for the shipped config). `qp::solve` gains general inequality rows (`A_in·x ≤ b_in`) for velocity limits. Per-knot residual/Jacobian come from the existing `Kinematics`; EE tracking is a config switch: soft (Gauss-Newton cost) or hard (linearized equality rows). Spec: `docs/superpowers/specs/2026-08-14-sqp-trajectory-optimization-design.md`.

**Tech Stack:** Rust 2024, clarabel 0.11 (already a dep), nalgebra via `k` 0.32 (OLD API: `slice_mut`, `rows_mut`, `fixed_slice` — no `view_mut`).

**Branch:** create `sqp-trajectory` off `main` before Task 1; all commits land there.

**Cost/QP convention (used consistently below):** the QP is `min ½ΔᵀPΔ + qᵀΔ` with
- soft EE: `P += w_ee·J_kᵀJ_k` (block k,k), `q += −w_ee·J_kᵀr_k` (block k)
- smoothness, per interval k with current difference `d_k = q_{k+1} − q_k`: `P` gets `+w_s·I` at blocks (k,k) and (k+1,k+1), `−w_s·I` at (k,k+1) and (k+1,k); `q` gets `−w_s·d_k` at block k and `+w_s·d_k` at block k+1
- damping: `P += damping·I` (the config value IS the added coefficient — the spec's "λ²" is resolved as: config stores the already-squared value)

One deliberate spec correction: the spec's integration assertion (c) — "mean EE tracking error ≤ warm start's" — is unachievable by construction (warm start is per-knot unconstrained optimal; the nonholonomic constraint can only trade EE accuracy for feasibility). Task 4 asserts a bounded absolute error instead and patches the spec line.

---

## File Structure

- Modify: `src/qp.rs` — add `a_in`/`b_in` params + 2 tests; update 7 existing tests mechanically.
- Modify: `src/kinematics.rs` — `pose_error_twist` becomes `pub(crate)`; IK call site passes empty `a_in`.
- Modify: `src/configs.rs` — `EeTracking` enum, `TrajectoryConfig`, `Config.trajectory: Option<TrajectoryConfig>`; parse tests.
- Create: `src/trajectory.rs` — `BaseIndices`, `nonholonomic_linearization`, `sqp_step` assembly, `pub fn optimize`, unit + integration tests.
- Modify: `src/lib.rs` — `pub mod trajectory;`, re-export `TrajectoryConfig`.
- Modify: `src/main.rs` — run optimizer after the IK warm-start loop.
- Modify: `assets/config.yaml` — `trajectory:` section.

---

### Task 1: `qp::solve` general inequality rows

**Files:**
- Modify: `src/qp.rs`
- Modify: `src/kinematics.rs` (call site only)

- [ ] **Step 1: Create the branch**

```bash
git checkout main && git pull && git checkout -b sqp-trajectory
```

- [ ] **Step 2: Write the failing tests**

In `src/qp.rs` tests module, add a helper next to `no_eq`:

```rust
    fn no_in(n: usize) -> (DMatrix<f64>, DVector<f64>) {
        (DMatrix::zeros(0, n), DVector::zeros(0))
    }
```

Add two tests at the end of the module:

```rust
    // x0 + x1 ≤ 1 cuts off the unconstrained minimum [1, 2]. KKT: x = [1−μ, 2−μ]
    // with 3 − 2μ = 1, so μ = 1 and x = [0, 1].
    #[test]
    fn inequality_row_activates() {
        let (a_eq, b_eq) = no_eq(2);
        let (lb, ub) = free_bounds(2);
        let mut a_in = DMatrix::zeros(1, 2);
        a_in[(0, 0)] = 1.0;
        a_in[(0, 1)] = 1.0;
        let b_in = DVector::from_vec(vec![1.0]);
        let x = solve(
            &DMatrix::identity(2, 2),
            &DVector::from_vec(vec![-1.0, -2.0]),
            &a_eq,
            &b_eq,
            &a_in,
            &b_in,
            &lb,
            &ub,
        )
        .unwrap();
        assert!((x[0] - 0.0).abs() < 1e-6, "got {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-6, "got {}", x[1]);
    }

    // A slack inequality (x0 + x1 ≤ 10) must leave the optimum untouched.
    #[test]
    fn inactive_inequality_row_ignored() {
        let (a_eq, b_eq) = no_eq(2);
        let (lb, ub) = free_bounds(2);
        let mut a_in = DMatrix::zeros(1, 2);
        a_in[(0, 0)] = 1.0;
        a_in[(0, 1)] = 1.0;
        let b_in = DVector::from_vec(vec![10.0]);
        let x = solve(
            &DMatrix::identity(2, 2),
            &DVector::from_vec(vec![-1.0, -2.0]),
            &a_eq,
            &b_eq,
            &a_in,
            &b_in,
            &lb,
            &ub,
        )
        .unwrap();
        assert!((x[0] - 1.0).abs() < 1e-6, "got {}", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-6, "got {}", x[1]);
    }
```

Also update ALL 7 existing tests' `solve(...)` calls to insert `&a_in, &b_in` (from `let (a_in, b_in) = no_in(N);`) between `&b_eq` and `&lb` — same variable dimension as each test's `no_eq` call.

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test qp`
Expected: FAIL to compile — `solve` takes 6 arguments, tests pass 8.

- [ ] **Step 4: Extend `solve`**

In `src/qp.rs`, change the signature and doc comment:

```rust
/// Solve min ½xᵀPx + qᵀx s.t. A_eq·x = b_eq, A_in·x ≤ b_in, lb ≤ x ≤ ub.
///
/// `p` must be symmetric positive semidefinite. Infinite bounds are allowed
/// and simply contribute no constraint row. `a_eq` and `a_in` may have zero
/// rows.
pub(crate) fn solve(
    p: &k::nalgebra::DMatrix<f64>,
    q: &k::nalgebra::DVector<f64>,
    a_eq: &k::nalgebra::DMatrix<f64>,
    b_eq: &k::nalgebra::DVector<f64>,
    a_in: &k::nalgebra::DMatrix<f64>,
    b_in: &k::nalgebra::DVector<f64>,
    lb: &k::nalgebra::DVector<f64>,
    ub: &k::nalgebra::DVector<f64>,
) -> Result<k::nalgebra::DVector<f64>, String> {
```

In the body, replace the assembly between the `finite_lb` line and the `cones` array with:

```rust
    let m_in = a_in.nrows();
    let m_ineq = m_in + finite_ub.len() + finite_lb.len();

    let mut a = k::nalgebra::DMatrix::<f64>::zeros(m_eq + m_ineq, n);
    let mut b = k::nalgebra::DVector::<f64>::zeros(m_eq + m_ineq);
    a.slice_mut((0, 0), (m_eq, n)).copy_from(a_eq);
    b.rows_mut(0, m_eq).copy_from(b_eq);
    // General A_in·x ≤ b_in rows join the same nonnegative cone as the
    // finite-bound rows: A_in·x + s = b_in with s ≥ 0.
    a.slice_mut((m_eq, 0), (m_in, n)).copy_from(a_in);
    b.rows_mut(m_eq, m_in).copy_from(b_in);
    let mut row = m_eq + m_in;
    for &i in &finite_ub {
        a[(row, i)] = 1.0;
        b[row] = ub[i];
        row += 1;
    }
    for &i in &finite_lb {
        a[(row, i)] = -1.0;
        b[row] = -lb[i];
        row += 1;
    }
```

(The `cones` array is already `NonnegativeConeT(m_ineq)` — unchanged.)

In `src/kinematics.rs`, update the one `qp::solve` call inside `differential_ik`: add these two arguments between the equality rhs and the lower-bound argument:

```rust
            &k::nalgebra::DMatrix::zeros(0, n),
            &k::nalgebra::DVector::zeros(0),
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test`
Expected: 18 passed (16 prior + 2 new), 0 failed.

- [ ] **Step 6: Commit**

```bash
git add src/qp.rs src/kinematics.rs
git commit -m "feat: support general inequality rows in qp solver"
```

---

### Task 2: trajectory config

**Files:**
- Modify: `src/configs.rs`
- Modify: `src/lib.rs`
- Modify: `assets/config.yaml`

- [ ] **Step 1: Write the failing tests**

In the `tests` module of `src/configs.rs`, append to `CONFIG_YAML` (inside the string, after the `differential_ik` block):

```
trajectory:
  enabled: true
  dt: 0.5
  ee_tracking: soft
  ee_weight: 100.0
  smoothness_weight: 1.0
  max_joint_velocity: 1.0
  sqp_max_iterations: 20
  trust_region: 0.2
  convergence_step_norm: 1.0e-4
  damping: 1.0e-3
  base_joint_names: [world_base_link_planar_prismatic_x, world_base_link_planar_prismatic_y, world_base_link_planar_yaw]
```

Add a test:

```rust
    #[test]
    fn trajectory_config_parses() {
        let config: Config = serde_yaml_ng::from_str(CONFIG_YAML).unwrap();
        let trajectory = config.trajectory.expect("trajectory section missing");
        assert!(trajectory.enabled);
        assert!(matches!(trajectory.ee_tracking, EeTracking::Soft));
        assert_eq!(trajectory.base_joint_names.len(), 3);
        assert_eq!(trajectory.sqp_max_iterations, 20);

        // The section is optional: configs without it parse to None.
        let without = CONFIG_YAML.split("trajectory:").next().unwrap();
        let config: Config = serde_yaml_ng::from_str(without).unwrap();
        assert!(config.trajectory.is_none());

        // The enum rejects unknown modes rather than defaulting.
        let bad = CONFIG_YAML.replace("ee_tracking: soft", "ee_tracking: rigid");
        assert!(serde_yaml_ng::from_str::<Config>(&bad).is_err());
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test configs`
Expected: FAIL to compile — `trajectory` field and `EeTracking` not defined.

- [ ] **Step 3: Implement the config types**

In `src/configs.rs`, after `DifferentialIkConfig`:

```rust
/// End-effector path-following mode for the trajectory optimizer.
#[derive(serde::Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum EeTracking {
    /// Weighted quadratic pose-error cost per knot; always feasible.
    Soft,
    /// Linearized pose equality rows per knot; may be infeasible when the
    /// path conflicts with base kinematics.
    Hard,
}

/// Whole-trajectory SQP settings. See
/// docs/superpowers/specs/2026-08-14-sqp-trajectory-optimization-design.md.
#[derive(serde::Deserialize)]
pub struct TrajectoryConfig {
    pub enabled: bool,
    /// Seconds per knot interval (velocities are finite differences over dt).
    pub dt: f64,
    pub ee_tracking: EeTracking,
    /// Soft-mode EE cost weight; unused in hard mode.
    pub ee_weight: f64,
    pub smoothness_weight: f64,
    /// One scalar for all joints; the k crate does not expose URDF velocity
    /// limits, so per-joint limits are deferred until it does.
    pub max_joint_velocity: f64,
    pub sqp_max_iterations: usize,
    /// Per-variable bound on each SQP step, intersected with joint limits.
    pub trust_region: f64,
    /// Converged when the step infinity-norm drops below this.
    pub convergence_step_norm: f64,
    /// Coefficient added to the diagonal of P; keeps P positive definite.
    pub damping: f64,
    /// Planar base joints in x, y, yaw order; validated against the chain.
    pub base_joint_names: Vec<String>,
}
```

Add to `Config`:

```rust
    /// Optional whole-trajectory optimization stage; absent = IK only.
    #[serde(default)]
    pub trajectory: Option<TrajectoryConfig>,
```

In `src/lib.rs`, extend the configs re-export:

```rust
pub use crate::configs::{Config, DifferentialIkConfig, EqualityConstraint, EeTracking, TrajectoryConfig};
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test`
Expected: 19 passed, 0 failed.

- [ ] **Step 5: Add the section to `assets/config.yaml`**

Append at the end of the file:

```yaml
# Whole-trajectory SQP (Gauss-Newton over Clarabel QPs), warm-started from the
# sequential IK above. Enforces diff-drive no-lateral-slip, joint velocity
# limits, and smoothness. ee_tracking: soft (weighted cost) | hard (exact,
# may be infeasible).
trajectory:
  enabled: true
  dt: 0.5
  ee_tracking: soft
  ee_weight: 100.0
  smoothness_weight: 1.0
  max_joint_velocity: 1.0
  sqp_max_iterations: 20
  trust_region: 0.2
  convergence_step_norm: 1.0e-4
  damping: 1.0e-3
  base_joint_names: [world_base_link_planar_prismatic_x, world_base_link_planar_prismatic_y, world_base_link_planar_yaw]
```

- [ ] **Step 6: Commit**

```bash
git add src/configs.rs src/lib.rs assets/config.yaml
git commit -m "feat: add trajectory optimization config section"
```

---

### Task 3: nonholonomic constraint linearization

**Files:**
- Create: `src/trajectory.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Create the module with the failing test**

In `src/lib.rs`, after `mod qp;`:

```rust
pub mod trajectory;
```

Create `src/trajectory.rs`:

```rust
//! Whole-body trajectory optimization: Gauss-Newton SQP over the stacked knot
//! configurations, one dense Clarabel QP per iteration. See
//! docs/superpowers/specs/2026-08-14-sqp-trajectory-optimization-design.md.

/// Chain indices of the planar base joints.
struct BaseIndices {
    x: usize,
    y: usize,
    yaw: usize,
}

/// No-lateral-slip constraint for one knot interval, linearized at the current
/// configurations: c = sin(θ̄)·Δx − cos(θ̄)·Δy with θ̄ the midpoint heading.
///
/// Returns the residual and the six nonzero partials as
/// (index into the concatenated [q_k ‖ q_k1], value); all other partials are
/// exactly zero because c involves only the base coordinates.
fn nonholonomic_linearization(
    q_k: &[f64],
    q_k1: &[f64],
    base: &BaseIndices,
) -> (f64, [(usize, f64); 6]) {
    let n = q_k.len();
    let dx = q_k1[base.x] - q_k[base.x];
    let dy = q_k1[base.y] - q_k[base.y];
    let mid_yaw = 0.5 * (q_k[base.yaw] + q_k1[base.yaw]);
    let (sin_mid, cos_mid) = mid_yaw.sin_cos();

    let residual = sin_mid * dx - cos_mid * dy;
    // d/dθ̄ (sin θ̄·Δx − cos θ̄·Δy) = cos θ̄·Δx + sin θ̄·Δy, and ∂θ̄/∂θ_k =
    // ∂θ̄/∂θ_{k+1} = ½, so both yaw partials share this value.
    let dyaw = 0.5 * (cos_mid * dx + sin_mid * dy);
    let partials = [
        (base.x, -sin_mid),
        (base.y, cos_mid),
        (base.yaw, dyaw),
        (n + base.x, sin_mid),
        (n + base.y, -cos_mid),
        (n + base.yaw, dyaw),
    ];
    (residual, partials)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Central finite differences over every entry of [q_k ‖ q_k1] must match
    // the analytic partials; entries not in the partial list must have zero
    // finite-difference gradient.
    #[test]
    fn nonholonomic_linearization_matches_finite_differences() {
        let base = BaseIndices { x: 0, y: 1, yaw: 2 };
        let q_k = [0.3, -0.2, 0.7, 0.1];
        let q_k1 = [0.6, 0.1, 1.1, -0.4];

        let (_, partials) = nonholonomic_linearization(&q_k, &q_k1, &base);
        let mut analytic = [0.0; 8];
        for (index, value) in partials {
            analytic[index] = value;
        }

        let eps = 1e-6;
        for i in 0..8 {
            let mut plus = [q_k, q_k1].concat();
            let mut minus = plus.clone();
            plus[i] += eps;
            minus[i] -= eps;
            let c_plus = nonholonomic_linearization(&plus[..4], &plus[4..], &base).0;
            let c_minus = nonholonomic_linearization(&minus[..4], &minus[4..], &base).0;
            let fd = (c_plus - c_minus) / (2.0 * eps);
            assert!(
                (fd - analytic[i]).abs() < 1e-6,
                "partial {i}: finite difference {fd} vs analytic {}",
                analytic[i]
            );
        }
    }

    // A pure-forward motion (heading exactly along the displacement) must
    // satisfy the constraint; a pure-sideways one must violate it maximally.
    #[test]
    fn nonholonomic_residual_semantics() {
        let base = BaseIndices { x: 0, y: 1, yaw: 2 };
        // Heading 0, moving +x: no slip.
        let (c, _) = nonholonomic_linearization(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0], &base);
        assert!(c.abs() < 1e-12, "forward motion flagged as slip: {c}");
        // Heading 0, moving +y: full lateral slip of magnitude 1.
        let (c, _) = nonholonomic_linearization(&[0.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &base);
        assert!((c.abs() - 1.0).abs() < 1e-12, "lateral motion residual: {c}");
    }
}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cargo test trajectory`
Expected: 2 passed. (Written-then-run rather than red-green here: the function and its verification test form one unit; the finite-difference check IS the falsifier. `cargo build` will show `dead_code` warnings for the unused items — expected until Task 4 wires them in.)

- [ ] **Step 3: Commit**

```bash
git add src/lib.rs src/trajectory.rs
git commit -m "feat: add nonholonomic constraint linearization"
```

---

### Task 4: SQP loop, subproblem assembly, integration tests

**Files:**
- Modify: `src/trajectory.rs`
- Modify: `src/kinematics.rs` (visibility only)
- Modify: `docs/superpowers/specs/2026-08-14-sqp-trajectory-optimization-design.md` (one line)

- [ ] **Step 1: Expose `pose_error_twist`**

In `src/kinematics.rs`, change the signature of the private helper (keep body and doc comment):

```rust
pub(crate) fn pose_error_twist(
```

- [ ] **Step 2: Write the failing integration tests**

Append to the `tests` module in `src/trajectory.rs` (after the existing two tests):

```rust
    use crate::configs::{DifferentialIkConfig, EeTracking, TrajectoryConfig};
    use crate::kinematics::{Kinematics, differential_ik, pose_error_twist};
    use k::nalgebra::{Isometry3, UnitQuaternion, Vector3};

    fn base_names() -> Vec<String> {
        vec![
            "world_base_link_planar_prismatic_x".to_string(),
            "world_base_link_planar_prismatic_y".to_string(),
            "world_base_link_planar_yaw".to_string(),
        ]
    }

    fn test_config(ee_tracking: EeTracking) -> TrajectoryConfig {
        TrajectoryConfig {
            enabled: true,
            dt: 0.5,
            ee_tracking,
            ee_weight: 100.0,
            smoothness_weight: 1.0,
            max_joint_velocity: 2.0,
            sqp_max_iterations: 30,
            trust_region: 0.2,
            convergence_step_norm: 1e-5,
            damping: 1e-3,
            base_joint_names: base_names(),
        }
    }

    // Downward-facing EE poses along +x, matching the shipped config's
    // orientation convention.
    fn straight_goals(count: usize, spacing: f64) -> Vec<Isometry3<f64>> {
        (0..count)
            .map(|i| {
                Isometry3::from_parts(
                    Vector3::new(0.8 + spacing * i as f64, 0.5, 0.9).into(),
                    UnitQuaternion::from_euler_angles(std::f64::consts::PI, 0.0, 0.0),
                )
            })
            .collect()
    }

    fn test_kinematics() -> Kinematics {
        Kinematics::build("assets/rox_diff_ur5e.urdf", "ur5ewrist_3_joint")
            .expect("failed to build kinematics")
    }

    // Sequential IK, exactly like main's warm-start loop.
    fn warm_start(kinematics: &mut Kinematics, goals: &[Isometry3<f64>]) -> Vec<Vec<f64>> {
        let ik = DifferentialIkConfig {
            num_steps: 50,
            damping_factor: 0.5,
            convergence_threshold: 0.01,
            equality_constraints: vec![],
        };
        goals
            .iter()
            .map(|goal| {
                differential_ik(goal, kinematics, &ik)
                    .expect("warm-start IK failed")
                    .pop()
                    .expect("trajectory is never empty")
            })
            .collect()
    }

    fn base_indices(kinematics: &Kinematics) -> BaseIndices {
        let names = kinematics.joint_names();
        let find = |name: &str| names.iter().position(|n| n == name).unwrap();
        BaseIndices {
            x: find("world_base_link_planar_prismatic_x"),
            y: find("world_base_link_planar_prismatic_y"),
            yaw: find("world_base_link_planar_yaw"),
        }
    }

    #[test]
    fn soft_mode_removes_lateral_slip_within_limits() {
        let mut kinematics = test_kinematics();
        let goals = straight_goals(4, 0.15);
        let start = warm_start(&mut kinematics, &goals);
        let config = test_config(EeTracking::Soft);

        let optimized = optimize(&goals, &mut kinematics, &config, &start).unwrap();
        assert_eq!(optimized.len(), goals.len());

        let base = base_indices(&kinematics);
        let (lower, upper) = kinematics.joint_limits();
        let n = lower.len();
        let mut ee_error_sum = 0.0;
        for k in 0..optimized.len() {
            // (b) Joint limits and velocity limits hold at every knot/interval.
            for j in 0..n {
                assert!(
                    optimized[k][j] >= lower[j] - 1e-9 && optimized[k][j] <= upper[j] + 1e-9,
                    "joint {j} out of limits at knot {k}"
                );
            }
            if k + 1 < optimized.len() {
                for j in 0..n {
                    let velocity = (optimized[k + 1][j] - optimized[k][j]).abs() / config.dt;
                    assert!(
                        velocity <= config.max_joint_velocity + 1e-6,
                        "joint {j} velocity {velocity} exceeds limit at interval {k}"
                    );
                }
                // (a) No lateral slip after convergence.
                let (residual, _) =
                    nonholonomic_linearization(&optimized[k], &optimized[k + 1], &base);
                assert!(
                    residual.abs() < 1e-6,
                    "lateral slip {residual} at interval {k}"
                );
            }
            // (c) EE tracking stays reasonable (it cannot beat the
            // unconstrained warm start; it must not collapse either).
            kinematics.set_positions(&optimized[k]);
            let twist =
                pose_error_twist(&goals[k].to_homogeneous(), &kinematics.end_pose());
            ee_error_sum += twist.norm();
        }
        let mean_ee_error = ee_error_sum / optimized.len() as f64;
        assert!(
            mean_ee_error < 0.25,
            "mean EE tracking error too large: {mean_ee_error}"
        );
    }

    // (d) Hard mode on a reachable straight path solves and also kills slip.
    #[test]
    fn hard_mode_solves_reachable_path() {
        let mut kinematics = test_kinematics();
        let goals = straight_goals(3, 0.1);
        let start = warm_start(&mut kinematics, &goals);
        let config = test_config(EeTracking::Hard);

        let optimized = optimize(&goals, &mut kinematics, &config, &start).unwrap();
        let base = base_indices(&kinematics);
        for k in 0..optimized.len() - 1 {
            let (residual, _) =
                nonholonomic_linearization(&optimized[k], &optimized[k + 1], &base);
            assert!(residual.abs() < 1e-6, "slip {residual} at interval {k}");
        }
    }

    // (d) Hard EE rows + a velocity budget too small to reach the next goal
    // must surface as an infeasibility error, not a panic or silent result.
    #[test]
    fn hard_mode_infeasible_velocity_budget_errors() {
        let mut kinematics = test_kinematics();
        let goals = straight_goals(3, 0.5);
        let start = warm_start(&mut kinematics, &goals);
        let mut config = test_config(EeTracking::Hard);
        // Deliberately conflicting: exact EE poses per knot, but joints may
        // move at most 0.001·0.5 per interval while consecutive warm-start
        // knots differ far more than that.
        config.max_joint_velocity = 0.001;

        // Force intervals to actually need motion: perturb the warm start so
        // knot 1 and 2 start from knot 0's configuration.
        let perturbed: Vec<Vec<f64>> = vec![start[0].clone(); 3];
        let result = optimize(&goals, &mut kinematics, &config, &perturbed);
        assert!(result.is_err(), "expected infeasibility, got Ok");
        let message = result.unwrap_err();
        assert!(
            message.contains("qp_not_solved"),
            "unexpected error text: {message}"
        );
    }

    #[test]
    fn validation_errors() {
        let mut kinematics = test_kinematics();
        let goals = straight_goals(3, 0.1);
        let start = warm_start(&mut kinematics, &goals);

        // Unknown base joint name.
        let mut config = test_config(EeTracking::Soft);
        config.base_joint_names[0] = "no_such_joint".to_string();
        assert!(optimize(&goals, &mut kinematics, &config, &start).is_err());

        // Warm start length mismatch.
        let config = test_config(EeTracking::Soft);
        assert!(optimize(&goals, &mut kinematics, &config, &start[..2]).is_err());

        // Fewer than two knots.
        assert!(optimize(&goals[..1], &mut kinematics, &config, &start[..1]).is_err());

        // Non-positive dt.
        let mut config = test_config(EeTracking::Soft);
        config.dt = 0.0;
        assert!(optimize(&goals, &mut kinematics, &config, &start).is_err());
    }
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test trajectory`
Expected: FAIL to compile — `optimize` not found.

- [ ] **Step 4: Implement assembly and the SQP loop**

Add to `src/trajectory.rs` above the tests module (imports go at the top of the file):

```rust
use crate::configs::{EeTracking, TrajectoryConfig};
use crate::kinematics::{Kinematics, pose_error_twist};
use crate::qp;

type DMat = k::nalgebra::DMatrix<f64>;
type DVec = k::nalgebra::DVector<f64>;

/// Elementwise block accumulation; avoids relying on slice AddAssign support
/// in the old nalgebra shipped through the k crate.
fn add_block(p: &mut DMat, row0: usize, col0: usize, block: &DMat) {
    for r in 0..block.nrows() {
        for c in 0..block.ncols() {
            p[(row0 + r, col0 + c)] += block[(r, c)];
        }
    }
}

fn resolve_base_indices(
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

/// Assemble and solve one SQP subproblem at the current iterate; returns the
/// stacked step ΔQ. Cost convention: min ½ΔᵀPΔ + qᵀΔ with P = w_ee·JᵀJ blocks
/// (soft mode) + smoothness difference operator + damping·I.
fn sqp_step(
    knots: &[DVec],
    goal_poses: &[k::Isometry3<f64>],
    kinematics: &mut Kinematics,
    config: &TrajectoryConfig,
    base: &BaseIndices,
    lower: &DVec,
    upper: &DVec,
) -> Result<DVec, String> {
    let num_knots = knots.len();
    let n = lower.len();
    let dim = num_knots * n;
    let num_intervals = num_knots - 1;

    // Per-knot pose error twists and Jacobians at the current iterate.
    let mut residuals = Vec::with_capacity(num_knots);
    let mut jacobians = Vec::with_capacity(num_knots);
    for (knot, goal) in knots.iter().zip(goal_poses) {
        kinematics.set_positions(knot.as_slice());
        residuals.push(pose_error_twist(&goal.to_homogeneous(), &kinematics.end_pose()));
        jacobians.push(kinematics.jacobian());
    }

    let mut p = DMat::zeros(dim, dim);
    let mut q_lin = DVec::zeros(dim);

    // EE tracking cost (soft mode only; hard mode uses equality rows below).
    if matches!(config.ee_tracking, EeTracking::Soft) {
        for k in 0..num_knots {
            let jt = jacobians[k].transpose();
            add_block(&mut p, k * n, k * n, &(config.ee_weight * (&jt * &jacobians[k])));
            let residual = DVec::from_vec(residuals[k].as_slice().to_vec());
            let gradient = config.ee_weight * (&jt * residual);
            for j in 0..n {
                q_lin[k * n + j] -= gradient[j];
            }
        }
    }

    // Smoothness on consecutive differences d_k = q_{k+1} − q_k: quadratic in
    // ΔQ through the first-difference operator, linear term from current d_k.
    let identity = DMat::identity(n, n);
    for k in 0..num_intervals {
        let w = config.smoothness_weight;
        add_block(&mut p, k * n, k * n, &(w * &identity));
        add_block(&mut p, (k + 1) * n, (k + 1) * n, &(w * &identity));
        add_block(&mut p, k * n, (k + 1) * n, &(-w * &identity));
        add_block(&mut p, (k + 1) * n, k * n, &(-w * &identity));
        for j in 0..n {
            let difference = knots[k + 1][j] - knots[k][j];
            q_lin[k * n + j] -= w * difference;
            q_lin[(k + 1) * n + j] += w * difference;
        }
    }

    // Damping keeps P positive definite even where J loses rank.
    for i in 0..dim {
        p[(i, i)] += config.damping;
    }

    // Equalities: one linearized no-slip row per interval, plus (hard mode)
    // six linearized EE pose rows per knot.
    let hard = matches!(config.ee_tracking, EeTracking::Hard);
    let m_eq = num_intervals + if hard { 6 * num_knots } else { 0 };
    let mut a_eq = DMat::zeros(m_eq, dim);
    let mut b_eq = DVec::zeros(m_eq);
    for k in 0..num_intervals {
        let (residual, partials) =
            nonholonomic_linearization(knots[k].as_slice(), knots[k + 1].as_slice(), base);
        for (offset, value) in partials {
            a_eq[(k, k * n + offset)] = value;
        }
        b_eq[k] = -residual;
    }
    if hard {
        for k in 0..num_knots {
            let row0 = num_intervals + 6 * k;
            a_eq
                .slice_mut((row0, k * n), (6, n))
                .copy_from(&jacobians[k]);
            for axis in 0..6 {
                b_eq[row0 + axis] = residuals[k][axis];
            }
        }
    }

    // Velocity limits: |d_k + (Δ_{k+1} − Δ_k)| ≤ v_max·dt, two rows per joint
    // per interval over ΔQ.
    let budget = config.max_joint_velocity * config.dt;
    let m_in = 2 * n * num_intervals;
    let mut a_in = DMat::zeros(m_in, dim);
    let mut b_in = DVec::zeros(m_in);
    for k in 0..num_intervals {
        for j in 0..n {
            let difference = knots[k + 1][j] - knots[k][j];
            let upper_row = 2 * (k * n + j);
            a_in[(upper_row, (k + 1) * n + j)] = 1.0;
            a_in[(upper_row, k * n + j)] = -1.0;
            b_in[upper_row] = budget - difference;
            let lower_row = upper_row + 1;
            a_in[(lower_row, (k + 1) * n + j)] = -1.0;
            a_in[(lower_row, k * n + j)] = 1.0;
            b_in[lower_row] = budget + difference;
        }
    }

    // Box: joint limits re-anchored at the iterate, intersected with the
    // trust region.
    let mut lb = DVec::zeros(dim);
    let mut ub = DVec::zeros(dim);
    for k in 0..num_knots {
        for j in 0..n {
            lb[k * n + j] = (lower[j] - knots[k][j]).max(-config.trust_region);
            ub[k * n + j] = (upper[j] - knots[k][j]).min(config.trust_region);
        }
    }

    qp::solve(&p, &q_lin, &a_eq, &b_eq, &a_in, &b_in, &lb, &ub)
}

/// Optimize the whole trajectory with Gauss-Newton SQP. `warm_start` is one
/// joint configuration per goal pose (from the sequential IK); the return
/// value has the same shape and feeds `log_trajectory` directly.
///
/// Non-convergence within `sqp_max_iterations` returns the last iterate (like
/// the IK loop); infeasibility (possible in hard mode) is an error.
pub fn optimize(
    goal_poses: &[k::Isometry3<f64>],
    kinematics: &mut Kinematics,
    config: &TrajectoryConfig,
    warm_start: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, String> {
    let joint_names = kinematics.joint_names();
    let n = joint_names.len();
    if goal_poses.len() < 2 {
        return Err("trajectory optimization needs at least 2 knots".to_string());
    }
    if warm_start.len() != goal_poses.len() {
        return Err(format!(
            "warm start has {} knots but the path has {}",
            warm_start.len(),
            goal_poses.len()
        ));
    }
    if warm_start.iter().any(|knot| knot.len() != n) {
        return Err("warm-start knot dimension does not match the chain".to_string());
    }
    if config.dt <= 0.0 || config.trust_region <= 0.0 || config.damping <= 0.0 {
        return Err("trajectory dt, trust_region, and damping must be > 0".to_string());
    }
    let base = resolve_base_indices(&joint_names, &config.base_joint_names)?;
    let (lower, upper) = kinematics.joint_limits();

    let mut knots: Vec<DVec> = warm_start
        .iter()
        .map(|knot| DVec::from_vec(knot.clone()))
        .collect();

    for iteration in 0..config.sqp_max_iterations {
        let step = sqp_step(
            &knots, goal_poses, kinematics, config, &base, &lower, &upper,
        )
        .map_err(|e| {
            let hint = if matches!(config.ee_tracking, EeTracking::Hard) {
                "; ee_tracking: hard can conflict with base kinematics — try soft"
            } else {
                ""
            };
            format!("sqp iteration {iteration}: {e}{hint}")
        })?;

        let mut step_norm: f64 = 0.0;
        for k in 0..knots.len() {
            for j in 0..n {
                let delta = step[k * n + j];
                step_norm = step_norm.max(delta.abs());
                // Same clamp rationale as the IK loop: interior-point steps
                // are feasible to tolerance, hard limits must hold exactly.
                knots[k][j] = (knots[k][j] + delta).clamp(lower[j], upper[j]);
            }
        }
        if step_norm < config.convergence_step_norm {
            break;
        }
    }

    Ok(knots.into_iter().map(|knot| knot.as_slice().to_vec()).collect())
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test`
Expected: 25 passed (19 prior + 2 Task 3 + 4 new), 0 failed. Integration tests take a few seconds (real URDF FK per iteration). If `soft_mode_removes_lateral_slip_within_limits` fails only the 1e-6 slip assertion, the SQP has not fully converged: raise `sqp_max_iterations` to 50 in `test_config` rather than loosening the assertion (converged fixed points force the slip residual to zero — see spec).

- [ ] **Step 6: Patch the spec's assertion (c)**

In `docs/superpowers/specs/2026-08-14-sqp-trajectory-optimization-design.md`, replace the line:

```
  (c) mean EE tracking error ≤ warm start's (soft mode),
```

with:

```
  (c) mean EE tracking error stays bounded (< 0.25 combined twist norm, soft
      mode) — it cannot beat the unconstrained warm start, which the original
      draft of this line wrongly claimed,
```

- [ ] **Step 7: Clippy, format, commit**

Run: `cargo clippy --all-targets && cargo fmt && cargo test`
Expected: no new warnings; 25 passed.

```bash
git add src/trajectory.rs src/kinematics.rs docs/superpowers/specs/2026-08-14-sqp-trajectory-optimization-design.md
git commit -m "feat: add Gauss-Newton SQP whole-body trajectory optimizer

One dense Clarabel QP per iteration over the stacked knots: soft or
hard EE tracking, linearized diff-drive no-slip equalities, joint
velocity inequality rows, smoothness cost, trust-region box."
```

---

### Task 5: wire into main, end-to-end, push

**Files:**
- Modify: `src/main.rs`

- [ ] **Step 1: Run the optimizer after the IK warm start**

In `src/main.rs`, extend the wbdd import:

```rust
use wbdd::{Config, EqualityConstraint, Kinematics, differential_ik, trajectory};
```

Insert between the solve loop's closing brace and the `log_trajectory` call:

```rust
    // Whole-trajectory SQP pass, warm-started from the sequential IK result.
    // Skipped in first-pose debug mode: the optimizer needs every knot.
    if let Some(trajectory_config) = &config.trajectory {
        if trajectory_config.enabled {
            if config.path.solve_first_pose_only {
                eprintln!("trajectory optimization skipped: path.solve_first_pose_only is set");
            } else {
                joint_positions = trajectory::optimize(
                    &goal_poses,
                    &mut kinematics,
                    trajectory_config,
                    &joint_positions,
                )?;
            }
        }
    }
```

- [ ] **Step 2: Full test suite + end-to-end run**

Run: `cargo test`
Expected: 25 passed.

Run: `cargo run --release`
Expected: exit 0, rerun viewer shows the optimized trajectory; base heading now stays tangent to its own motion (no sideways sliding). If any iteration errors with `qp_not_solved`, check the config was not switched to `ee_tracking: hard` before debugging the solver.

- [ ] **Step 3: Commit and push**

```bash
git add src/main.rs
git commit -m "feat: run trajectory optimization after IK warm start"
git push -u origin sqp-trajectory
```

---

## Self-review notes

- Spec coverage: soft/hard switch (Tasks 2, 4), fixed dt + FD velocities (Task 4 velocity rows), nonholonomic-on-q with FD-verified linearization (Task 3), velocity limits + smoothness (Task 4), warm start from sequential IK (Tasks 4 test helper, 5 wiring), `qp::solve` inequality extension (Task 1), config section + validation (Tasks 2, 4), error handling incl. hard-mode hint and non-convergence-returns-last-iterate (Task 4), out-of-scope list untouched. Spec assertion (c) corrected with the change recorded in both plan and spec (Task 4 Step 6).
- Types consistent: `optimize(&[Isometry3], &mut Kinematics, &TrajectoryConfig, &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String>` matches the Task 5 call site and every test; `qp::solve` 8-arg order (p, q, a_eq, b_eq, a_in, b_in, lb, ub) identical in Task 1 tests, Task 1 IK call site, and Task 4 `sqp_step`; `BaseIndices`/`nonholonomic_linearization` defined in Task 3 exactly as consumed in Task 4.
- Placeholders: none; every step carries complete code or an exact command with expected output.
- Known risks localized: slice `AddAssign` avoided entirely via `add_block`; test count expectations stated per task; the one numerically-sensitive assertion (slip < 1e-6) has an explicit remedy that tightens convergence instead of loosening the test.
