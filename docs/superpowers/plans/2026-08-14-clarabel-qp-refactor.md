# Clarabel QP Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-rolled active-set solver (`src/active_set.rs`) with the pure-Rust Clarabel interior-point QP solver, behind a small generic QP module that the future SQP/Gauss-Newton trajectory optimizer will reuse as its subproblem solver.

**Architecture:** A new `src/qp.rs` module solves the generic convex QP `min ½xᵀPx + qᵀx s.t. A_eq·x = b_eq, lb ≤ x ≤ ub` via Clarabel (equalities → `ZeroConeT`, finite box rows → `NonnegativeConeT`). `differential_ik` in `src/kinematics.rs` builds the damped Gauss-Newton QP per step (`P = JᵀJ + λ²I`, `q = −Jᵀν`) and calls `qp::solve`. `src/active_set.rs` is deleted. The math is equivalent: the old unconstrained DLS `dq = Jᵀ(JJᵀ+λ²I)⁻¹ν` and the QP minimizer `dq = (JᵀJ+λ²I)⁻¹Jᵀν` are the same by the push-through identity, and the old active-set KKT loop is exactly what a QP solver does internally.

**Tech Stack:** Rust 2024, `clarabel = "0.11"` (pure Rust, Apache-2.0, no C toolchain — see `docs/qp-sqp-solver-crates.md`), nalgebra via `k` 0.32 (old nalgebra API: `slice_mut`, not `view_mut`).

**Scope note:** The SQP/Gauss-Newton trajectory optimizer is a separate future plan. This plan's contribution to it is the `qp::solve` seam: general `P`/`q`/`A_eq`/`b_eq`/box inputs, not IK-specific types. When trajectory work starts, write a new plan that adds general inequality rows and (if profiling demands) sparse assembly; also reevaluate POUNCE per the research doc.

---

## File Structure

- Create: `src/qp.rs` — dense→CSC conversion + generic Clarabel QP wrapper + unit tests. One responsibility: solve one QP.
- Modify: `src/kinematics.rs` — build per-step QP data, call `qp::solve`; keep name/limit validation of config constraints. Existing tests unchanged (they are the behavioral safety net).
- Delete: `src/active_set.rs` — fully replaced.
- Modify: `src/lib.rs` — swap module declarations.
- Modify: `Cargo.toml` — add clarabel.

API-verification note (from research doc, unverified offline): in Clarabel 0.10+ `DefaultSolver::new` returns `Result`; the code below assumes that. If 0.11.1 docs.rs shows it returning `Self` directly, drop the `.map_err(...)?`. Same check for whether `P` must be upper-triangular-only: docs.rs examples pass full symmetric `P`; if the solver rejects it, pass only the upper triangle in `to_csc` (`keep r <= c` for `P`). Check https://docs.rs/clarabel/0.11.1 at Task 1 Step 3 before writing `solve`.

---

### Task 1: `qp` module — generic Clarabel wrapper

**Files:**
- Modify: `Cargo.toml`
- Modify: `src/lib.rs`
- Create: `src/qp.rs` (tests inline in the same file, matching repo convention)

- [ ] **Step 1: Add the dependency and verify it builds**

In `Cargo.toml` under `[dependencies]` add:

```toml
clarabel = "0.11"
```

Run: `cargo build`
Expected: compiles cleanly (clarabel is pure Rust — no cmake/C errors possible).

- [ ] **Step 2: Declare the module and write the failing tests**

In `src/lib.rs` add after `mod kinematics;`:

```rust
mod qp;
```

Create `src/qp.rs` containing only the doc comment and tests for now:

```rust
//! Generic dense convex QP solved with Clarabel:
//!   min ½ xᵀPx + qᵀx   s.t.   A_eq·x = b_eq,   lb ≤ x ≤ ub
//! Dense inputs are fine at IK size (n ≈ 10). This is also the seam the
//! future SQP/Gauss-Newton trajectory optimizer hands its subproblems to.

#[cfg(test)]
mod tests {
    use super::*;
    use k::nalgebra::{DMatrix, DVector};

    fn no_eq(n: usize) -> (DMatrix<f64>, DVector<f64>) {
        (DMatrix::zeros(0, n), DVector::zeros(0))
    }

    fn free_bounds(n: usize) -> (DVector<f64>, DVector<f64>) {
        (
            DVector::from_element(n, f64::NEG_INFINITY),
            DVector::from_element(n, f64::INFINITY),
        )
    }

    // min ½xᵀIx - [1,2]ᵀx  →  x = [1, 2].
    #[test]
    fn unconstrained_minimum() {
        let (a_eq, b_eq) = no_eq(2);
        let (lb, ub) = free_bounds(2);
        let x = solve(
            &DMatrix::identity(2, 2),
            &DVector::from_vec(vec![-1.0, -2.0]),
            &a_eq,
            &b_eq,
            &lb,
            &ub,
        )
        .unwrap();
        assert!((x[0] - 1.0).abs() < 1e-6, "got {}", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-6, "got {}", x[1]);
    }

    // Same objective, but x0 pinned to 0.3 by an equality row.
    #[test]
    fn equality_row_held() {
        let mut a_eq = DMatrix::zeros(1, 2);
        a_eq[(0, 0)] = 1.0;
        let b_eq = DVector::from_vec(vec![0.3]);
        let (lb, ub) = free_bounds(2);
        let x = solve(
            &DMatrix::identity(2, 2),
            &DVector::from_vec(vec![-1.0, -2.0]),
            &a_eq,
            &b_eq,
            &lb,
            &ub,
        )
        .unwrap();
        assert!((x[0] - 0.3).abs() < 1e-6, "got {}", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-6, "got {}", x[1]);
    }

    // Unconstrained minimum is x1 = 2; upper bound 0.5 must clamp it.
    #[test]
    fn upper_bound_activates() {
        let (a_eq, b_eq) = no_eq(2);
        let (lb, mut ub) = free_bounds(2);
        ub[1] = 0.5;
        let x = solve(
            &DMatrix::identity(2, 2),
            &DVector::from_vec(vec![-1.0, -2.0]),
            &a_eq,
            &b_eq,
            &lb,
            &ub,
        )
        .unwrap();
        assert!((x[1] - 0.5).abs() < 1e-6, "got {}", x[1]);
        assert!((x[0] - 1.0).abs() < 1e-6, "got {}", x[0]);
    }

    // Mirror of the above for the lower bound.
    #[test]
    fn lower_bound_activates() {
        let (a_eq, b_eq) = no_eq(2);
        let (mut lb, ub) = free_bounds(2);
        lb[1] = 3.0;
        let x = solve(
            &DMatrix::identity(2, 2),
            &DVector::from_vec(vec![-1.0, -2.0]),
            &a_eq,
            &b_eq,
            &lb,
            &ub,
        )
        .unwrap();
        assert!((x[1] - 3.0).abs() < 1e-6, "got {}", x[1]);
    }

    // Contradictory equalities: x0 = 0 and x0 = 1. Must error, not panic.
    #[test]
    fn infeasible_problem_errors() {
        let mut a_eq = DMatrix::zeros(2, 1);
        a_eq[(0, 0)] = 1.0;
        a_eq[(1, 0)] = 1.0;
        let b_eq = DVector::from_vec(vec![0.0, 1.0]);
        let (lb, ub) = free_bounds(1);
        let result = solve(
            &DMatrix::identity(1, 1),
            &DVector::zeros(1),
            &a_eq,
            &b_eq,
            &lb,
            &ub,
        );
        assert!(result.is_err());
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test qp`
Expected: FAIL to compile — `solve` not found. (Also open https://docs.rs/clarabel/0.11.1 now and confirm the two API-verification notes above.)

- [ ] **Step 4: Implement `to_csc` and `solve`**

Insert above the `tests` module in `src/qp.rs`:

```rust
use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus, SupportedConeT,
};

/// Dense column-major nalgebra matrix → Clarabel CSC, dropping exact zeros.
fn to_csc(dense: &k::nalgebra::DMatrix<f64>) -> CscMatrix<f64> {
    let (nrows, ncols) = dense.shape();
    let mut colptr = Vec::with_capacity(ncols + 1);
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    colptr.push(0);
    for c in 0..ncols {
        for r in 0..nrows {
            let v = dense[(r, c)];
            if v != 0.0 {
                rowval.push(r);
                nzval.push(v);
            }
        }
        colptr.push(rowval.len());
    }
    CscMatrix::new(nrows, ncols, colptr, rowval, nzval)
}

/// Solve min ½xᵀPx + qᵀx s.t. A_eq·x = b_eq, lb ≤ x ≤ ub.
///
/// `p` must be symmetric positive semidefinite. Infinite bounds are allowed
/// and simply contribute no constraint row. `a_eq` may have zero rows.
pub(crate) fn solve(
    p: &k::nalgebra::DMatrix<f64>,
    q: &k::nalgebra::DVector<f64>,
    a_eq: &k::nalgebra::DMatrix<f64>,
    b_eq: &k::nalgebra::DVector<f64>,
    lb: &k::nalgebra::DVector<f64>,
    ub: &k::nalgebra::DVector<f64>,
) -> Result<k::nalgebra::DVector<f64>, String> {
    let n = q.len();
    let m_eq = a_eq.nrows();

    // Finite box bounds become one-sided rows in the nonnegative cone:
    // x_i ≤ ub_i  →  (+e_i)·x + s = ub_i ;  x_i ≥ lb_i  →  (−e_i)·x + s = −lb_i.
    let finite_ub: Vec<usize> = (0..n).filter(|&i| ub[i].is_finite()).collect();
    let finite_lb: Vec<usize> = (0..n).filter(|&i| lb[i].is_finite()).collect();
    let m_ineq = finite_ub.len() + finite_lb.len();

    let mut a = k::nalgebra::DMatrix::<f64>::zeros(m_eq + m_ineq, n);
    let mut b = k::nalgebra::DVector::<f64>::zeros(m_eq + m_ineq);
    a.slice_mut((0, 0), (m_eq, n)).copy_from(a_eq);
    b.rows_mut(0, m_eq).copy_from(b_eq);
    let mut row = m_eq;
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

    let cones = [
        SupportedConeT::ZeroConeT(m_eq),
        SupportedConeT::NonnegativeConeT(m_ineq),
    ];
    let settings = DefaultSettingsBuilder::default()
        .verbose(false)
        .build()
        .map_err(|e| format!("clarabel_settings: {e:?}"))?;

    let mut solver = DefaultSolver::new(
        &to_csc(p),
        q.as_slice(),
        &to_csc(&a),
        b.as_slice(),
        &cones,
        settings,
    )
    .map_err(|e| format!("clarabel_setup: {e:?}"))?;
    solver.solve();

    match solver.solution.status {
        SolverStatus::Solved | SolverStatus::AlmostSolved => {
            Ok(k::nalgebra::DVector::from_vec(solver.solution.x.clone()))
        }
        status => Err(format!("qp_not_solved: {status:?}")),
    }
}
```

If Step 3's docs check showed `DefaultSolver::new` returning `Self` (not `Result`), delete that `.map_err(...)?`. If it showed `P` must be upper-triangular, change the `P` conversion call to a variant of `to_csc` that keeps only `r <= c` entries — add a `triu: bool` parameter rather than a second function.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test qp`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock src/lib.rs src/qp.rs
git commit -m "feat: add Clarabel-backed generic QP solver module"
```

---

### Task 2: rewire `differential_ik` onto `qp::solve`, delete `active_set.rs`

**Files:**
- Modify: `src/kinematics.rs` (lines 1, 116–154, 165–215 in the current file)
- Delete: `src/active_set.rs`
- Modify: `src/lib.rs`

The existing tests in `src/kinematics.rs` (`equality_constraint_holds`, `limits_enforced_across_trajectory`, `limit_constraint_drops_when_goal_pulls_inward`, `equality_target_outside_limits_errors`) are the safety net — they must pass **unchanged**. Do not edit them.

- [ ] **Step 1: Replace the constraint-resolution helper**

In `src/kinematics.rs`, replace the import on line 1:

```rust
use crate::active_set::{Constraint, ConstraintKind, active_set_step};
```

with:

```rust
use crate::qp;
```

Replace the whole `resolve_equality_constraints` function (lines 116–154) with a version that produces the equality system directly. Validation behavior (unknown joint name errors, out-of-limits target errors) is identical:

```rust
/// Build the equality system A_eq·q = targets from the configured constraints.
/// A_eq is one selector row per constraint; `targets` are absolute joint
/// values. Errors on unknown joint names and on targets outside joint limits.
fn resolve_equality_constraints(
    joint_names: &[String],
    config: &DifferentialIkConfig,
    lower: &k::nalgebra::DVector<f64>,
    upper: &k::nalgebra::DVector<f64>,
) -> Result<(k::nalgebra::DMatrix<f64>, k::nalgebra::DVector<f64>), String> {
    let n = joint_names.len();
    let m = config.equality_constraints.len();
    let mut a_eq = k::nalgebra::DMatrix::<f64>::zeros(m, n);
    let mut targets = k::nalgebra::DVector::<f64>::zeros(m);

    for (row, c) in config.equality_constraints.iter().enumerate() {
        let i = joint_names
            .iter()
            .position(|name| name == &c.joint_name)
            .ok_or_else(|| {
                format!("equality constraint joint '{}' not in serial chain", c.joint_name)
            })?;
        if c.target_value < lower[i] || c.target_value > upper[i] {
            return Err(format!(
                "equality target {} for joint '{}' outside limits [{}, {}]",
                c.target_value, c.joint_name, lower[i], upper[i]
            ));
        }
        a_eq[(row, i)] = 1.0;
        targets[row] = c.target_value;
    }
    Ok((a_eq, targets))
}
```

- [ ] **Step 2: Replace the solve step inside `differential_ik`**

In `differential_ik`, replace:

```rust
    // Resolve constraint joint names to indices once at startup
    let equality_constraints =
        resolve_equality_constraints(&kinematics.joint_names(), config, &lower, &upper)?;
```

with:

```rust
    // Resolve constraint joint names to an equality system once at startup.
    let (a_eq, eq_targets) =
        resolve_equality_constraints(&kinematics.joint_names(), config, &lower, &upper)?;
```

and replace the body of the step loop — the `active_set_step` call and the two lines above it:

```rust
        let current_joint_positions = k::nalgebra::DVector::from_vec(kinematics.positions());
        let jacobian = kinematics.jacobian();

        let updated_joint_positions = active_set_step(
            &jacobian,
            &twist,
            &current_joint_positions,
            &equality_constraints,
            &lower,
            &upper,
            config.damping_factor,
        )?;
```

with the damped Gauss-Newton QP over the step `dq`:

```rust
        let current_joint_positions = k::nalgebra::DVector::from_vec(kinematics.positions());
        let jacobian = kinematics.jacobian();
        let n = current_joint_positions.len();

        // Damped Gauss-Newton step as a QP over dq:
        //   min ½dqᵀ(JᵀJ + λ²I)dq − (Jᵀν)ᵀdq
        //   s.t. A_eq·dq = targets − q  (pinned joints, absolute targets)
        //        lower − q ≤ dq ≤ upper − q  (joint limits)
        // Equivalent to the old DLS/KKT step; Clarabel handles which limits
        // are active, replacing the hand-rolled active-set loop.
        let jt = jacobian.transpose();
        let p = &jt * &jacobian
            + config.damping_factor.powi(2) * k::nalgebra::DMatrix::identity(n, n);
        let twist_dyn = k::nalgebra::DVector::from_vec(twist.as_slice().to_vec());
        let q_lin = -(&jt * &twist_dyn);

        let dq = qp::solve(
            &p,
            &q_lin,
            &a_eq,
            &(&eq_targets - &a_eq * &current_joint_positions),
            &(&lower - &current_joint_positions),
            &(&upper - &current_joint_positions),
        )?;

        // Interior-point iterates are feasible to solver tolerance, not
        // exactly; clamp so downstream consumers can rely on hard limits.
        let mut updated_joint_positions = current_joint_positions + dq;
        for i in 0..n {
            updated_joint_positions[i] = updated_joint_positions[i].clamp(lower[i], upper[i]);
        }
```

(The lines after — `kinematics.set_positions(...)` and the trajectory push — stay as they are.)

- [ ] **Step 3: Delete the active-set module**

```bash
git rm src/active_set.rs
```

In `src/lib.rs` delete the line:

```rust
mod active_set;
```

- [ ] **Step 4: Run the full test suite**

Run: `cargo test`
Expected: all tests pass — the 4 remaining kinematics tests (the old 4 active-set unit tests left with the deleted file; their behaviors are covered by the kinematics tests plus Task 1's qp tests), 2 config tests, 1 se3_log test, 5 qp tests. If `equality_constraint_holds` fails its `1e-9` assertion because the solver's equality residual is looser than the old exact KKT solve, tighten Clarabel's tolerances in `qp::solve` (`.tol_feas(1e-12)` on the settings builder) rather than loosening the test.

- [ ] **Step 5: Clippy and format**

Run: `cargo clippy --all-targets && cargo fmt`
Expected: no warnings from the new code.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: solve IK steps with Clarabel QP, drop active-set solver

The hand-rolled active-set KKT loop is what a QP solver does
internally; Clarabel additionally certifies optimality and
infeasibility, and the qp module becomes the subproblem seam for the
future SQP trajectory optimizer."
```

---

### Task 3: end-to-end sanity run and push

**Files:** none modified.

- [ ] **Step 1: Solve the shipped config end to end**

Run: `cargo run --release`
Expected: exits 0 with the rerun viewer showing the same qualitative trajectory as before the refactor (base drives along the path, first pose respects the pinned shoulder/elbow). If any pose returns `qp_not_solved: PrimalInfeasible`, the goal is unreachable under the constraints — same class of failure the old solver reported as a singular system; verify the config didn't change before debugging the solver.

- [ ] **Step 2: Push**

```bash
git push
```

---

## Self-review notes

- Spec coverage: clarabel adoption (Tasks 1–2), SQP-ready seam (generic `qp::solve` signature; scope note pins SQP itself to a future plan), old solver removal (Task 2 Step 3), behavior preserved (Task 2 Step 4 runs the untouched kinematics tests).
- Types consistent: `qp::solve(&DMatrix, &DVector, &DMatrix, &DVector, &DVector, &DVector) -> Result<DVector, String>` used identically in Task 1 tests and Task 2 call site; `resolve_equality_constraints` returns `(DMatrix, DVector)` and both call-site bindings match.
- Known unknowns are localized: the two unverified Clarabel API details are flagged where they bite (Task 1 Steps 3–4) with the exact fallback edit.
