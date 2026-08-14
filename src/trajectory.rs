//! Whole-body trajectory optimization: Gauss-Newton SQP over the stacked knot
//! configurations, one dense Clarabel QP per iteration. See
//! docs/superpowers/specs/2026-08-14-sqp-trajectory-optimization-design.md.

use crate::configs::{EeTracking, TrajectoryConfig};
use crate::kinematics::{Kinematics, pose_error_twist};
use crate::qp;

type DMat = k::nalgebra::DMatrix<f64>;
type DVec = k::nalgebra::DVector<f64>;

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
// The argument list mirrors the QP subproblem's mathematical structure (same
// rationale as qp::solve); bundling into a struct would obscure it.
#[allow(clippy::too_many_arguments)]
fn sqp_step(
    knots: &[DVec],
    goal_poses: &[k::Isometry3<f64>],
    kinematics: &mut Kinematics,
    config: &TrajectoryConfig,
    base: &BaseIndices,
    lower: &DVec,
    upper: &DVec,
    // Relaxes the linearized hard-mode EE targets (JΔ = α·r instead of
    // JΔ = r): when the full target conflicts with the trust region or
    // velocity rows, shrinking α enlarges the feasible set by asking for
    // less EE correction this iteration, rather than shrinking the box
    // (which only ever shrinks the feasible set and can never rescue an
    // infeasible subproblem). Unused in soft mode.
    hard_target_scale: f64,
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
        residuals.push(pose_error_twist(
            &goal.to_homogeneous(),
            &kinematics.end_pose(),
        ));
        if residuals.last().unwrap().iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "non-finite pose residual at knot {}; goal or configuration is degenerate",
                residuals.len() - 1
            ));
        }
        jacobians.push(kinematics.jacobian());
    }

    let mut p = DMat::zeros(dim, dim);
    let mut q_lin = DVec::zeros(dim);

    // EE tracking cost (soft mode only; hard mode uses equality rows below).
    if matches!(config.ee_tracking, EeTracking::Soft) {
        for k in 0..num_knots {
            let jt = jacobians[k].transpose();
            add_block(
                &mut p,
                k * n,
                k * n,
                &(config.ee_weight * (&jt * &jacobians[k])),
            );
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
            a_eq.slice_mut((row0, k * n), (6, n))
                .copy_from(&jacobians[k]);
            for axis in 0..6 {
                b_eq[row0 + axis] = hard_target_scale * residuals[k][axis];
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
    // Negative weights make P indefinite (JᵀJ and the smoothness Laplacian
    // are only guaranteed PSD with a nonnegative coefficient); a nonpositive
    // velocity budget makes every interval's inequality rows infeasible.
    if config.ee_weight < 0.0 || config.smoothness_weight < 0.0 {
        return Err("trajectory ee_weight and smoothness_weight must be >= 0".to_string());
    }
    if config.max_joint_velocity <= 0.0 {
        return Err("trajectory max_joint_velocity must be > 0".to_string());
    }
    let base = resolve_base_indices(&joint_names, &config.base_joint_names)?;
    let (lower, upper) = kinematics.joint_limits();

    let mut knots: Vec<DVec> = warm_start
        .iter()
        .map(|knot| {
            // An out-of-limits warm start would make the box lb > ub for
            // that variable, surfacing as an opaque QP infeasibility rather
            // than the actual cause.
            let mut positions = DVec::from_vec(knot.clone());
            for j in 0..n {
                positions[j] = positions[j].clamp(lower[j], upper[j]);
            }
            positions
        })
        .collect();

    let mut converged = false;
    for iteration in 0..config.sqp_max_iterations {
        // Hard mode only: relax the linearized EE equality targets
        // (JΔ = α·r instead of JΔ = r) when the full-target subproblem is
        // infeasible, e.g. near a kinematic singularity or a tight velocity
        // budget. Shrinking the trust region instead (the previous approach)
        // only ever shrinks the feasible set and can never rescue an
        // infeasible subproblem; relaxing the target enlarges it. Soft mode
        // has no equality EE rows, so retrying would repeat the identical
        // subproblem — call once and propagate any error directly. Four
        // halvings spans a 16x range — beyond that the iterate is genuinely
        // stuck and the error is real.
        let (step, relaxed) = if matches!(config.ee_tracking, EeTracking::Hard) {
            let mut alpha = 1.0;
            let mut step = None;
            let mut last_error = String::new();
            for _ in 0..=4 {
                match sqp_step(
                    &knots, goal_poses, kinematics, config, &base, &lower, &upper, alpha,
                ) {
                    Ok(s) => {
                        step = Some(s);
                        break;
                    }
                    Err(e) => {
                        last_error = e;
                        alpha *= 0.5;
                    }
                }
            }
            let step = step.ok_or_else(|| {
                format!(
                    "sqp iteration {iteration}: {last_error}; ee_tracking: hard can conflict with base kinematics — try soft"
                )
            })?;
            (step, alpha < 1.0)
        } else {
            let step = sqp_step(
                &knots, goal_poses, kinematics, config, &base, &lower, &upper, 1.0,
            )
            .map_err(|e| format!("sqp iteration {iteration}: {e}"))?;
            (step, false)
        };

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
        // A relaxed-target step can be small without meaning converged: it
        // only satisfied a scaled-down EE target, not the real one.
        if !relaxed && step_norm < config.convergence_step_norm {
            converged = true;
            break;
        }
    }

    if !converged {
        // The spec promises a stderr signal for best-effort returns; hard
        // mode can finish on a relaxed step with EE targets only partially
        // met, so silence here would hide real tracking error.
        eprintln!(
            "trajectory optimization did not converge within {} iterations; returning last iterate",
            config.sqp_max_iterations
        );
    }

    Ok(knots
        .into_iter()
        .map(|knot| knot.as_slice().to_vec())
        .collect())
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
        assert!(
            (c.abs() - 1.0).abs() < 1e-12,
            "lateral motion residual: {c}"
        );
    }

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
                // Both optimized[k] and optimized[k + 1] are indexed by j, so
                // an iterator/enumerate rewrite would not actually simplify
                // this.
                #[allow(clippy::needless_range_loop)]
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
            let twist = pose_error_twist(&goals[k].to_homogeneous(), &kinematics.end_pose());
            ee_error_sum += twist.norm();
        }
        let mean_ee_error = ee_error_sum / optimized.len() as f64;
        assert!(
            mean_ee_error < 0.25,
            "mean EE tracking error too large: {mean_ee_error}"
        );

        // Smoothness sign: a flipped linear term would bias the SQP toward
        // *larger* consecutive differences instead of penalizing them, so
        // the optimized trajectory must not be less smooth than warm start.
        let sum_sq_diff = |trajectory: &[Vec<f64>]| -> f64 {
            (0..trajectory.len() - 1)
                .map(|k| {
                    (0..n)
                        .map(|j| (trajectory[k + 1][j] - trajectory[k][j]).powi(2))
                        .sum::<f64>()
                })
                .sum()
        };
        let warm_start_smoothness = sum_sq_diff(&start);
        let optimized_smoothness = sum_sq_diff(&optimized);
        assert!(
            optimized_smoothness <= warm_start_smoothness + 1e-9,
            "optimized trajectory less smooth than warm start: {optimized_smoothness} > {warm_start_smoothness}"
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
            let (residual, _) = nonholonomic_linearization(&optimized[k], &optimized[k + 1], &base);
            assert!(residual.abs() < 1e-6, "slip {residual} at interval {k}");
        }
        // Hard-row sign: JΔ = r (not -r) must drive the EE pose to the goal,
        // not away from it.
        for (k, knot) in optimized.iter().enumerate() {
            kinematics.set_positions(knot);
            let twist = pose_error_twist(&goals[k].to_homogeneous(), &kinematics.end_pose());
            assert!(
                twist.norm() < 0.05,
                "hard-mode EE error {} too large at knot {k}",
                twist.norm()
            );
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
}
