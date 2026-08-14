use crate::configs::DifferentialIkConfig;
use crate::qp;

/// Matrix logarithm of a homogeneous transform: SE(3) -> se(3).
///
/// Returns the twist `[v; omega]` whose matrix exponential reproduces `pose`.
/// `omega` is the axis-angle vector; `v` is the translation mapped through the
/// inverse left Jacobian, *not* the raw translation.
///
/// Linear-first to match `k::jacobian`, whose rows are `[linear; angular]`.
/// Modern Robotics uses the opposite order (`[omega; v]`) -- reorder when
/// cross-checking against the book.
///
/// The rotation block is taken as-is; a non-orthonormal `pose` yields garbage
/// rather than an error.
pub fn se3_log(pose: &k::nalgebra::Matrix4<f64>) -> k::nalgebra::Vector6<f64> {
    let rotation =
        k::nalgebra::Rotation3::from_matrix_unchecked(pose.fixed_slice::<3, 3>(0, 0).into_owned());
    let translation = pose.fixed_slice::<3, 1>(0, 3).into_owned();

    // Quaternion extraction (Shepperd's method) is well-conditioned at all
    // angles including π, where the matrix-log route (acos of a trace that
    // float drift can push below −1) returns NaN.
    let omega = k::nalgebra::UnitQuaternion::from_rotation_matrix(&rotation).scaled_axis();
    let theta = omega.norm();
    let omega_hat = omega.cross_matrix();

    // Coefficient on (omega_hat)^2 in V^-1. The closed form
    // 1/theta^2 - (1 + cos)/(2 * theta * sin) is 0/0 at theta = 0, so use its
    // Taylor series near the singularity. Same story at theta = pi, where sin
    // vanishes again -- the series is not valid there, but scaled_axis() keeps
    // theta <= pi and the term stays finite in the limit.
    let coeff = if theta < 1e-6 {
        1.0 / 12.0 + theta * theta / 720.0
    } else {
        1.0 / (theta * theta) - (1.0 + theta.cos()) / (2.0 * theta * theta.sin())
    };

    let v_inv = k::nalgebra::Matrix3::identity() - 0.5 * omega_hat + coeff * omega_hat * omega_hat;
    let v = v_inv * translation;

    k::nalgebra::Vector6::new(v[0], v[1], v[2], omega[0], omega[1], omega[2])
}

/// Serial-chain kinematics loaded from a URDF.
///
/// Invariant: link transforms are always consistent with the joint positions
/// — `build` and `set_positions` both refresh them, so `end_pose` and
/// `jacobian` never need a separate update call.
pub struct Kinematics {
    serial_chain: k::SerialChain<f64>,
}

impl Kinematics {
    pub fn build(
        urdf_path: &str,
        serial_chain_end_joint: &str,
    ) -> Result<Kinematics, &'static str> {
        let chain = k::Chain::<f64>::from_urdf_file(urdf_path).unwrap();
        let end = chain
            .find(serial_chain_end_joint)
            .ok_or("joint_not_found")?;
        let serial_chain = k::SerialChain::from_end(end);
        serial_chain.update_transforms();

        Ok(Kinematics { serial_chain })
    }

    /// Joint names in serial-chain order; all other per-joint values
    /// (positions, limits) align with this order by index.
    pub fn joint_names(&self) -> Vec<String> {
        self.serial_chain
            .iter_joints()
            .map(|j| j.name.clone())
            .collect()
    }

    pub fn positions(&self) -> Vec<f64> {
        self.serial_chain.joint_positions()
    }

    /// Set joint positions and refresh link transforms.
    pub fn set_positions(&mut self, positions: &[f64]) {
        self.serial_chain.set_joint_positions_unchecked(positions);
        self.serial_chain.update_transforms();
    }

    pub fn jacobian(&self) -> k::nalgebra::DMatrix<f64> {
        k::jacobian(&self.serial_chain)
    }

    /// Current end-effector pose in the world frame.
    pub fn end_pose(&self) -> k::Isometry3<f64> {
        self.serial_chain.end_transform()
    }

    /// Joint lower and upper bounds for the serial chain.
    /// Returns (lower_bounds, upper_bounds) aligned by index with joint_names().
    pub fn joint_limits(&self) -> (k::nalgebra::DVector<f64>, k::nalgebra::DVector<f64>) {
        let n = self.serial_chain.iter_joints().count();
        let mut lower = k::nalgebra::DVector::<f64>::zeros(n);
        let mut upper = k::nalgebra::DVector::<f64>::zeros(n);

        for (i, joint) in self.serial_chain.iter_joints().enumerate() {
            // Joint.limits is Option<Range<f64>> with min/max fields;
            // None means no limit (use -inf/+inf as sentinels).
            if let Some(limits) = &joint.limits {
                lower[i] = limits.min;
                upper[i] = limits.max;
            } else {
                // No explicit limit: use wide sentinel bounds
                lower[i] = f64::NEG_INFINITY;
                upper[i] = f64::INFINITY;
            }
        }

        (lower, upper)
    }
}

/// Build the equality system A_eq·q = targets from the configured constraints.
/// A_eq is one selector row per constraint; `targets` are absolute joint
/// values. Errors on unknown joint names, duplicate joints, and on targets
/// outside joint limits.
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
    let mut seen_joints = std::collections::HashSet::new();

    for (row, c) in config.equality_constraints.iter().enumerate() {
        let i = joint_names
            .iter()
            .position(|name| name == &c.joint_name)
            .ok_or_else(|| {
                format!(
                    "equality constraint joint '{}' not in serial chain",
                    c.joint_name
                )
            })?;
        // Two rows pinning the same joint to different targets make the
        // per-step QP infeasible, surfacing later as an opaque
        // `qp_not_solved: PrimalInfeasible` -- catch it here with a clear
        // message instead.
        if !seen_joints.insert(&c.joint_name) {
            return Err(format!(
                "duplicate equality constraint for joint '{}'",
                c.joint_name
            ));
        }
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

/// Spatial twist [v; omega] taking `current_pose` to `goal_pose`.
pub(crate) fn pose_error_twist(
    goal_pose: &k::nalgebra::Matrix4<f64>,
    current_pose: &k::Isometry3<f64>,
) -> k::nalgebra::Vector6<f64> {
    // Isometry inversion is exact (transposed rotation), never singular.
    se3_log(&(goal_pose * current_pose.inverse().to_homogeneous()))
}

/// Solve inverse kinematics using differential IK.
///
/// Each step solves a damped-least-squares QP subject to user equality
/// constraints and URDF joint limits, via Clarabel (see `qp::solve`). Clarabel's
/// interior-point method handles which limits are active internally, replacing
/// the hand-rolled active-set loop this function used to run.
pub fn differential_ik(
    goal_pose: &k::Isometry3<f64>,
    kinematics: &mut Kinematics,
    config: &DifferentialIkConfig,
) -> Result<Vec<Vec<f64>>, String> {
    // P = JᵀJ + λ²I must be positive definite; with λ ≤ 0 and a singular J
    // the QP has no unique minimizer and Clarabel would return one
    // arbitrarily rather than erroring, unlike the old DLS solve.
    if config.damping_factor <= 0.0 {
        return Err(format!(
            "damping_factor must be > 0: P = JᵀJ + λ²I must be positive definite (got {})",
            config.damping_factor
        ));
    }

    let (lower, upper) = kinematics.joint_limits();

    // Resolve constraint joint names to an equality system once at startup.
    let (a_eq, eq_targets) =
        resolve_equality_constraints(&kinematics.joint_names(), config, &lower, &upper)?;

    let goal_pose = goal_pose.to_homogeneous();

    // Initialize trajectory log with starting configuration
    let mut joint_positions: Vec<Vec<f64>> = vec![kinematics.positions()];

    for _ in 0..config.num_steps {
        let twist = pose_error_twist(&goal_pose, &kinematics.end_pose());
        if twist.norm() < config.convergence_threshold {
            break;
        }

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
        let p =
            &jt * &jacobian + config.damping_factor.powi(2) * k::nalgebra::DMatrix::identity(n, n);
        let twist_dyn = k::nalgebra::DVector::from_vec(twist.as_slice().to_vec());
        let q_lin = -(&jt * &twist_dyn);

        let dq = qp::solve(
            &p,
            &q_lin,
            &a_eq,
            &(&eq_targets - &a_eq * &current_joint_positions),
            &k::nalgebra::DMatrix::zeros(0, n),
            &k::nalgebra::DVector::zeros(0),
            &(&lower - &current_joint_positions),
            &(&upper - &current_joint_positions),
        )?;

        // Interior-point iterates are feasible to solver tolerance, not
        // exactly; clamp so downstream consumers can rely on hard limits.
        let mut updated_joint_positions = current_joint_positions + dq;
        for i in 0..n {
            updated_joint_positions[i] = updated_joint_positions[i].clamp(lower[i], upper[i]);
        }

        // Feed back for next iteration's Jacobian computation
        kinematics.set_positions(updated_joint_positions.as_slice());

        // Record trajectory snapshot
        joint_positions.push(kinematics.positions());
    }

    Ok(joint_positions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use k::nalgebra::{Isometry3, Matrix4, UnitQuaternion, Vector3};

    // Cross-check se3_log against nalgebra's general matrix exponential: expm of
    // the hat matrix must reproduce the original homogeneous transform.
    fn assert_roundtrip(pose: &Isometry3<f64>) {
        let twist = se3_log(&pose.to_homogeneous());
        let omega_hat = Vector3::new(twist[3], twist[4], twist[5]).cross_matrix();

        let mut hat = Matrix4::zeros();
        hat.fixed_slice_mut::<3, 3>(0, 0).copy_from(&omega_hat);
        hat[(0, 3)] = twist[0];
        hat[(1, 3)] = twist[1];
        hat[(2, 3)] = twist[2];

        let rebuilt = hat.exp();
        assert!(
            (rebuilt - pose.to_homogeneous()).norm() < 1e-9,
            "roundtrip failed\nexpected:{}\ngot:{}",
            pose.to_homogeneous(),
            rebuilt
        );
    }

    #[test]
    fn log_roundtrips() {
        assert_roundtrip(&Isometry3::identity());
        // Pure translation: rotation is exactly zero, so the small-angle branch runs.
        assert_roundtrip(&Isometry3::translation(1.0, -2.0, 3.0));
        assert_roundtrip(&Isometry3::from_parts(
            Vector3::new(0.3, -0.7, 1.1).into(),
            UnitQuaternion::from_scaled_axis(Vector3::new(0.2, -1.3, 0.5)),
        ));
        // Near the theta = pi singularity.
        assert_roundtrip(&Isometry3::from_parts(
            Vector3::new(-1.0, 4.0, 0.5).into(),
            UnitQuaternion::from_scaled_axis(Vector3::x() * (std::f64::consts::PI - 1e-4)),
        ));
    }

    // Exactly-π rotations are the acos-of-trace singularity of the matrix
    // log; the quaternion route must stay finite and roundtrip.
    #[test]
    fn log_roundtrips_at_exactly_pi() {
        for axis in [Vector3::x(), Vector3::y(), Vector3::z()] {
            let pose = Isometry3::from_parts(
                Vector3::new(0.4, -0.2, 1.3).into(),
                UnitQuaternion::from_scaled_axis(axis * std::f64::consts::PI),
            );
            let twist = se3_log(&pose.to_homogeneous());
            assert!(
                twist.iter().all(|v| v.is_finite()),
                "NaN twist for axis {axis:?}"
            );
            assert_roundtrip(&pose);
        }
    }

    fn test_kinematics() -> Kinematics {
        Kinematics::build("assets/rox_diff_ur5e.urdf", "ur5ewrist_3_joint")
            .expect("failed to build kinematics")
    }

    fn test_config(
        equality_constraints: Vec<crate::configs::EqualityConstraint>,
    ) -> DifferentialIkConfig {
        DifferentialIkConfig {
            num_steps: 50,
            damping_factor: 0.5,
            convergence_threshold: 0.01,
            equality_constraints,
        }
    }

    fn joint_index(kinematics: &Kinematics, name: &str) -> usize {
        kinematics
            .joint_names()
            .iter()
            .position(|n| n == name)
            .expect("joint not found")
    }

    #[test]
    fn joint_limits_extraction() {
        let kinematics = test_kinematics();
        let (lower, upper) = kinematics.joint_limits();
        let joint_count = kinematics.joint_names().len();

        assert_eq!(lower.len(), joint_count);
        assert_eq!(upper.len(), joint_count);

        for i in 0..joint_count {
            assert!(
                lower[i] <= upper[i],
                "invalid limits at joint {}: lower={} > upper={}",
                i,
                lower[i],
                upper[i]
            );
        }
    }

    #[test]
    fn equality_constraint_holds() {
        let mut kinematics = test_kinematics();

        // Offset goal so the solver actually iterates instead of converging
        // immediately at the start pose.
        let mut goal = kinematics.end_pose();
        goal.translation.vector += Vector3::new(0.1, 0.1, 0.0);

        let config = test_config(vec![crate::configs::EqualityConstraint {
            joint_name: "ur5eshoulder_pan_joint".to_string(),
            target_value: 1.0,
        }]);
        let trajectory = differential_ik(&goal, &mut kinematics, &config).unwrap();
        assert!(trajectory.len() > 1, "solver took no steps");

        let pan = joint_index(&kinematics, "ur5eshoulder_pan_joint");
        let last = trajectory.last().unwrap();
        assert!(
            (last[pan] - 1.0).abs() < 1e-9,
            "equality constraint not held: pan = {}",
            last[pan]
        );
    }

    #[test]
    fn limits_enforced_across_trajectory() {
        let mut kinematics = test_kinematics();
        let (lower, upper) = kinematics.joint_limits();
        let n = lower.len();

        // Start the elbow inside the activation tolerance of its upper limit.
        let elbow = joint_index(&kinematics, "ur5eelbow_joint");
        let mut start = vec![0.0; n];
        start[elbow] = upper[elbow] - 0.005;
        kinematics.set_positions(&start);

        let goal = Isometry3::from_parts(
            Vector3::new(1.0, 1.0, 1.0).into(),
            UnitQuaternion::from_euler_angles(std::f64::consts::PI, 0.0, 0.0),
        );
        let trajectory = differential_ik(&goal, &mut kinematics, &test_config(vec![])).unwrap();
        assert!(trajectory.len() > 1, "solver took no steps");

        for (step, positions) in trajectory.iter().enumerate() {
            for i in 0..n {
                assert!(
                    positions[i] >= lower[i] - 1e-9 && positions[i] <= upper[i] + 1e-9,
                    "joint {} out of limits at step {}: {} not in [{}, {}]",
                    i,
                    step,
                    positions[i],
                    lower[i],
                    upper[i]
                );
            }
        }
    }

    #[test]
    fn limit_constraint_drops_when_goal_pulls_inward() {
        let mut kinematics = test_kinematics();
        let (_, upper) = kinematics.joint_limits();
        let elbow = joint_index(&kinematics, "ur5eelbow_joint");
        let n = kinematics.joint_names().len();

        // Goal is the end pose of the all-zero configuration; the elbow then
        // starts pinned at its upper limit, so reaching the goal requires the
        // active-set loop to drop the limit constraint and move it inward.
        let goal = kinematics.end_pose();

        let mut start = vec![0.0; n];
        start[elbow] = upper[elbow];
        kinematics.set_positions(&start);

        let trajectory = differential_ik(&goal, &mut kinematics, &test_config(vec![])).unwrap();
        let last = trajectory.last().unwrap();
        assert!(
            last[elbow] < upper[elbow] - 1e-3,
            "elbow stayed pinned at its limit: {}",
            last[elbow]
        );
    }

    #[test]
    fn equality_target_outside_limits_errors() {
        let mut kinematics = test_kinematics();
        let goal = kinematics.end_pose();

        // Elbow limit is ±π; 100.0 is far outside.
        let config = test_config(vec![crate::configs::EqualityConstraint {
            joint_name: "ur5eelbow_joint".to_string(),
            target_value: 100.0,
        }]);
        assert!(differential_ik(&goal, &mut kinematics, &config).is_err());

        // Unknown joint names error too.
        let config = test_config(vec![crate::configs::EqualityConstraint {
            joint_name: "no_such_joint".to_string(),
            target_value: 0.0,
        }]);
        assert!(differential_ik(&goal, &mut kinematics, &config).is_err());
    }

    #[test]
    fn duplicate_equality_constraint_errors() {
        let mut kinematics = test_kinematics();
        let goal = kinematics.end_pose();

        // Two constraints on the same joint make every per-step QP
        // infeasible; this must be rejected up front with a clear message.
        let config = test_config(vec![
            crate::configs::EqualityConstraint {
                joint_name: "ur5eshoulder_pan_joint".to_string(),
                target_value: 0.5,
            },
            crate::configs::EqualityConstraint {
                joint_name: "ur5eshoulder_pan_joint".to_string(),
                target_value: 1.0,
            },
        ]);
        assert!(differential_ik(&goal, &mut kinematics, &config).is_err());
    }

    #[test]
    fn non_positive_damping_errors() {
        let mut kinematics = test_kinematics();
        let goal = kinematics.end_pose();

        let mut config = test_config(vec![]);
        config.damping_factor = 0.0;
        assert!(differential_ik(&goal, &mut kinematics, &config).is_err());
    }
}
