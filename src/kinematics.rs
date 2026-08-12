// use k::prelude::*;
use k::InverseKinematicsSolver;

use crate::configs::{DifferentialIkConfig, SolverConfig};

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

    let omega = rotation.scaled_axis();
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

/// A wrapper around k::Chain and k::SerialChain with an IK solver.
pub struct Kinematics {
    pub chain: k::Chain<f64>,
    pub serial_chain: k::SerialChain<f64>,
    pub solver: k::JacobianIkSolver<f64>,
}

impl Kinematics {
    pub fn build(
        urdf_path: &str,
        serial_chain_end_joint: &str,
        solver_config: &SolverConfig,
    ) -> Result<Kinematics, &'static str> {
        let chain = k::Chain::<f64>::from_urdf_file(urdf_path).unwrap();
        let end = chain
            .find(serial_chain_end_joint)
            .ok_or("joint_not_found")?;
        let serial_chain = k::SerialChain::from_end(end);
        let solver = k::JacobianIkSolver::new(
            solver_config.allowable_target_distance,
            solver_config.allowable_target_angle,
            solver_config.jacobian_multiplier,
            solver_config.num_max_try,
        );

        Ok(Kinematics {
            chain,
            serial_chain,
            solver,
        })
    }

    pub fn solve(&self, target_pose: &k::Isometry3<f64>) -> Result<(), k::Error> {
        self.solver.solve(&self.serial_chain, target_pose)?;
        Ok(())
    }

    pub fn get_serial_chain_joint_names_and_positions(&self) -> (Vec<String>, Vec<f64>) {
        let names: Vec<String> = self
            .serial_chain
            .iter_joints()
            .map(|j| j.name.clone())
            .collect();
        let positions = self.serial_chain.joint_positions();
        (names, positions)
    }

    /// Joint lower and upper bounds for the serial chain.
    /// Returns (lower_bounds, upper_bounds) aligned by index with iter_joints().
    pub fn get_joint_limits(&self) -> (k::nalgebra::DVector<f64>, k::nalgebra::DVector<f64>) {
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

    /// Build active-set equality constraints from joints near their limits.
    ///
    /// Scans all joints and creates equality constraints for any joint within
    /// `tolerance` of its lower or upper bound. Returns (A, residuals) for use
    /// in the KKT solver:
    /// - A is m×n matrix with single 1.0 per row at the constrained joint index
    /// - residuals is length-m vector of (target_value - current_position)
    ///
    /// Joints with infinite bounds are never considered active.
    pub fn build_joint_limit_constraints(
        &self,
        current_positions: &k::nalgebra::DVector<f64>,
        tolerance: f64,
    ) -> (
        Option<k::nalgebra::DMatrix<f64>>,
        Option<k::nalgebra::DVector<f64>>,
    ) {
        let (lower, upper) = self.get_joint_limits();
        let mut active_indices: Vec<usize> = Vec::new();
        let mut active_targets: Vec<f64> = Vec::new();

        for i in 0..current_positions.len() {
            let q = current_positions[i];

            // Check lower limit (skip if unbounded)
            if lower[i].is_finite() && (q - lower[i]).abs() <= tolerance {
                active_indices.push(i);
                active_targets.push(lower[i]);
            }
            // Check upper limit (skip if unbounded)
            else if upper[i].is_finite() && (upper[i] - q).abs() <= tolerance {
                active_indices.push(i);
                active_targets.push(upper[i]);
            }
        }

        if active_indices.is_empty() {
            return (None, None);
        }

        let m = active_indices.len();
        let n = current_positions.len();
        let mut a = k::nalgebra::DMatrix::<f64>::zeros(m, n);
        let mut residuals = k::nalgebra::DVector::<f64>::zeros(m);

        for (row, (&idx, &target)) in active_indices.iter().zip(active_targets.iter()).enumerate() {
            a[(row, idx)] = 1.0;
            residuals[row] = target - current_positions[idx];
        }

        (Some(a), Some(residuals))
    }
}

/// Resolve equality constraints to (index, value) pairs.
/// Returns error if any joint name not found in serial chain.
fn resolve_equality_constraints(
    serial_chain: &k::SerialChain<f64>,
    config: &DifferentialIkConfig,
) -> Result<Vec<(usize, f64)>, String> {
    let mut resolved = Vec::new();
    for (i, joint) in serial_chain.iter_joints().enumerate() {
        if let Some(c) = config
            .equality_constraints
            .iter()
            .find(|c| c.joint_name == joint.name)
        {
            resolved.push((i, c.target_value));
        }
    }
    Ok(resolved)
}

/// Solve using KKT system with optional equality constraints.
/// Returns (updated_joint_positions, lagrange_multipliers).
/// When no constraints: uses damped least squares, returns empty multipliers.
/// When constraints present: solves full KKT system, returns non-zero multipliers.
fn solve_kkt_with_constraints(
    jacobian: &k::nalgebra::DMatrix<f64>,
    twist: &k::nalgebra::Vector6<f64>,
    constraints: Option<&[(usize, f64)]>,
    current_positions: &k::nalgebra::DVector<f64>,
    damping_factor: f64,
) -> (k::nalgebra::DVector<f64>, k::nalgebra::DVector<f64>) {
    let n = jacobian.ncols();
    let constraints = constraints.unwrap_or(&[]);
    let m = constraints.len();

    // Convert fixed-size twist to dynamic
    let twist_dyn = k::nalgebra::DVector::from_vec(twist.as_slice().to_vec());

    // No constraints: use damped least squares (Jᵀ * (J*Jᵀ + λ²*I)⁻¹ * twist)
    if m == 0 {
        let j_transpose = jacobian.transpose();
        let j_times_jt = jacobian * &j_transpose;
        let regularization = damping_factor.powi(2)
            * k::nalgebra::DMatrix::identity(jacobian.nrows(), jacobian.nrows());
        let dq = &j_transpose
            * (j_times_jt + regularization)
                .lu()
                .solve(&twist_dyn)
                .unwrap();
        return (current_positions + dq, k::nalgebra::DVector::zeros(0));
    }

    // Constraints present: build and solve full KKT system
    let j_transpose = jacobian.transpose();

    // H = Jᵀ * J + λ² * I (regularized Hessian)
    let h = &j_transpose * jacobian + damping_factor.powi(2) * k::nalgebra::DMatrix::identity(n, n);

    // Build constraint matrix A (each row has single 1.0 at constrained joint index)
    let mut a = k::nalgebra::DMatrix::<f64>::zeros(m, n);
    for (r, (idx, _value)) in constraints.iter().enumerate() {
        a[(r, *idx)] = 1.0;
    }

    // KKT matrix: [[H, Aᵀ], [A, 0]]
    let mut kkt = k::nalgebra::DMatrix::<f64>::zeros(n + m, n + m);
    kkt.slice_mut((0, 0), (n, n)).copy_from(&h);
    kkt.slice_mut((0, n), (n, m)).copy_from(&a.transpose());
    kkt.slice_mut((n, 0), (m, n)).copy_from(&a);

    // RHS: [Jᵀ * twist; residuals]
    let mut rhs = k::nalgebra::DVector::<f64>::zeros(n + m);
    rhs.rows_mut(0, n).copy_from(&(&j_transpose * &twist_dyn));
    for (r, (idx, target_value)) in constraints.iter().enumerate() {
        rhs[n + r] = target_value - current_positions[*idx];
    }

    // Solve KKT system
    let solution = kkt.lu().solve(&rhs).unwrap();

    // Extract dq (first n components) and lambdas (last m components)
    let dq = solution.rows(0, n);
    let mut lambdas = k::nalgebra::DVector::<f64>::zeros(m);
    lambdas.copy_from(&solution.rows(n, m));

    (current_positions + dq, lambdas)
}

/// Solve inverse kinematics using differential IK.
/// Uses KKT when equality constraints exist, otherwise damped least squares.
pub fn differential_ik(
    goal_pose: &k::Isometry3<f64>,
    kinematics: &Kinematics,
    config: &DifferentialIkConfig,
) -> Result<Vec<Vec<f64>>, String> {
    // Resolve constraint joint names to indices once at startup
    let constraints = resolve_equality_constraints(&kinematics.serial_chain, config)?;

    let goal_pose = goal_pose.to_homogeneous();

    // Initialize trajectory log with starting configuration
    let mut joint_positions: Vec<Vec<f64>> = vec![];
    joint_positions.push(kinematics.serial_chain.joint_positions());

    for _ in 0..config.num_steps {
        // Current end-effector pose in world frame
        let current_pose = kinematics.serial_chain.end_transform().to_homogeneous();

        // Compute relative error transform: T_err = T_goal * T_current⁻¹
        let current_pose_inverted = current_pose
            .try_inverse()
            .ok_or("singular_current_pose".to_string())?;
        let temp = goal_pose * current_pose_inverted;

        // Convert relative pose to body twist [v; omega]
        let twist = se3_log(&temp);

        // Check convergence
        if twist.norm() < config.convergence_threshold {
            break;
        }

        // Read current joint configuration as nalgebra vector
        let current_joint_positions =
            k::nalgebra::DVector::from_vec(kinematics.serial_chain.joint_positions());

        // Cache Jacobian
        let jacobian = k::jacobian(&kinematics.serial_chain);

        // Build joint limit constraints using active-set method
        let (limit_a, limit_residuals) =
            kinematics.build_joint_limit_constraints(&current_joint_positions, 0.01);

        // Merge equality constraints with joint limit constraints into single vec
        let mut merged_constraints = constraints.clone();
        if let (Some(a), Some(res)) = (&limit_a, &limit_residuals) {
            for row in 0..a.nrows() {
                if let Some(idx) = a.row(row).iter().position(|&val| val == 1.0) {
                    merged_constraints.push((idx, res[row]));
                }
            }
        }

        // Solve KKT system (handles both constrained and unconstrained cases)
        let (updated_joint_positions, _lagrange_multipliers) = solve_kkt_with_constraints(
            &jacobian,
            &twist,
            if merged_constraints.is_empty() {
                None
            } else {
                Some(&merged_constraints)
            },
            &current_joint_positions,
            config.damping_factor,
        );

        // Feed back to chain for next iteration's Jacobian computation
        kinematics
            .serial_chain
            .set_joint_positions_unchecked(updated_joint_positions.as_slice());

        // Record trajectory snapshot
        joint_positions.push(kinematics.serial_chain.joint_positions());
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

    #[test]
    fn joint_limits_extraction() {
        use crate::configs::SolverConfig;

        let solver_config = SolverConfig {
            allowable_target_distance: 0.01,
            allowable_target_angle: 0.01,
            jacobian_multiplier: 0.5,
            num_max_try: 100,
        };

        let kinematics = Kinematics::build(
            "assets/rox_diff_ur5e.urdf",
            "ur5ewrist_3_joint",
            &solver_config,
        )
        .expect("failed to build kinematics");

        let (lower, upper) = kinematics.get_joint_limits();
        let joint_count = kinematics.serial_chain.iter_joints().count();

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
    fn build_joint_limit_constraints() {
        use crate::configs::SolverConfig;

        let solver_config = SolverConfig {
            allowable_target_distance: 0.01,
            allowable_target_angle: 0.01,
            jacobian_multiplier: 0.5,
            num_max_try: 100,
        };

        let kinematics = Kinematics::build(
            "assets/rox_diff_ur5e.urdf",
            "ur5ewrist_3_joint",
            &solver_config,
        )
        .expect("failed to build kinematics");

        let (lower, upper) = kinematics.get_joint_limits();
        let n = lower.len();

        // Test 1: No active constraints when joints are far from limits
        let positions_far = k::nalgebra::DVector::from_vec(vec![0.0; n]);
        let (a, res) = kinematics.build_joint_limit_constraints(&positions_far, 0.01);
        assert!(a.is_none(), "no constraints when far from limits");
        assert!(res.is_none(), "no residuals when no constraints");

        // Test 2: Active constraint at upper limit
        let mut positions_at_upper = k::nalgebra::DVector::from_vec(vec![0.0; n]);
        if upper[0].is_finite() {
            positions_at_upper[0] = upper[0]; // Exactly at upper limit
            let (a, res) = kinematics.build_joint_limit_constraints(&positions_at_upper, 0.01);
            assert!(a.is_some(), "constraint created at upper limit");
            let a = a.unwrap();
            assert_eq!(a.nrows(), 1, "one active constraint");
            assert_eq!(a.ncols(), n);
            assert_eq!(a[(0, 0)], 1.0, "A matrix has 1.0 at constrained joint");
            assert!((res.unwrap()[0]) < 1e-10, "residual near zero at limit");
        }

        // Test 3: Active constraint at lower limit
        let mut positions_at_lower = k::nalgebra::DVector::from_vec(vec![0.0; n]);
        if lower[1].is_finite() {
            positions_at_lower[1] = lower[1]; // Exactly at lower limit
            let (a, _res) = kinematics.build_joint_limit_constraints(&positions_at_lower, 0.01);
            assert!(a.is_some(), "constraint created at lower limit");
            let a = a.unwrap();
            assert_eq!(a.nrows(), 1, "one active constraint");
            assert_eq!(a[(0, 1)], 1.0, "A matrix has 1.0 at constrained joint");
        }

        // Test 4: Multiple active constraints (within tolerance)
        let mut positions_multi = k::nalgebra::DVector::from_vec(vec![0.0; n]);
        let mut expected_count = 0;
        for i in 0..n {
            if upper[i].is_finite() && (upper[i] - positions_multi[i]).abs() <= 0.1 {
                positions_multi[i] = upper[i] - 0.005; // Within tolerance of upper
                expected_count += 1;
            }
        }
        let (a, _res) = kinematics.build_joint_limit_constraints(&positions_multi, 0.01);
        if expected_count > 0 {
            assert!(a.is_some(), "constraints created for multiple joints");
            assert_eq!(a.as_ref().unwrap().nrows(), expected_count);
        }
    }
}
