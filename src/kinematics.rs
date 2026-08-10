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
}

/// Solve inverse kinematics using differential IK (damped least squares).
///
/// Iteratively advances joint positions along the pseudo-inverse Jacobian
/// direction to minimize the twist error between current and goal pose.
/// Uses damped least squares (Levenberg-Marquardt) for numerical stability
/// near singularities.
///
/// # Arguments
/// * `goal_pose` - Target end-effector pose in world frame
/// * `kinematics` - Kinematics struct containing chain and solver
/// * `config` - Configuration including steps, damping, and convergence threshold
///
/// # Returns
/// `Ok` with vector of joint position snapshots, one per iteration (including initial state).
/// Length ≤ `config.num_steps + 1`, stops early if converged.
/// `Err` if the current pose is singular or matrix inversion fails.
///
/// # Algorithm
/// For each iteration:
/// 1. Compute current end-effector pose
/// 2. Calculate relative transform: goal * current⁻¹
/// 3. Extract twist via se3_log (matrix logarithm)
/// 4. Check convergence: if twist norm < threshold, stop
/// 5. Compute damped pseudo-inverse of Jacobian (cached once per iteration)
/// 6. Update joints: q_new = q_old + J†_damp * twist
/// 7. Feed back updated positions to chain for next iteration
pub fn differential_ik(
    goal_pose: &k::Isometry3<f64>,
    kinematics: &Kinematics,
    config: &DifferentialIkConfig,
) -> Result<Vec<Vec<f64>>, &'static str> {
    // Convert goal to homogeneous matrix for composition
    let goal_pose = goal_pose.to_homogeneous();

    // Initialize trajectory log with starting configuration
    let mut joint_positions: Vec<Vec<f64>> = vec![];
    joint_positions.push(kinematics.serial_chain.joint_positions());

    for _ in 0..config.num_steps {
        // Current end-effector pose in world frame
        let current_pose = kinematics.serial_chain.end_transform().to_homogeneous();

        // Compute relative error transform: T_err = T_goal * T_current⁻¹
        // This gives the transform from current to goal
        let current_pose_inverted = current_pose.try_inverse().ok_or("singular_current_pose")?;
        let temp = goal_pose * current_pose_inverted;

        // Convert relative pose to body twist [v; omega]
        // twist represents instantaneous velocity needed to reach goal
        let twist = se3_log(&temp);

        // Check convergence: if twist norm is small enough, we're done
        if twist.norm() < config.convergence_threshold {
            break;
        }

        // Read current joint configuration as nalgebra vector
        let current_joint_positions =
            k::nalgebra::DVector::from_vec(kinematics.serial_chain.joint_positions());

        // Cache Jacobian: compute once per iteration, reuse for DLS computation
        let jacobian = k::jacobian(&kinematics.serial_chain);
        let j_transpose = jacobian.transpose();

        // Damping factor from config for numerical stability near singularities
        let lambda = config.damping_factor;

        // Damped least squares pseudo-inverse: J†_damp = Jᵀ * (J*Jᵀ + λ²*I)⁻¹
        // Avoids singularity issues when J is rank-deficient
        let j_times_jt = jacobian * j_transpose.clone();
        let regularization = f64::powf(lambda, 2.0) * k::nalgebra::DMatrix::identity(6, 6);
        let dls_term = j_transpose
            * (j_times_jt + regularization)
                .try_inverse()
                .ok_or("matrix_inversion_failed")?;

        // Gradient step: q_new = q_old + J†_damp * twist
        let updated_joint_positions = current_joint_positions + dls_term * twist;

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
}
