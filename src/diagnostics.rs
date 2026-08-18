//! Read-only trajectory diagnostics: end-effector pose-tracking error and
//! differential-drive lateral-slip residuals. Never feeds back into the
//! solvers. See docs/superpowers/specs/2026-08-18-trajectory-diagnostics-design.md.

use crate::kinematics::Kinematics;

/// Chain indices of the planar base joints.
#[derive(Debug, Clone, Copy)]
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
/// Signed: positive is slip to the base's right.
pub fn slip_residual(q_k: &[f64], q_k1: &[f64], base: &BaseIndices) -> f64 {
    let dx = q_k1[base.x] - q_k[base.x];
    let dy = q_k1[base.y] - q_k[base.y];
    let mid_yaw = 0.5 * (q_k[base.yaw] + q_k1[base.yaw]);
    let (sin_mid, cos_mid) = mid_yaw.sin_cos();
    sin_mid * dx - cos_mid * dy
}

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
#[derive(Debug, Clone, Copy)]
pub struct SlipSummary {
    pub max_abs: f64,
    /// Knot interval of the worst violation; `None` for an empty series.
    pub max_index: Option<usize>,
    pub count_above_tol: usize,
}

/// Summarize a slip-residual series; a residual counts as a violation when
/// its magnitude exceeds `SLIP_TOLERANCE`.
pub fn summarize_slip(residuals: &[f64]) -> SlipSummary {
    let mut max_abs = 0.0;
    let mut max_index = None;
    let mut count_above_tol = 0;
    for (i, r) in residuals.iter().enumerate() {
        let abs = r.abs();
        if abs > max_abs {
            max_abs = abs;
            max_index = Some(i);
        }
        if abs > SLIP_TOLERANCE {
            count_above_tol += 1;
        }
    }
    SlipSummary {
        max_abs,
        max_index,
        count_above_tol,
    }
}

/// End-effector tracking error at one knot.
#[derive(Debug, Clone, Copy)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinematics::Kinematics;

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
        let names: Vec<String> = ["arm", "yaw", "px", "py"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let base_names: Vec<String> = ["px", "py", "yaw"].iter().map(|s| s.to_string()).collect();
        let base = resolve_base_indices(&names, &base_names).unwrap();
        assert_eq!((base.x, base.y, base.yaw), (2, 3, 1));
    }

    #[test]
    fn resolve_base_indices_rejects_missing_joint() {
        let names: Vec<String> = ["px"].iter().map(|s| s.to_string()).collect();
        let base_names: Vec<String> = ["px", "py", "yaw"].iter().map(|s| s.to_string()).collect();
        assert!(resolve_base_indices(&names, &base_names).is_err());
    }

    #[test]
    fn resolve_base_indices_rejects_wrong_count() {
        let names: Vec<String> = ["px", "py"].iter().map(|s| s.to_string()).collect();
        assert!(resolve_base_indices(&names, &names).is_err());
    }

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
        assert_eq!(summary.max_index, Some(1));
        assert_eq!(summary.count_above_tol, 2);
    }

    #[test]
    fn summarize_slip_empty_is_clean() {
        let summary = summarize_slip(&[]);
        assert_eq!(summary.max_abs, 0.0);
        assert!(summary.max_index.is_none());
        assert_eq!(summary.count_above_tol, 0);
    }

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
}
