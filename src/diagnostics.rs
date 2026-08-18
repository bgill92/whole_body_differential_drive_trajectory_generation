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
/// Signed: positive is slip to the base's right.
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
        let base_names: Vec<String> = ["px", "py", "yaw"].iter().map(|s| s.to_string()).collect();
        let base = resolve_base_indices(&names, &base_names).unwrap();
        assert_eq!((base.x, base.y, base.yaw), (0, 1, 2));
    }

    #[test]
    fn resolve_base_indices_rejects_missing_joint() {
        let names: Vec<String> = ["px"].iter().map(|s| s.to_string()).collect();
        let base_names: Vec<String> = ["px", "py", "yaw"].iter().map(|s| s.to_string()).collect();
        assert!(resolve_base_indices(&names, &base_names).is_err());
    }
}
