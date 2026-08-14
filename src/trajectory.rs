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
