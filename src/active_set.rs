//! Active-set solver for one damped-least-squares differential-IK step under
//! joint equality constraints and joint limits. Pure matrix math: no chain,
//! no URDF — the interface is `active_set_step`.

/// A joint within this distance of a limit joins the active set.
const LIMIT_TOLERANCE: f64 = 0.01;

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum ConstraintKind {
    /// User-requested joint value; never dropped from the working set.
    Equality,
    LowerLimit,
    UpperLimit,
}

/// One row of the KKT constraint block. `target` is an absolute joint value.
#[derive(Clone, Copy)]
pub(crate) struct Constraint {
    pub(crate) joint_index: usize,
    pub(crate) target: f64,
    pub(crate) kind: ConstraintKind,
}

/// Solve one damped-least-squares step, optionally under equality rows.
/// Returns (updated_joint_positions, lagrange_multipliers).
/// When no constraints: uses damped least squares, returns empty multipliers.
/// When constraints present: solves full KKT system.
fn solve_kkt_with_constraints(
    jacobian: &k::nalgebra::DMatrix<f64>,
    twist: &k::nalgebra::Vector6<f64>,
    constraints: &[Constraint],
    current_positions: &k::nalgebra::DVector<f64>,
    damping_factor: f64,
) -> Result<(k::nalgebra::DVector<f64>, k::nalgebra::DVector<f64>), String> {
    let n = jacobian.ncols();
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
                .ok_or_else(|| "singular_dls_system".to_string())?;
        return Ok((current_positions + dq, k::nalgebra::DVector::zeros(0)));
    }

    // Constraints present: build and solve full KKT system
    let j_transpose = jacobian.transpose();

    // H = Jᵀ * J + λ² * I (regularized Hessian)
    let h = &j_transpose * jacobian + damping_factor.powi(2) * k::nalgebra::DMatrix::identity(n, n);

    // Build constraint matrix A (each row has single 1.0 at constrained joint index)
    let mut a = k::nalgebra::DMatrix::<f64>::zeros(m, n);
    for (r, c) in constraints.iter().enumerate() {
        a[(r, c.joint_index)] = 1.0;
    }

    // KKT matrix: [[H, Aᵀ], [A, 0]]
    let mut kkt = k::nalgebra::DMatrix::<f64>::zeros(n + m, n + m);
    kkt.slice_mut((0, 0), (n, n)).copy_from(&h);
    kkt.slice_mut((0, n), (n, m)).copy_from(&a.transpose());
    kkt.slice_mut((n, 0), (m, n)).copy_from(&a);

    // RHS: [Jᵀ * twist; residuals]
    let mut rhs = k::nalgebra::DVector::<f64>::zeros(n + m);
    rhs.rows_mut(0, n).copy_from(&(&j_transpose * &twist_dyn));
    for (r, c) in constraints.iter().enumerate() {
        rhs[n + r] = c.target - current_positions[c.joint_index];
    }

    // Solve KKT system
    let solution = kkt
        .lu()
        .solve(&rhs)
        .ok_or_else(|| "singular_kkt_system".to_string())?;

    // Extract dq (first n components) and lambdas (last m components)
    let dq = solution.rows(0, n);
    let mut lambdas = k::nalgebra::DVector::<f64>::zeros(m);
    lambdas.copy_from(&solution.rows(n, m));

    Ok((current_positions + dq, lambdas))
}

/// Seed the working set: equalities (never dropped) plus limits active at the
/// current configuration. One-sided checks also catch joints that are already
/// past a limit. Equality-constrained joints skip limit rows; their targets
/// are validated in-bounds, so a limit row would only conflict.
fn seed_working_set(
    equality_constraints: &[Constraint],
    current_positions: &k::nalgebra::DVector<f64>,
    lower: &k::nalgebra::DVector<f64>,
    upper: &k::nalgebra::DVector<f64>,
) -> Vec<Constraint> {
    let mut working_set = equality_constraints.to_vec();
    for i in 0..current_positions.len() {
        if equality_constraints.iter().any(|c| c.joint_index == i) {
            continue;
        }
        let q = current_positions[i];
        if q >= upper[i] - LIMIT_TOLERANCE {
            working_set.push(Constraint {
                joint_index: i,
                target: upper[i],
                kind: ConstraintKind::UpperLimit,
            });
        } else if q <= lower[i] + LIMIT_TOLERANCE {
            working_set.push(Constraint {
                joint_index: i,
                target: lower[i],
                kind: ConstraintKind::LowerLimit,
            });
        }
    }
    working_set
}

/// Row of the worst wrong-signed limit multiplier, if any.
///
/// Sign convention from H·dq + Aᵀλ = Jᵀν with +1 constraint rows: an active
/// upper limit needs λ >= 0, a lower limit λ <= 0. Equality rows never drop.
fn wrong_signed_limit_row(
    working_set: &[Constraint],
    lambdas: &k::nalgebra::DVector<f64>,
) -> Option<usize> {
    let mut drop_row = None;
    let mut worst_wrong_sign = 1e-12;
    for (row, c) in working_set.iter().enumerate() {
        let wrong_sign = match c.kind {
            ConstraintKind::Equality => continue,
            ConstraintKind::UpperLimit => -lambdas[row],
            ConstraintKind::LowerLimit => lambdas[row],
        };
        if wrong_sign > worst_wrong_sign {
            worst_wrong_sign = wrong_sign;
            drop_row = Some(row);
        }
    }
    drop_row
}

/// Most violated limit among joints not already in the working set, if any.
fn most_violated_limit(
    working_set: &[Constraint],
    candidate: &k::nalgebra::DVector<f64>,
    lower: &k::nalgebra::DVector<f64>,
    upper: &k::nalgebra::DVector<f64>,
) -> Option<Constraint> {
    let mut add_constraint = None;
    let mut worst_violation = 1e-9;
    for i in 0..candidate.len() {
        if working_set.iter().any(|c| c.joint_index == i) {
            continue;
        }
        if candidate[i] - upper[i] > worst_violation {
            worst_violation = candidate[i] - upper[i];
            add_constraint = Some(Constraint {
                joint_index: i,
                target: upper[i],
                kind: ConstraintKind::UpperLimit,
            });
        }
        if lower[i] - candidate[i] > worst_violation {
            worst_violation = lower[i] - candidate[i];
            add_constraint = Some(Constraint {
                joint_index: i,
                target: lower[i],
                kind: ConstraintKind::LowerLimit,
            });
        }
    }
    add_constraint
}

/// One differential-IK step under the active-set method: drop one wrong-signed
/// limit multiplier, else add the most violated inactive limit, else accept.
/// Returns updated joint positions satisfying every limit.
pub(crate) fn active_set_step(
    jacobian: &k::nalgebra::DMatrix<f64>,
    twist: &k::nalgebra::Vector6<f64>,
    current_positions: &k::nalgebra::DVector<f64>,
    equality_constraints: &[Constraint],
    lower: &k::nalgebra::DVector<f64>,
    upper: &k::nalgebra::DVector<f64>,
    damping_factor: f64,
) -> Result<k::nalgebra::DVector<f64>, String> {
    let n = current_positions.len();
    let mut working_set = seed_working_set(equality_constraints, current_positions, lower, upper);

    let mut accepted = None;
    for _ in 0..=2 * n {
        let (candidate, lambdas) = solve_kkt_with_constraints(
            jacobian,
            twist,
            &working_set,
            current_positions,
            damping_factor,
        )?;

        if let Some(row) = wrong_signed_limit_row(&working_set, &lambdas) {
            accepted = Some(candidate);
            working_set.remove(row);
            continue;
        }

        let violated = most_violated_limit(&working_set, &candidate, lower, upper);
        accepted = Some(candidate);
        match violated {
            Some(c) => working_set.push(c),
            None => break,
        }
    }

    let mut updated_positions = accepted.expect("active-set loop solves at least once");
    // ponytail: safety clamp only matters if the iteration cap above is ever
    // hit; accepted solutions already satisfy the limits exactly.
    for i in 0..n {
        updated_positions[i] = updated_positions[i].clamp(lower[i], upper[i]);
    }
    Ok(updated_positions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use k::nalgebra::{DMatrix, DVector, Vector6};

    // Toy 2-joint system: joint 0 drives x, joint 1 drives y, one-to-one.
    fn toy_jacobian() -> DMatrix<f64> {
        let mut j = DMatrix::zeros(6, 2);
        j[(0, 0)] = 1.0;
        j[(1, 1)] = 1.0;
        j
    }

    fn wide_limits() -> (DVector<f64>, DVector<f64>) {
        (DVector::from_element(2, -10.0), DVector::from_element(2, 10.0))
    }

    #[test]
    fn equality_row_held_exactly() {
        let (lower, upper) = wide_limits();
        let equalities = [Constraint {
            joint_index: 0,
            target: 0.3,
            kind: ConstraintKind::Equality,
        }];
        let positions = active_set_step(
            &toy_jacobian(),
            &Vector6::new(1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            &DVector::zeros(2),
            &equalities,
            &lower,
            &upper,
            0.5,
        )
        .unwrap();
        assert!((positions[0] - 0.3).abs() < 1e-9, "got {}", positions[0]);
    }

    #[test]
    fn violated_upper_limit_activates() {
        let (lower, mut upper) = wide_limits();
        upper[0] = 0.5;
        // Twist demands dq0 ≈ 8, far past the 0.5 limit.
        let positions = active_set_step(
            &toy_jacobian(),
            &Vector6::new(10.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            &DVector::zeros(2),
            &[],
            &lower,
            &upper,
            0.5,
        )
        .unwrap();
        assert!((positions[0] - 0.5).abs() < 1e-9, "got {}", positions[0]);
    }

    #[test]
    fn pinned_limit_drops_when_twist_pulls_inward() {
        let (lower, mut upper) = wide_limits();
        upper[0] = 0.5;
        // Joint 0 starts at its upper limit (seeded into the working set);
        // the twist pulls inward, so the multiplier is wrong-signed and the
        // limit row must drop.
        let positions = active_set_step(
            &toy_jacobian(),
            &Vector6::new(-1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            &DVector::from_vec(vec![0.5, 0.0]),
            &[],
            &lower,
            &upper,
            0.5,
        )
        .unwrap();
        assert!(positions[0] < 0.5 - 1e-6, "stayed pinned: {}", positions[0]);
    }

    #[test]
    fn singular_system_errors() {
        // Zero Jacobian with zero damping: J*Jᵀ + 0·I is singular.
        let result = active_set_step(
            &DMatrix::zeros(6, 2),
            &Vector6::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            &DVector::zeros(2),
            &[],
            &wide_limits().0,
            &wide_limits().1,
            0.0,
        );
        assert!(result.is_err());
    }
}
