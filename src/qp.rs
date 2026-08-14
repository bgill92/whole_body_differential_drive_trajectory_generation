//! Generic dense convex QP solved with Clarabel:
//!   min ½ xᵀPx + qᵀx   s.t.   A_eq·x = b_eq,   lb ≤ x ≤ ub
//! Dense inputs are fine at IK size (n ≈ 10). This is also the seam the
//! future SQP/Gauss-Newton trajectory optimizer hands its subproblems to.

use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus, SupportedConeT,
};

/// Dense column-major nalgebra matrix → Clarabel CSC, dropping exact zeros.
///
/// Clarabel 0.11 accepts a full symmetric `P` and internally converts it to
/// upper-triangular form via `P.to_triu()` if needed, so no `triu` filtering
/// is required here.
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
    let n = q.len();
    let m_eq = a_eq.nrows();

    // Finite box bounds become one-sided rows in the nonnegative cone:
    // x_i ≤ ub_i  →  (+e_i)·x + s = ub_i ;  x_i ≥ lb_i  →  (−e_i)·x + s = −lb_i.
    let finite_ub: Vec<usize> = (0..n).filter(|&i| ub[i].is_finite()).collect();
    let finite_lb: Vec<usize> = (0..n).filter(|&i| lb[i].is_finite()).collect();
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

    let cones = [
        SupportedConeT::ZeroConeT(m_eq),
        SupportedConeT::NonnegativeConeT(m_ineq),
    ];
    // Default tol_feas (1e-8) leaves pinned-joint assertions at 1e-9 passing
    // only on incidental margin; tighten so equality rows are held to a
    // tolerance the callers can actually rely on.
    let settings = DefaultSettingsBuilder::default()
        .verbose(false)
        .tol_feas(1e-12)
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

#[cfg(test)]
mod tests {
    use super::*;
    use k::nalgebra::{DMatrix, DVector};

    fn no_eq(n: usize) -> (DMatrix<f64>, DVector<f64>) {
        (DMatrix::zeros(0, n), DVector::zeros(0))
    }

    fn no_in(n: usize) -> (DMatrix<f64>, DVector<f64>) {
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
        let (a_in, b_in) = no_in(2);
        let (lb, ub) = free_bounds(2);
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

    // Off-diagonal P guards the full-symmetric-P contract with Clarabel (it
    // triangularizes internally) and exercises to_csc with multiple entries
    // per column. min ½xᵀ[[2,1],[1,2]]x - [3,3]ᵀx has its minimum at [1, 1].
    #[test]
    fn off_diagonal_p_solves() {
        let mut p = DMatrix::zeros(2, 2);
        p[(0, 0)] = 2.0;
        p[(0, 1)] = 1.0;
        p[(1, 0)] = 1.0;
        p[(1, 1)] = 2.0;
        let (a_eq, b_eq) = no_eq(2);
        let (a_in, b_in) = no_in(2);
        let (lb, ub) = free_bounds(2);
        let x = solve(
            &p,
            &DVector::from_vec(vec![-3.0, -3.0]),
            &a_eq,
            &b_eq,
            &a_in,
            &b_in,
            &lb,
            &ub,
        )
        .unwrap();
        assert!((x[0] - 1.0).abs() < 1e-6, "got {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-6, "got {}", x[1]);
    }

    // Same objective, but x0 pinned to 0.3 by an equality row.
    #[test]
    fn equality_row_held() {
        let mut a_eq = DMatrix::zeros(1, 2);
        a_eq[(0, 0)] = 1.0;
        let b_eq = DVector::from_vec(vec![0.3]);
        let (a_in, b_in) = no_in(2);
        let (lb, ub) = free_bounds(2);
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
        assert!((x[0] - 0.3).abs() < 1e-6, "got {}", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-6, "got {}", x[1]);
    }

    // Unconstrained minimum is x1 = 2; upper bound 0.5 must clamp it.
    #[test]
    fn upper_bound_activates() {
        let (a_eq, b_eq) = no_eq(2);
        let (a_in, b_in) = no_in(2);
        let (lb, mut ub) = free_bounds(2);
        ub[1] = 0.5;
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
        assert!((x[1] - 0.5).abs() < 1e-6, "got {}", x[1]);
        assert!((x[0] - 1.0).abs() < 1e-6, "got {}", x[0]);
    }

    // Mirror of the above for the lower bound.
    #[test]
    fn lower_bound_activates() {
        let (a_eq, b_eq) = no_eq(2);
        let (a_in, b_in) = no_in(2);
        let (mut lb, ub) = free_bounds(2);
        lb[1] = 3.0;
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
        assert!((x[1] - 3.0).abs() < 1e-6, "got {}", x[1]);
    }

    // Contradictory equalities: x0 = 0 and x0 = 1. Must error, not panic.
    #[test]
    fn infeasible_problem_errors() {
        let mut a_eq = DMatrix::zeros(2, 1);
        a_eq[(0, 0)] = 1.0;
        a_eq[(1, 0)] = 1.0;
        let b_eq = DVector::from_vec(vec![0.0, 1.0]);
        let (a_in, b_in) = no_in(1);
        let (lb, ub) = free_bounds(1);
        let result = solve(
            &DMatrix::identity(1, 1),
            &DVector::zeros(1),
            &a_eq,
            &b_eq,
            &a_in,
            &b_in,
            &lb,
            &ub,
        );
        assert!(result.is_err());
    }

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
}
