# QP / SQP Solver Crates for Rust — Survey

Research date: 2026-08-14. All versions, dates, and licenses verified against the
crates.io API, docs.rs, and the upstream GitHub repositories on that date.
Context: this project solves small dense QPs (~10 joint variables, box limits,
pinned-joint equalities) repeatedly inside a differential-IK loop
(`src/active_set.rs`, `src/kinematics.rs`, nalgebra via the `k` crate), and
later wants SQP over whole trajectories (hundreds–thousands of variables,
sparse block structure, nonlinear kinematic constraints).

## TL;DR

**For the IK loop now: [`clarabel`](https://crates.io/crates/clarabel).**
Pure Rust (no C toolchain, no cmake), Apache-2.0, a mature interior-point
conic/QP solver (the CVXPY ecosystem's default open-source IPM). Equalities
map to `ZeroConeT`, joint limits to `NonnegativeConeT` rows; input is its own
CSC sparse type, which is a few lines of conversion from a small nalgebra
`DMatrix`. Its one real gap — no warm starting — is irrelevant at n≈10, where a
cold solve is microseconds. If warm starting across control ticks ever becomes
a measured bottleneck, [`osqp`](https://crates.io/crates/osqp) (official
bindings, first-class `warm_start`/`update_*` API) is the alternative, at the
cost of a bundled-C cmake build and ADMM's lower per-iteration accuracy.

**For SQP later: there is no mature, battle-tested native Rust sparse SQP
today.** The pragmatic path, and what the ecosystem's structure points to, is a
custom SQP/Gauss-Newton outer loop whose linearized subproblems go to Clarabel
or OSQP — both are sparse-native, so the block-banded trajectory KKT structure
is exploited for free. Two credible alternatives: (1)
[`optimization_engine`](https://crates.io/crates/optimization_engine) (OpEn),
pure Rust PANOC+ALM explicitly built for embedded robotics NMPC, which avoids
SQP entirely; (2) the very new [POUNCE](https://github.com/jkitchin/pounce)
project — a pure-Rust port of Ipopt that includes an active-set SQP driver with
a warm-started sparse parametric QP subproblem solver (`pounce-qp`) — worth
watching but weeks-old and EPL-2.0. FFI fallback for large sparse NLPs:
[`ipopt`](https://crates.io/crates/ipopt) (ipopt-rs).

## Comparison table

| Crate | Version (date) | Pure Rust? | Algorithm | Constraints | Warm start | Dense/Sparse | License | Fit (a) small IK QP | Fit (b) sparse SQP |
|---|---|---|---|---|---|---|---|---|---|
| [clarabel](https://crates.io/crates/clarabel) | 0.11.1 (2025-06-11) | Yes | Interior point (homogeneous embedding) | eq (ZeroCone), ineq/box (NonnegativeCone), SOC/SDP/exp | **No** | Sparse CSC | Apache-2.0 | **Excellent** | Good as QP subsolver |
| [osqp](https://crates.io/crates/osqp) | 1.0.1 (2025-04-21) | No (bundled C, cmake) | ADMM | `l ≤ Ax ≤ u` (eq via `l=u`) | **Yes** + data updates | Sparse CSC | Apache-2.0 | Very good | Good as QP subsolver |
| [quadprog](https://crates.io/crates/quadprog) | 0.1.0 (2026-07-03) | Yes | Goldfarb–Idnani dual active set | eq + ineq (dense rows) | No | Dense (row-major slices) | **GPL-2.0** | Good technically; license caveat | No (dense only) |
| [optimization_engine](https://crates.io/crates/optimization_engine) | 0.12.0 (2026-03-31) | Yes | PANOC + ALM/penalty | Projectable sets; eq/ineq via ALM | Yes (parametric re-solve) | Matrix-free | MIT OR Apache-2.0 | Overkill/awkward for QP | **Good** (NMPC-style, not SQP) |
| [slsqp](https://crates.io/crates/slsqp) | 1.0.1 (2026-05-26) | Yes (c2rust from NLopt 2.7.1) | SLSQP (dense SQP) | Nonlinear eq/ineq + bounds | No | Dense | MIT | Awkward (NLP API for a QP) | No (dense, O(n³) LSQ subproblems) |
| [cobyla](https://crates.io/crates/cobyla) | 1.0.2 (2026-05-26) | Yes (c2rust) | COBYLA (derivative-free) | Nonlinear ineq | No | Dense | MIT | No (ignores gradients you have) | No |
| [nlopt](https://crates.io/crates/nlopt) | 0.8.1 (2025-03-26) | No (bundles C nlopt, cmake) | Many (incl. SLSQP, MMA) | Nonlinear eq/ineq + bounds | No | Dense | MIT wrapper; bundled lib has LGPL parts | Awkward | Marginal (dense SLSQP) |
| [ipopt](https://crates.io/crates/ipopt) (ipopt-rs) | 0.6.0 (2024-12-14) | No (links C++ Ipopt) | Primal-dual interior point NLP | Nonlinear `g_L ≤ g(x) ≤ g_U` + bounds | Limited (IPM) | **Sparse** | MIT/Apache wrapper; Ipopt is EPL-2.0 | Overkill | **Proven class** for trajectory opt; heavy build |
| [ripopt](https://crates.io/crates/ripopt) | 0.8.2 (2026-05-22) | Yes (~21.7k lines) | Primal-dual IPM (Ipopt-style) | Nonlinear eq/ineq + bounds | No (IPM) | Dense + sparse (feral LDLᵀ, n+m ≥ 110) | EPL-2.0 | Overkill | **Promising**, young |
| [pounce-rs](https://crates.io/crates/pounce-rs) / [pounce-qp](https://crates.io/crates/pounce-qp) | 0.10.0 (2026-08-11) | Yes | Ipopt-port IPM + **active-set SQP** paths | Nonlinear eq/ineq + bounds; QP path | **Yes** (parametric active-set QP) | Sparse | EPL-2.0 | Unproven | **Only native Rust sparse SQP found**; very new |
| [herculesabqp](https://crates.io/crates/herculesabqp) | 0.1.2 (2026-05-30) | Yes | Accelerated proj. gradient + active-set polish (ProxQP-style) | **Box only** (`l ≤ x ≤ u`) | **Yes** (`PreparedSolver`) | Dense + sparse (ndarray) | BSD-3-Clause | Only if eqs become `l=u` boxes; v0.1 | No (box only) |
| [argmin](https://crates.io/crates/argmin) | 0.11.0 (2025-09-28) | Yes | Framework: (L-)BFGS, Newton, trust region, … | No general linear-constraint QP | n/a | Backend-agnostic (incl. nalgebra) | MIT OR Apache-2.0 | No QP solver | Framework only |
| [good_lp](https://crates.io/crates/good_lp) | 1.15.3 (2026-08-06) | Wrapper | LP/MILP modeler | Linear only — **no quadratic objectives** | n/a | — | MIT | **Not applicable** | Not applicable |
| [totsu](https://crates.io/crates/totsu) | 0.10.2 (2023-01-02) | Yes | First-order conic (LP/QP/QCQP/SOCP/SDP) | eq/ineq/conic | Not documented | Operator-based | Unlicense | Stale (no release in ~3.5 y) | Stale |
| [levenberg-marquardt](https://crates.io/crates/levenberg-marquardt) | 0.15.0 (2025-08-03) | Yes | LM (trust region NLLS) | **Unconstrained** | n/a | Dense, **native nalgebra** | MIT | No constraints | No constraints |
| [lstsq](https://crates.io/crates/lstsq) | 0.8.0 (2026-08-07) | Yes | SVD least squares | None | n/a | Dense, nalgebra | MIT OR Apache-2.0 | Not a QP solver | No |
| [scirs2-optimize](https://crates.io/crates/scirs2-optimize) | 0.6.5 (2026-07-31) | Yes | SciPy-style grab-bag | eq/ineq/bounds (minimize API) | Partial (Bayesian warm-start) | Mixed | Apache-2.0 | Unvetted (see caveats) | Unvetted |

Fit key: (a) = small dense repeated IK QPs; (b) = large sparse SQP trajectory optimization.

---

## Recommended: clarabel

- **What**: "Clarabel Conic Interior Point Solver for Rust / Python" — pure
  Rust implementation of an interior-point solver with a novel homogeneous
  embedding; solves LPs, QPs, SOCPs, SDPs, and exponential/power-cone problems,
  and "handles quadratic objectives without requiring any epigraphical
  reformulation." ([repo](https://github.com/oxfordcontrol/Clarabel.rs),
  [crates.io](https://crates.io/crates/clarabel))
- **Version / activity**: 0.11.1 (2025-06-11); repo pushed 2026-04-13, 585
  stars, 630 commits, ~1.7 M downloads (819 k in the last 90 days). Maintained
  by the Oxford Control group (Goulart et al.).
- **Pure Rust**: yes — no C/C++ toolchain, `cargo add clarabel` just builds.
  MSRV 1.70 ([crates.io API](https://crates.io/api/v1/crates/clarabel)).
- **Problem form**: min ½xᵀPx + qᵀx s.t. Ax + s = b, s ∈ K. Equalities use
  `ZeroConeT(m)`, one-sided inequalities `NonnegativeConeT(m)` — so joint
  limits become two nonnegative rows (or one row per active bound) and pinned
  joints become zero-cone rows
  ([Rust getting-started guide](https://clarabel.org/stable/rust/getting_started_rs/)).
- **Matrix format**: its own `CscMatrix` (compressed sparse column) in
  `clarabel::algebra`. No native nalgebra interop; for a 10×10 dense H you
  convert `DMatrix` → CSC triplets in a small helper (nalgebra is column-major,
  so this is a direct column walk). Not verified: any built-in dense-to-CSC
  convenience constructor — the docs only show sparse construction.
- **Warm starting**: **not supported**. The
  [qpsolvers supported-solvers table](https://qpsolvers.github.io/qpsolvers/supported-solvers.html)
  lists Clarabel with warm-start ✗, and
  [Clarabel.rs issue #59](https://github.com/oxfordcontrol/Clarabel.rs/issues/59)
  is an open feature request for re-solving with updated data. For n≈10 this
  does not matter; for a 100 Hz loop over thousands of variables it would.
- **License**: Apache-2.0.
- **Robotics evidence**: default open-source IPM in CVXPY
  ([cvxpy discussion #2178](https://github.com/cvxpy/cvxpy/discussions/2178));
  covered as a real-time-capable conic option in the legged-robotics QP-solver
  review ([arXiv:2510.21773](https://arxiv.org/html/2510.21773)).
- **Verdict**: best default. Replaces the hand-rolled active-set KKT loop in
  `src/active_set.rs` with a solver that certifies optimality/infeasibility,
  and scales to the sparse trajectory QP subproblems later with the same API.

## Runner-up: osqp (official Rust bindings)

- **What**: Rust wrapper for the C OSQP solver (ADMM operator splitting), form
  min ½xᵀPx + qᵀx s.t. l ≤ Ax ≤ u; equalities via `l_i = u_i`
  ([docs.rs](https://docs.rs/osqp/latest/osqp/),
  [repo under the official osqp org](https://github.com/osqp/osqp.rs)).
- **Version / activity**: 1.0.1 (2025-04-21), tracking OSQP 1.x; repo pushed
  2025-04-21; 42 stars but the upstream C solver is the de-facto standard for
  embedded/robotic MPC ([osqp.org](https://osqp.org/)).
- **FFI**: `osqp-sys` bundles the OSQP C sources as a git submodule and builds
  them with **cmake + cc** at compile time
  ([osqp-sys/build.rs](https://github.com/osqp/osqp.rs/blob/master/osqp-sys/Cargo.toml)) —
  a real toolchain dependency, unlike Clarabel.
- **Warm start / updates**: first-class — `Problem::warm_start`,
  `warm_start_x`, `update_lin_cost`, `update_bounds`, `update_P`, `update_A`
  (same sparsity pattern required)
  ([docs.rs 0.6.3 Problem page](https://docs.rs/osqp/0.6.3/osqp/struct.Problem.html)).
  Exactly the API shape a per-tick IK QP or an SQP inner loop wants.
- **Matrix format**: `CscMatrix<'a>`, accepted via `Into<CscMatrix>`; no
  nalgebra interop, manual conversion as with Clarabel.
- **Caveats**: docs.rs failed to build the 1.0.1 docs (last rendered docs are
  0.6.3) — cosmetic but worth knowing; ADMM gives moderate-accuracy solutions
  and may need tighter tolerances or polishing for stiff IK steps.
- **License**: Apache-2.0 (wrapper and C solver).
- **Verdict**: choose over Clarabel only if warm starting / in-place data
  updates are a demonstrated need, and the cmake build cost is acceptable.

## Other dense-QP options

### quadprog (pure Rust Goldfarb–Idnani)

0.1.0 (2026-07-03), pure Rust port of the classic Goldfarb–Idnani dual
active-set method for strictly convex dense QPs; explicit equality rows
(`meq` first rows) plus inequalities; plain **row-major `&[f64]` slices** (note:
nalgebra `DMatrix` is column-major — transpose when flattening)
([repo](https://github.com/erikbrinkman/quadprog-rs),
[docs.rs `solve_qp`](https://docs.rs/quadprog/0.1.0/quadprog/fn.solve_qp.html)).
Requires positive-definite Q (your damped Hessian JᵀJ + λ²I qualifies), mutates
Q in place, returns Lagrange multipliers and the active-set iteration count.
Technically the closest drop-in for `active_set.rs`, and G–I is exactly the
right algorithm for tiny QPs — **but it is GPL-2.0** (both crate metadata and
repo license), 4 stars, one 0.1.0 release, no warm start. The license alone is
disqualifying if this project ever wants MIT/Apache distribution.

### herculesabqp

0.1.2 (2026-05-30), BSD-3-Clause, pure Rust. Box-constrained QPs only
(l ≤ x ≤ u) via accelerated projected gradient with ProxQP-style active-set
polishing; dense + sparse via ndarray; explicit warm starts and a
`PreparedSolver` for repeated solves with changing bounds
([crates.io readme](https://crates.io/crates/herculesabqp)). Pinned joints
would have to be encoded as `l_i = u_i` boxes, which the box-only formulation
does allow. Early (v0.1.x) and the linked repository
`https://github.com/DKenefake/herculesapgd` returned 404 at research time
(**could not verify** the source repo), so treat as experimental.

### totsu

Pure Rust first-order conic solver family (LP/QP/QCQP/SOCP/SDP), Unlicense —
but the last release is 0.10.2 on **2023-01-02**, ~3.5 years stale
([crates.io](https://crates.io/crates/totsu)). Not recommended for new work.

## Not QP solvers (verified and excluded)

- **good_lp** 1.15.3: LP/MILP modeling front-end over external solvers; the
  README states explicitly "You cannot use it with quadratic functions"
  ([readme](https://crates.io/crates/good_lp)). Not applicable.
- **argmin** 0.11.0: pure-Rust optimization *framework* (line searches, trust
  region, (L-)BFGS, Newton, Gauss-Newton, Nelder-Mead, …) with nalgebra/ndarray
  backends — but its solver list contains **no linearly-constrained QP method**
  ([readme](https://crates.io/crates/argmin),
  [argmin.org](https://argmin-rs.org/)). Useful as scaffolding (observers,
  checkpointing) if you build a custom SQP, not as a solver.
- **levenberg-marquardt** 0.15.0: excellent nalgebra-native trust-region NLLS,
  used across the rust-cv ecosystem — but strictly unconstrained
  ([repo](https://github.com/rust-cv/levenberg-marquardt)). Could replace the
  damped-least-squares core only if constraints were handled elsewhere.
- **lstsq** 0.8.0: minimal `min ‖Ax − b‖` via SVD, nalgebra-based, no
  constraints ([crates.io](https://crates.io/crates/lstsq)).
- **cobyla** 1.0.2: derivative-free linear-approximation method; throws away
  the analytic Jacobians this project has
  ([readme](https://crates.io/crates/cobyla)).
- **proxsuite / ProxQP**: C++ with official Python and Julia bindings only —
  **no Rust crate exists** (crates.io searches for `proxsuite`/`proxqp` return
  nothing relevant;
  [upstream repo](https://github.com/Simple-Robotics/proxsuite) lists no Rust
  interface). Same result for qpOASES, PIQP, and DAQP: no Rust bindings on
  crates.io as of 2026-08-14.
- **osqp-rust** 0.6.2: an unofficial fork of osqp.rs, stale since 2023-03 —
  use the official `osqp` crate instead.
- **scirs2-optimize** 0.6.5 (2026-07-31, Apache-2.0): claims a very large
  SciPy-equivalent surface including constrained minimization. The breadth of
  claims relative to the project's age, and its sprawling auto-generated feel,
  could not be independently vetted — **not verified**; treat with caution
  before depending on it for control-loop numerics.

## The SQP landscape

**Is there a native Rust SQP?** Two, with big asterisks:

1. **slsqp** 1.0.1 ([repo](https://github.com/relf/slsqp)) — Kraft's classic
   SLSQP (the same algorithm behind `scipy.optimize.minimize(method="SLSQP")`
   and NLopt's `LD_SLSQP`), machine-translated from NLopt 2.7.1 C via c2rust
   and hand-cleaned by Rémi Lafage (who maintains it actively alongside
   `cobyla` and the `egobox` optimization ecosystem). License: MIT per the
   repo's LICENSE.md (crates.io shows "non-standard" only because it uses
   `license-file`). It is a genuine SQP for nonlinear eq/ineq constraints —
   but **dense**: fine for ~10–50 variables, not for thousand-variable
   trajectories with band structure.
2. **POUNCE** ([repo](https://github.com/jkitchin/pounce), pushed 2026-08-14,
   EPL-2.0, by John Kitchin) — a pure-Rust port of Ipopt (`pounce-algorithm`
   ports `src/Algorithm`, `pounce-rs` exposes a TNLP trait + builder) whose
   feature-flagged solver paths include an **active-set SQP driver** backed by
   [`pounce-qp`](https://crates.io/crates/pounce-qp), a "sparse parametric
   active-set quadratic programming subproblem solver … the right choice for
   warm-started SQP / MPC subproblem sequences" (its own readme). This is the
   only sparse, warm-started, native-Rust SQP found in the entire registry
   sweep. It is also weeks old at 0.10.0 with the readme describing itself in
   "Phase 5a — feature-complete on correctness" terms — promising, unproven.
   Sibling project **ripopt** 0.8.2
   ([repo](https://github.com/jkitchin/ripopt)) is the same author's
   from-scratch Ipopt-inspired pure-Rust IPM (~21.7k lines, dense LDLᵀ +
   sparse multifrontal LDLᵀ via `feral` for n+m ≥ 110), EPL-2.0.

**What do people actually do for trajectory-scale NLPs in Rust?**

- **Custom SQP over a sparse QP crate** — linearize constraints, quadratize
  the objective, hand the subproblem to Clarabel (accuracy) or OSQP
  (warm-startable ADMM). This matches how the broader real-time robotics
  world structures whole-body/MPC solvers
  ([arXiv:2510.21773 review](https://arxiv.org/html/2510.21773)) and reuses
  whatever QP crate the IK loop already adopted. Both solvers consume CSC, so
  the block-banded trajectory structure costs nothing extra to express.
- **optimization_engine (OpEn)** 0.12.0
  ([repo](https://github.com/alphaville/optimization-engine), pushed
  2026-03-31, MIT OR Apache-2.0) — pure Rust PANOC (proximal averaged
  Newton-type) with ALM/penalty for equality/inequality constraints; built
  explicitly for embedded nonconvex robotics MPC ("A pure Rust framework for
  embedded nonconvex optimization. Ideal for robotics!"), with a documented
  deployment doing autonomous navigation at 20 Hz on an Intel Atom at 15%
  CPU. Gradients come either from hand-written closures or CasADi codegen via
  its Python front-end. It side-steps SQP rather than implementing it: a
  legitimate, robotics-proven route for the trajectory problem, at the cost of
  first-order (not Newton) convergence and a codegen-oriented workflow.
- **ipopt-rs** 0.6.0 ([repo](https://github.com/elrnv/ipopt-rs), pushed
  2026-07-15) — safe bindings to real Ipopt: large-scale sparse interior-point
  NLP, the standard tool for direct-transcription trajectory optimization.
  Costs: you must link the C++ Ipopt library (plus a sparse linear solver such
  as MUMPS/HSL), releases are slow (0.6.0 is 2024-12-14, ~350 recent
  downloads), and Ipopt itself is EPL-2.0.
- **nlopt** 0.8.1 ([repo](https://github.com/adwhit/rust-nlopt)) — bundles and
  statically links C NLopt (cmake required); gives SLSQP/MMA among many
  algorithms; the README itself flags that bundling carries licensing
  implications (NLopt contains LGPL components). Dense; nothing here beats the
  pure-Rust `slsqp` crate for the same algorithm.

**Bottom line for this project**: adopt Clarabel for the IK QP now; when the
trajectory work starts, first try a hand-rolled SQP/Gauss-Newton loop over
Clarabel/OSQP sparse subproblems (you keep full control of the block
structure, and this repo already owns the linearization machinery); reevaluate
POUNCE at that point — if it has matured, it is the only crate offering the
warm-started sparse SQP inner loop off the shelf.

## Things explicitly not verified

- Whether Clarabel's Rust API has any dense-matrix convenience constructor for
  `CscMatrix` (docs only demonstrate sparse construction).
- `herculesabqp`'s source repository (linked repo 404s; only the crates.io
  readme and metadata were inspectable).
- `scirs2-optimize`'s quality claims (test counts, "production-ready") — taken
  from its own readme, not independently checked.
- Runtime performance numbers for any solver on this project's specific 10-var
  QP — no benchmarks were run; the dense-vs-sparse and warm-start reasoning is
  structural, not measured.
- `pounce`/`ripopt` numerical robustness — both are too new to have
  third-party usage reports.
