# Solver Theory: From Waypoints to a Whole-Body Trajectory

This document explains the math behind the solver — what problem is being
solved, why it is posed as optimization, what every cost and constraint term
means, and why the trajectory pass has to be SQP rather than a single QP. It
is written for readers with robotics background (kinematics, Jacobians,
transforms) but no particular optimization background. Everything here maps
directly onto the code; each section points at the file that implements it.

Contents:

1. [The problem](#1-the-problem)
2. [The robot model: one kinematic chain](#2-the-robot-model-one-kinematic-chain)
3. [Pose error: the SE(3) logarithm](#3-pose-error-the-se3-logarithm)
4. [What a QP is, and what Clarabel does](#4-what-a-qp-is-and-what-clarabel-does)
5. [Stage 1: differential IK as a sequence of QPs](#5-stage-1-differential-ik-as-a-sequence-of-qps)
6. [The whole-trajectory problem](#6-the-whole-trajectory-problem)
7. [The nonholonomic constraint](#7-the-nonholonomic-constraint)
8. [Why this has to be SQP](#8-why-this-has-to-be-sqp)
9. [The QP subproblem, term by term](#9-the-qp-subproblem-term-by-term)
10. [Globalization: making SQP actually converge](#10-globalization-making-sqp-actually-converge)
11. [Convergence, failure modes, and diagnostics](#11-convergence-failure-modes-and-diagnostics)
12. [Symbol and config reference](#12-symbol-and-config-reference)

---

## 1. The problem

Given a sequence of end-effector poses in the world frame (a densified
waypoint path), find a joint trajectory for the **whole robot** — a
differential-drive base and a 6-DoF arm together — such that:

- the end-effector tracks the pose path,
- the base never slips sideways (it is a differential drive: two driven
  wheels, no lateral motion possible),
- joint position and velocity limits hold,
- the motion is smooth,
- optionally, the base prefers driving forward over reversing.

The output is one joint configuration per path pose — a list of **knots**
$q_0, q_1, \dots, q_{N-1}$, each $q_k \in \mathbb{R}^n$.

The pipeline solves this in two stages:

1. **Sequential differential IK** (`src/kinematics.rs`) — solve each pose
   independently, seeding from the previous solution. Fast and reliable, but
   it knows nothing about the no-slip constraint or velocity limits: the base
   is modeled as a free planar joint, so the IK happily slides it sideways.
2. **Whole-trajectory SQP** (`src/trajectory.rs`) — take the IK result as a
   warm start and optimize *all knots at once*, now enforcing the
   differential-drive constraint, velocity limits, and smoothness.

Stage 1 exists to give stage 2 a good starting point. Nonlinear optimization
converges to *a* local solution near where it starts; starting it from a
trajectory that already tracks the end-effector path means the SQP only has
to *repair* the base motion, not discover the whole solution from scratch.

## 2. The robot model: one kinematic chain

The differential-drive base is not modeled as a base at all. The URDF inserts
three planar joints between the world and the base link:

```
world ──(prismatic x)──(prismatic y)──(revolute yaw)── base_link ── arm ── gripper
```

so the entire robot is a single serial chain with configuration

$$
q = [\underbrace{x,\; y,\; \theta}_{\text{base}},\; \underbrace{q_1, \dots, q_6}_{\text{arm}}, \dots] \in \mathbb{R}^n .
$$

This buys two things:

- **One Jacobian for everything.** The chain Jacobian
  $J(q) \in \mathbb{R}^{6 \times n}$ from the `k` crate relates *all* joint
  velocities — base included — to the end-effector spatial velocity. Base
  and arm redundancy resolve jointly instead of being coordinated by hand.
- **A place to put the drive constraint.** Three planar joints are a
  *holonomic* stand-in for the base: they can move in any planar direction.
  The real base cannot. The fix is not in the model but in the optimizer:
  a constraint (Section 7) forbids exactly the motions the real base cannot
  perform.

This "model permissively, constrain in the optimizer" pattern is the core
design decision of the project, and it is why stage 2 must exist.

## 3. Pose error: the SE(3) logarithm

Both stages need a vector-valued error between the current end-effector pose
$T(q) \in SE(3)$ and a goal pose $T_{\text{goal}}$. The code uses the **matrix
logarithm** of the relative transform:

$$
r(q) \;=\; \log\!\left( T_{\text{goal}}\, T(q)^{-1} \right) \;\in\; \mathbb{R}^6,
$$

implemented by `se3_log` / `pose_error_twist` in `src/kinematics.rs`. The
result is a **twist** $[v;\, \omega]$: the constant spatial velocity that,
applied for one unit of time, carries the current pose to the goal.
$\omega$ is the axis–angle vector of the rotation error; $v$ is the
translation error mapped through the inverse left Jacobian of $SO(3)$ (not
the raw translation difference — the two coincide only when the rotation
error is zero).

Why the log instead of "position error + some orientation error"?

- It is a *principled* 6-vector: zero exactly when the poses match, smooth
  in a neighborhood of the goal, and its Jacobian with respect to $q$ is the
  chain's geometric Jacobian (to first order at small error). That is
  precisely what a Newton-type method needs.
- It weights rotation and translation in one consistent object, rather than
  gluing two heuristic errors together.

Two numerical details worth knowing (they are in comments in the code too):

- The rotation log is extracted via quaternions (Shepperd's method) rather
  than $\arccos$ of the trace, because the trace route returns NaN when
  floating-point drift pushes the trace argument outside $[-1, 1]$ near
  $180°$ rotations.
- The coefficient in the inverse left Jacobian is $0/0$ at $\theta = 0$, so
  a Taylor series is used below $\theta = 10^{-6}$.

Ordering convention: this codebase uses $[v;\, \omega]$ (linear first) to
match `k::jacobian`. Modern Robotics and much of the literature use
$[\omega;\, v]$ — reorder when cross-checking formulas.

## 4. What a QP is, and what Clarabel does

A **quadratic program** (QP) is an optimization problem of the form

$$
\min_{x \in \mathbb{R}^n}\;\; \tfrac{1}{2} x^\top P x + q^\top x
\quad \text{s.t.} \quad
A_{\text{eq}}\, x = b_{\text{eq}},\qquad
A_{\text{in}}\, x \le b_{\text{in}},\qquad
l \le x \le u,
$$

with $P$ symmetric **positive semidefinite** (all its eigenvalues
$\ge 0$; equivalently the objective is bowl-shaped, never saddle-shaped).
This is the workhorse of real-time robotics optimization because it is
**convex**: it has no spurious local minima, a solver either finds the global
optimum or proves there is no feasible point, and it does so in a predictable
number of iterations. All the hard structure — which limits are active,
which constraints bind — is handled inside the solver.

This project uses [Clarabel](https://clarabel.org), an **interior-point**
solver. Two facts about interior-point methods matter for reading this code:

- **Conic form.** Clarabel wants constraints as $Ax + s = b,\; s \in K$
  where $K$ is a product of cones. `src/qp.rs` does the translation:
  equality rows go into the *zero cone* ($s = 0$), and inequality rows and
  finite box bounds are all rewritten as one-sided rows in the *nonnegative
  cone* ($s \ge 0$), e.g. $x_i \le u_i \Rightarrow e_i^\top x + s = u_i$.
- **Feasible to tolerance, not exactly.** An interior-point solution
  satisfies constraints to $\sim 10^{-8}$–$10^{-10}$, not to the last bit.
  That is why both solver loops clamp the updated joints to their limits
  after every step: downstream code must be able to rely on hard limits
  exactly.

`qp::solve` is the single seam through which every optimization in the
project passes — the differential IK calls it once per iteration with an
$n \approx 10$ variable problem, and the SQP calls it once per iteration
with an $N \cdot n$ (hundreds to ~2000) variable problem.

## 5. Stage 1: differential IK as a sequence of QPs

The classical starting point for IK is resolved-rate motion control: at the
current $q$, compute the pose error twist $r$ and take the least-squares step

$$
\Delta q = J^{+} r \qquad \text{(Jacobian pseudo-inverse)}.
$$

This explodes near kinematic singularities, where $J$ loses rank and
$J^{+}$ blows up. The standard fix is **damped least squares** (DLS):

$$
\Delta q = \arg\min_{\Delta q}\; \tfrac{1}{2}\, \| J \Delta q - r \|^2 + \tfrac{1}{2}\, \lambda^2 \| \Delta q \|^2
\;=\; (J^\top J + \lambda^2 I)^{-1} J^\top r,
$$

which trades a little tracking accuracy for bounded, well-conditioned steps
($\lambda$ is `damping_factor` in the config).

Expanding the DLS objective shows it is *already* a QP:

$$
\tfrac{1}{2}\, \Delta q^\top \underbrace{(J^\top J + \lambda^2 I)}_{P} \Delta q
\;-\; \underbrace{(J^\top r)}_{-q_{\text{lin}}}{}^\top \Delta q \;+\; \text{const}.
$$

Solving it *as* a QP instead of by linear solve costs a little speed and
buys the ability to add constraints that a matrix inverse cannot express:

$$
\begin{aligned}
\min_{\Delta q}\;\; & \tfrac{1}{2} \Delta q^\top (J^\top J + \lambda^2 I)\, \Delta q - (J^\top r)^\top \Delta q \\
\text{s.t.}\;\; & A_{\text{eq}}\, \Delta q = q^{\text{pin}} - q &&\text{(pinned joints, e.g. elbow posture)}\\
& q_{\text{lower}} - q \;\le\; \Delta q \;\le\; q_{\text{upper}} - q &&\text{(URDF joint limits).}
\end{aligned}
$$

Note the change of variables: the QP optimizes the *step* $\Delta q$, so
absolute constraints on $q$ become constraints on $\Delta q$ shifted by the
current iterate ("re-anchoring"). This pattern repeats in the SQP.

The IK loop (`differential_ik` in `src/kinematics.rs`) is then simply:
compute $r$; if $\|r\|$ is below threshold, stop; otherwise solve the QP,
apply $\Delta q$, clamp to limits, recompute FK, repeat. Each path pose is
seeded with the previous pose's solution, which keeps the resulting knot
sequence continuous.

Historically this replaced a hand-rolled active-set loop (deciding manually
which joint limits are "active" and pinning them): the interior-point solver
does that reasoning internally and correctly, including dropping a limit
when the goal pulls the joint back inside.

## 6. The whole-trajectory problem

Stage 2 optimizes the stacked variable

$$
Q = \begin{bmatrix} q_0 \\ q_1 \\ \vdots \\ q_{N-1} \end{bmatrix} \in \mathbb{R}^{Nn},
$$

one block per knot. The problem it would *like* to solve is the nonlinear
program (NLP):

$$
\begin{aligned}
\min_{Q}\;\;
& \sum_{k=0}^{N-1} \tfrac{1}{2}\, w_{\text{ee}} \| r_k(q_k) \|^2
&& \text{end-effector tracking (soft mode)} \\
&+ \sum_{k=0}^{N-2} \tfrac{1}{2}\, w_{\text{smooth}} \| q_{k+1} - q_k \|^2
&& \text{smoothness} \\
&+ \sum_{k=0}^{N-2} \tfrac{1}{2}\, w_{\text{back}} \min(s_k(Q), 0)^2
&& \text{backward-motion penalty} \\[4pt]
\text{s.t.}\;\;
& c_k(Q) = 0, \quad k = 0,\dots,N\!-\!2
&& \text{no lateral slip (Section 7)} \\
& |q_{k+1} - q_k| \le v_{\max}\, \Delta t \;\; \text{elementwise}
&& \text{joint velocity limits} \\
& q_{\text{lower}} \le q_k \le q_{\text{upper}}
&& \text{joint position limits} \\
& \big[\; r_k(q_k) = 0 \;\big]
&& \text{(hard tracking mode only).}
\end{aligned}
$$

Here $r_k$ is the SE(3) pose error of knot $k$ against its goal pose,
$c_k$ is the slip residual of interval $k$, and $s_k$ is the base's forward
progress over interval $k$. Velocities are finite differences over the knot
spacing $\Delta t$ — the trajectory is a discrete sequence, so
$\dot q \approx (q_{k+1} - q_k)/\Delta t$.

Two tracking modes:

- **Soft** (`ee_tracking: soft`): tracking is a weighted cost. The solver
  may trade tracking error against constraint satisfaction — which is
  usually what you want, since exact tracking may be *impossible* once the
  base can't slide sideways.
- **Hard** (`ee_tracking: hard`): tracking becomes six equality constraints
  per knot. Exact when feasible; genuinely infeasible when the path demands
  motion the constrained base cannot deliver, in which case the solver
  reports an error rather than silently degrading.

## 7. The nonholonomic constraint

### What "nonholonomic" means

A **holonomic** constraint restricts the *configurations* a system can
occupy: $g(q) = 0$ removes a dimension from configuration space. A
**nonholonomic** constraint restricts *velocities* without restricting
configurations: it is a constraint of the form $a(q)^\top \dot q = 0$ that
cannot be integrated into any $g(q) = 0$.

A differential-drive base is the canonical example. Its wheels can produce a
forward speed $v$ and a turn rate $\omega$, but no sideways velocity:

$$
\dot x = v \cos\theta, \qquad
\dot y = v \sin\theta, \qquad
\dot\theta = \omega .
$$

Eliminating $v$ from the first two equations gives the constraint in
velocity form:

$$
\boxed{\;\sin\theta \cdot \dot x \;-\; \cos\theta \cdot \dot y \;=\; 0\;}
$$

i.e. *the velocity component perpendicular to the heading is zero*. This is
the projection of $(\dot x, \dot y)$ onto the base's lateral axis. Crucially,
the base can still *reach* any $(x, y, \theta)$ — you can parallel-park a
differential drive with back-and-forth maneuvers — so no configuration
constraint $g(q) = 0$ exists. That is exactly why the constraint could not
be baked into the kinematic model (which describes configurations) and must
be imposed on the *trajectory* (which contains the velocity information as
differences between knots).

### Discretization: the midpoint heading

The trajectory is discrete, so the velocity constraint must be discretized.
Over one knot interval, write $\Delta x = x_{k+1} - x_k$,
$\Delta y = y_{k+1} - y_k$, and define the **midpoint heading**
$\bar\theta = \tfrac{1}{2}(\theta_k + \theta_{k+1})$. The discrete slip
residual is

$$
c_k \;=\; \sin\bar\theta \cdot \Delta x \;-\; \cos\bar\theta \cdot \Delta y ,
$$

implemented by `slip_residual` in `src/diagnostics.rs`. Geometrically,
$-c_k$ is the component of the displacement along the base's lateral
($+y$ body) axis evaluated at the midpoint heading — so $c_k > 0$ means the
base slipped to its *right*, $c_k < 0$ to its left, and $c_k = 0$ means the
displacement chord is parallel to the midpoint heading.

Why the midpoint rather than $\theta_k$ or $\theta_{k+1}$? If the base moves
with constant $v$ and $\omega$ across the interval, it traces a circular
arc, and the chord of a circular arc points exactly along the *average* of
the start and end headings. So the midpoint rule is not just a second-order
finite-difference nicety — it is **exact** for the constant-twist motions
the interval is meant to represent, and it treats the two endpoint knots
symmetrically (using $\theta_k$ alone would bias every interval toward its
start).

The same construction with the longitudinal axis gives the **forward
progress**

$$
s_k \;=\; \cos\bar\theta \cdot \Delta x \;+\; \sin\bar\theta \cdot \Delta y ,
$$

which is the signed distance driven along the heading ($s_k = v\,\Delta t$
when the no-slip constraint holds; negative $s_k$ is reverse motion). It
feeds the backward-motion penalty.

### Linearization

The SQP (next section) needs $c_k$ and its gradient with respect to the six
base coordinates it touches, $(x_k, y_k, \theta_k, x_{k+1}, y_{k+1},
\theta_{k+1})$. Straightforward calculus, using
$\partial\bar\theta / \partial\theta_k = \partial\bar\theta /
\partial\theta_{k+1} = \tfrac{1}{2}$:

$$
\frac{\partial c_k}{\partial x_k} = -\sin\bar\theta, \quad
\frac{\partial c_k}{\partial y_k} = \cos\bar\theta, \quad
\frac{\partial c_k}{\partial x_{k+1}} = \sin\bar\theta, \quad
\frac{\partial c_k}{\partial y_{k+1}} = -\cos\bar\theta,
$$

$$
\frac{\partial c_k}{\partial \theta_k} =
\frac{\partial c_k}{\partial \theta_{k+1}} =
\tfrac{1}{2}\left( \cos\bar\theta \cdot \Delta x + \sin\bar\theta \cdot \Delta y \right).
$$

All other partials are exactly zero (the constraint involves only the base
coordinates), so the constraint Jacobian row is stored as six sparse
(index, value) pairs. `nonholonomic_linearization` in `src/trajectory.rs`
implements this, and a unit test checks every partial against central finite
differences — cheap insurance, since a sign error here would still "solve"
but converge to subtly wrong trajectories.

## 8. Why this has to be SQP

Look again at the NLP in Section 6 and ask: can this be a single QP?

A QP needs a quadratic objective and **linear** constraints. The NLP fails
this on two counts:

1. **The no-slip constraint is nonlinear in the decision variables.**
   $c_k = \sin\bar\theta\,\Delta x - \cos\bar\theta\,\Delta y$ multiplies
   trigonometric functions of some variables ($\theta$) by differences of
   others ($\Delta x, \Delta y$). Worse, the feasible set it defines is
   **nonconvex**: the base can get from A to B along many qualitatively
   different no-slip motions (drive forward and turn; reverse; turn in
   place first), and the straight-line interpolation between two feasible
   trajectories is generally *not* feasible. No convex problem can have a
   nonconvex feasible set, so no exact single-QP formulation exists.
2. **Forward kinematics is nonlinear.** The tracking residual $r_k(q_k)$
   involves the chain FK — products of rotations depending on every joint
   angle. Quadratic cost in $r$ is *not* quadratic in $q$.

The standard remedy is **Sequential Quadratic Programming**: solve a
sequence of QPs, each one a local model of the NLP at the current iterate,
and use each QP's solution as a step direction. At iterate $Q^{(i)}$:

1. **Linearize every nonlinear constraint** around $Q^{(i)}$:
   $c(Q^{(i)} + \Delta) \approx c(Q^{(i)}) + \nabla c^\top \Delta$, and
   constrain the model to zero: $\nabla c^\top \Delta = -c(Q^{(i)})$. The
   right-hand side is the *current violation*; the QP step is asked to
   cancel it.
2. **Build a convex quadratic model of the cost** (next paragraph).
3. **Solve the QP** for the step $\Delta Q$, restricted to a trust region
   (Section 10).
4. **Accept or reject** the step based on a merit function, adapt the trust
   region, repeat until the step is negligible.

The fixed point of this iteration satisfies the *nonlinear* constraints
exactly — see Section 10 for why — even though each subproblem only ever
sees linearizations.

**Gauss-Newton Hessian.** For the tracking cost
$\tfrac{1}{2} w \|r(q)\|^2$, the true Hessian is
$w (J^\top J + \sum_i r_i \nabla^2 r_i)$. Gauss-Newton drops the
second-derivative term and uses just

$$
H \approx w\, J^\top J .
$$

Three reasons, all load-bearing:

- $J^\top J$ is **always positive semidefinite**, so the QP stays convex no
  matter how contorted the configuration is. The dropped term can be
  indefinite.
- It needs only first derivatives — the same Jacobian FK already provides.
  Second derivatives of chain FK are expensive and messy.
- The dropped term is weighted by the residual $r$, so near a good solution
  ($r \to 0$) the approximation becomes exact, and convergence is fast
  precisely where it matters.

This is why the code calls itself "Gauss-Newton SQP": SQP structure
(linearized constraints, QP subproblems, merit function, trust region) with
Gauss-Newton Hessians for the nonlinear least-squares cost terms.

**Why not something else?** For completeness: a nonlinear interior-point
solver (IPOPT-style) applied to the full NLP would also work but gives up
the warm-starting friendliness and the small, auditable subproblem
structure; unconstrained penalty methods (just add $\mu c^2$ to the cost)
need $\mu \to \infty$ for exact constraint satisfaction and become
ill-conditioned along the way. SQP with exact-penalty globalization gets
exact feasibility at finite penalty and reuses the same QP seam the IK
already built.

## 9. The QP subproblem, term by term

Everything in this section is `sqp_step` in `src/trajectory.rs`. The
decision variable is the stacked step
$\Delta Q = [\Delta q_0; \dots; \Delta q_{N-1}] \in \mathbb{R}^{Nn}$, and
the subproblem has exactly the generic form of Section 4:

$$
\min_{\Delta Q}\; \tfrac{1}{2} \Delta Q^\top P\, \Delta Q + q_{\text{lin}}^\top \Delta Q
\quad \text{s.t.} \quad
A_{\text{eq}} \Delta Q = b_{\text{eq}},\;\;
A_{\text{in}} \Delta Q \le b_{\text{in}},\;\;
l \le \Delta Q \le u .
$$

### Cost: end-effector tracking (soft mode)

Per knot, Gauss-Newton on $\tfrac{1}{2} w_{\text{ee}} \| r_k \|^2$:

$$
P \mathrel{+}= w_{\text{ee}}\, J_k^\top J_k \;\;\text{(block $k$,$k$)},
\qquad
q_{\text{lin}} \mathrel{-}= w_{\text{ee}}\, J_k^\top r_k \;\;\text{(block $k$)}.
$$

The linear term is the (negated) gradient: it pulls each knot in the
direction that reduces its pose error. Because each knot's tracking term
touches only its own block, $P$ gets block-diagonal contributions here.

### Cost: smoothness

The smoothness cost $\sum_k \tfrac{1}{2} w_s \|d_k\|^2$ with
$d_k = q_{k+1} - q_k$ is *already quadratic* in $Q$, so its "model" is
exact. Substituting $q \to q + \Delta$ gives, per interval, the $2\times 2$
block pattern

$$
P \mathrel{+}= w_s \begin{bmatrix} I & -I \\ -I & I \end{bmatrix}
\;\;\text{(blocks $k$ and $k{+}1$)},
\qquad
q_{\text{lin}}\text{: } -w_s d_k \text{ on block } k,\;\; +w_s d_k \text{ on block } k{+}1 .
$$

Summed over intervals this assembles the standard first-difference
(graph Laplacian) operator: it penalizes consecutive knots moving apart and
couples neighboring blocks of $P$, which is what lets a constraint violation
at one interval redistribute motion across the whole trajectory instead of
being absorbed locally.

### Cost: backward-motion penalty

The penalty $\tfrac{1}{2} w_b \min(s_k, 0)^2$ is one-sided: zero whenever
the base progresses forward, quadratic in the reverse distance otherwise.
It is treated Gauss-Newton style with an active set read off the current
iterate: intervals with $s_k < 0$ contribute

$$
P \mathrel{+}= w_b\, g_k g_k^\top, \qquad
q_{\text{lin}} \mathrel{+}= w_b\, s_k\, g_k,
$$

where $g_k = \nabla s_k$ is the six-entry gradient from
`forward_progress_linearization`; forward intervals contribute nothing. The
cost is $C^1$ at $s = 0$ (value and slope both vanish), so intervals
flipping in and out of the active set between iterations do not create
kinks that would confuse the merit-function bookkeeping.

This is a *penalty*, not a constraint, by design: reversing is sometimes
necessary (that's how a differential drive escapes some geometries), so the
weight expresses a preference with a price, and `backward_weight: 0` turns
it off entirely.

### Cost: damping

$$
P \mathrel{+}= \lambda_{\text{damp}} I .
$$

$J^\top J$ is singular wherever the arm is at a singularity or the problem
is redundant (it always is — $Nn$ variables, far fewer constraints), and
the smoothness Laplacian has a nullspace (uniform shifts of all knots).
The damping term makes $P$ strictly positive definite, so the QP always has
a unique minimizer. This is the Levenberg–Marquardt idea transplanted into
the subproblem; it also shrinks step length, cooperating with the trust
region.

### Equalities: no-slip rows (and hard-mode tracking rows)

One row per interval, the linearization from Section 7:

$$
\nabla c_k^\top \Delta Q = \alpha \,(-c_k),
$$

six nonzeros per row. The right-hand side asks the step to cancel the
current slip; $\alpha \in (0, 1]$ is the relaxation factor explained in
Section 10.

In hard tracking mode, six more rows per knot:

$$
J_k\, \Delta q_k = \alpha\, r_k ,
$$

demanding the step cancel the pose error exactly. (Sign note: the residual
convention here is "twist *toward* the goal", so the right-hand side is
$+r_k$, and a test pins this down — a flipped sign would drive the arm away
from the goal while still looking superficially like it "solves".)

### Inequalities: velocity limits with a progressive budget

The desired constraint per interval and joint is
$|d_k^j + (\Delta q_{k+1}^j - \Delta q_k^j)| \le v_{\max} \Delta t$, split
into two one-sided rows on $\Delta Q$ (absolute values are not linear; a
pair of linear inequalities is).

The wrinkle is feasibility at the warm start. On a sharp corner, the
sequential IK can produce consecutive knots whose joint difference *already*
exceeds $v_{\max}\Delta t$ by more than the trust region allows fixing in
one step. Handing Clarabel a constraint that no in-trust-region step can
satisfy yields an infeasible QP and no progress at all. So the actual bound
per row is the **progressive budget**

$$
b_k^j = \max\!\big( |d_k^j| - \rho,\;\; v_{\max}\, \Delta t \big),
$$

with $\rho$ the trust-region radius: wherever the limit is reachable this
iteration, it is enforced outright; wherever it is not, the constraint
demands one trust-region-step of improvement per iteration and tightens as
the iterates approach the limit. The merit function (Section 10) always
judges against the *real* limit, so this relaxation cannot silently leak a
violation into an "accepted" trajectory — and a final check after the loop
warns if the budget never fully tightened within the iteration limit.

### Box: joint limits ∩ trust region

Per variable, the same re-anchoring as the IK QP, intersected with the
trust region:

$$
\max\!\big(q_{\text{lower}}^j - q_k^j,\; -\rho\big)
\;\le\; \Delta q_k^j \;\le\;
\min\!\big(q_{\text{upper}}^j - q_k^j,\; \rho\big).
$$

The trust region is thus an $\infty$-norm ball — a box — which costs
nothing extra in a QP (box bounds are the cheapest constraint a conic solver
handles), unlike a Euclidean-ball trust region which would need a
second-order cone. Warm-start knots are clamped into limits *before* the
loop, because a variable with $l > u$ makes the QP infeasible with an
opaque error rather than pointing at the actual cause.

## 10. Globalization: making SQP actually converge

Solving QP subproblems is the easy half of SQP. The hard half is
**globalization**: making sure the sequence of steps actually descends to a
solution instead of oscillating or diverging. A pure "always take the full
QP step" iteration demonstrably fails here — on L-shaped test paths it
limit-cycles, taking full-radius steps that hop back and forth across the
no-slip manifold forever. `optimize` in `src/trajectory.rs` layers three
standard mechanisms.

### The ℓ1 exact-penalty merit function

Constrained steps present a dilemma: a step may reduce cost but increase
constraint violation, or vice versa. Which is "better"? A **merit function**
scalarizes the comparison:

$$
\phi(Q) \;=\; \underbrace{\text{cost}(Q)}_{\text{tracking + smoothness + backward}}
\;+\; \nu \sum_k |c_k(Q)|
\;+\; \nu \sum_{k,j} \max\!\big(|d_k^j| - v_{\max}\Delta t,\; 0\big)
\;\;\left[+\; \nu \sum_k \|r_k\|_1 \;\text{in hard mode}\right],
$$

with penalty weight $\nu = 10^4$. The $\ell_1$ (absolute-value) penalty has
a property the more obvious quadratic penalty lacks: it is **exact**. For
any $\nu$ larger than the biggest Lagrange multiplier of the NLP, the
unconstrained minimizers of $\phi$ *are* the constrained solutions —
violations are driven to exactly zero at finite $\nu$, not merely made
small. The weight is a fixed constant here (marked as a deliberate
simplification in the code); adaptive $\nu$ update rules exist but haven't
been needed.

Important structural point: **the QP never sees the merit function.** The
QP works with linearized constraints and the smooth cost model; the merit
function is used only *afterward*, evaluated on the true nonlinear
quantities, to decide whether the proposed step actually improved matters.

### Trust region with accept/reject

The QP's quadratic-model-plus-linearization is only trustworthy near the
current iterate — that is what the trust region radius $\rho$ encodes. The
adaptation loop is the classic one:

- Solve the QP with radius $\rho$; form the candidate
  $Q^{+} = \text{clamp}(Q + \Delta Q)$.
- If $\phi(Q^{+}) < \phi(Q)$ (with a small relative margin — an absolute
  margin of $10^{-12}$ would be below floating-point resolution at
  hard-mode merit magnitudes of $\sim 10^4$): **accept**, and grow
  $\rho \leftarrow \min(2\rho, \rho_{\max})$.
- Otherwise **reject**: the model over-promised at this radius; keep $Q$
  and shrink $\rho \leftarrow \rho / 2$. If $\rho$ collapses below
  $\rho_{\max}/64$, the merit landscape offers no improvement at any usable
  radius — report a stall and return the best iterate.

Convergence is declared when a *full-target* ($\alpha = 1$) step has
$\|\Delta Q\|_\infty$ below `convergence_step_norm`: the QP, asked for
complete constraint correction, answered "you barely need to move" — the
definition of a fixed point.

### Equality relaxation ($\alpha$-homotopy)

Sometimes the subproblem is infeasible outright: the linearized rows demand
$\nabla c^\top \Delta = -c$, but no step within the trust region and
velocity rows can cancel *all* of the violation at once (typical near
kinematic singularities or right after a warm start with large slip). Note
that shrinking the trust region can never fix this — a smaller box only
shrinks the feasible set further. Instead the right-hand sides are scaled:

$$
\nabla c^\top \Delta = \alpha (-c), \qquad \alpha \in \{1, \tfrac12, \tfrac14, \tfrac18, \tfrac1{16}\},
$$

halving $\alpha$ until the QP becomes feasible (in hard mode the EE rows
scale with the same $\alpha$). This asks for *partial* correction per step
— a homotopy toward feasibility, not permission to slip: at the SQP fixed
point, $\Delta = 0$ solving the QP forces $\alpha \cdot c = 0$, hence
$c = 0$ for any $\alpha \neq 0$. Relaxed steps also never count as
converged, precisely because they only satisfied a scaled-down target. If
even $\alpha = 1/16$ is infeasible, the iterate is genuinely stuck and the
error is surfaced (before any accepted step) or the run degrades to
best-effort with a stderr warning (after progress has been made).

### Why the fixed point satisfies the true constraints

It is worth spelling out the punchline, because it is the reason the whole
construction is sound. Suppose the iteration has converged: the QP at the
current iterate returns $\Delta Q \approx 0$. Then the equality rows read
$\nabla c^\top \cdot 0 = -\alpha\, c$, i.e. $c = 0$: **zero slip, measured
by the true nonlinear residual**, not the linearization. The linearization
error, which is second-order in the step, vanishes with the step itself.
The same argument covers the hard-mode tracking rows ($r_k = 0$) and, since
the velocity rows' progressive budget equals the true budget wherever
feasible, the velocity limits. The tests confirm this: converged
trajectories show slip residuals below $10^{-6}$.

## 11. Convergence, failure modes, and diagnostics

The loop distinguishes its endings deliberately:

| Ending | Meaning | Behavior |
|---|---|---|
| Step norm below threshold at $\alpha = 1$ | Fixed point: constraints hold, cost locally optimal | Return, silent |
| Iteration budget exhausted | More iterations would likely help | Return last iterate, stderr warning |
| Trust region collapsed | Merit-local point the model can't improve | Return last iterate, stderr warning |
| QP infeasible before any accepted step | Problem is wrong (e.g. hard mode conflicts with base kinematics) | Error |
| QP solver stalls mid-run after progress | Numerical, not modeling, failure | Return best iterate, stderr warning |

Hard mode deserves its warning label: exact pose equality at every knot,
combined with the no-slip constraint, can simply have no solution — the
constrained base may be unable to place the arm's shoulder where the exact
pose requires at that knot spacing. Soft mode turns that conflict into a
graceful trade-off, which is why it is the default and why the hard-mode
error message suggests trying soft.

Because the SQP can return best-effort results, the pipeline never trusts
it blindly: `src/diagnostics.rs` recomputes end-effector pose errors and
per-interval slip residuals on the *final* trajectories (both IK and SQP)
and reports the worst offenders, using the same `slip_residual` the
constraint is built from. A velocity-limit check after the loop likewise
warns if the progressive budget never fully tightened. Diagnostics are
read-only — they never feed back into the solve — so they function as an
independent audit of what the optimizer claims.

Numerical footnotes that shaped the code, collected:

- Clarabel's feasibility tolerance is tightened to $10^{-10}$ so pinned
  joints hold to the $10^{-9}$ the tests assert; tightening further to
  $10^{-12}$ makes Clarabel stall (`InsufficientProgress`) on the large
  dense trajectory subproblems.
- Every constraint gradient in the SQP is verified against central finite
  differences in unit tests. For hand-derived linearizations this is the
  single highest-value test you can write.
- Steps are clamped to joint limits after every accept, because
  interior-point feasibility is tolerance-level, not exact.

## 12. Symbol and config reference

| Symbol | Meaning | Config key / code |
|---|---|---|
| $q_k \in \mathbb{R}^n$ | Joint configuration at knot $k$ (base $x, y, \theta$ + arm) | — |
| $Q \in \mathbb{R}^{Nn}$ | All knots stacked; SQP decision variable is the step $\Delta Q$ | — |
| $r_k \in \mathbb{R}^6$ | Pose-error twist $[v; \omega]$ at knot $k$ | `pose_error_twist` |
| $J_k \in \mathbb{R}^{6 \times n}$ | Chain geometric Jacobian at knot $k$ | `Kinematics::jacobian` |
| $c_k$ | Lateral slip residual, interval $k$ | `slip_residual` |
| $s_k$ | Forward progress, interval $k$ | `forward_progress_linearization` |
| $\bar\theta$ | Midpoint heading $\tfrac12(\theta_k + \theta_{k+1})$ | — |
| $w_{\text{ee}}$ | Soft tracking weight | `ee_weight` |
| $w_s$ | Smoothness weight | `smoothness_weight` |
| $w_b$ | Backward-motion weight (0 = allow reversing) | `backward_weight` |
| $\lambda_{\text{damp}}$ | Hessian damping (Levenberg regularization) | `damping` |
| $\lambda$ | IK damped-least-squares factor | `damping_factor` |
| $v_{\max}$ | Per-joint velocity limit | `max_joint_velocity` |
| $\Delta t$ | Knot spacing in time | `dt` |
| $\rho$ | Trust-region radius (adapted; config value is the max) | `trust_region` |
| $\alpha$ | Equality-relaxation scale, $\{1, \dots, 1/16\}$ | internal |
| $\nu$ | Merit penalty weight ($10^4$, fixed) | `MERIT_PENALTY` |

Further reading, matched to this document's structure: Nocedal & Wright,
*Numerical Optimization* (ch. 18: SQP, merit functions, trust regions;
ch. 10: Gauss-Newton); Lynch & Park, *Modern Robotics* (ch. 3: SE(3) logs —
note the $[\omega; v]$ ordering; ch. 13: nonholonomic wheeled robots);
Siciliano et al., *Robotics: Modelling, Planning and Control* (ch. 3.7:
damped-least-squares IK).
