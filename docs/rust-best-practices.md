# Rust Best Practices

Generic Rust guidance, independent of this project. Project-specific
conventions (crate layout, re-exports, workflow) live in `AGENTS.md` and take
precedence. Baseline beyond this doc: [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
and idioms enforced by `clippy`.

## Error Handling

- Return `Result<T, E>` for anything that can fail at runtime (I/O, parsing,
  singular matrices, bad user input). Propagate with `?`.
- No `unwrap()`/`expect()` on fallible operations in library code. A failed
  LU solve, file read, or lookup is an `Err`, not a panic.
- `expect("...")` is acceptable only for invariants the surrounding code makes
  impossible to violate — and the message must state the invariant, e.g.
  `accepted.expect("active-set loop solves at least once")`. Reads as
  documentation; panics only on a bug.
- `panic!`/`assert!` are for programmer errors (violated preconditions), never
  for expected failure paths.
- Small crates: `Result<T, String>` is fine. Graduate to a dedicated error
  enum (`thiserror`) once callers need to match on failure kinds — not before.
- `Option<T>` models absence; don't overload sentinel values (`-1`, `NaN`,
  empty string) to mean "missing". Exception: interop boundaries where the
  format demands it (e.g. `±inf` for unbounded joint limits) — document it.
- Validate preconditions at the layer where the cause has a name. A duplicate
  constraint caught at parse time reads "duplicate equality constraint for
  joint 'x'"; the same mistake surfacing three layers down reads
  `qp_not_solved: PrimalInfeasible`. Deep layers report symptoms; boundaries
  can report causes.
- When an iterative algorithm can't finish, decide explicitly between error
  and best-effort: fail if no progress was ever made (the problem is wrong),
  but return the best iterate with a `stderr` warning if progress was made —
  and make the warning say *why* it stopped (budget exhausted vs. stalled),
  because the two have different fixes.

## Type Design

- Make invalid states unrepresentable. A closed set of alternatives is an
  `enum`, not a string, bool flag, or integer code — the compiler then forces
  every `match` to handle all cases.
- Related values travel together as a `struct` with named fields, not a tuple.
  `Constraint { joint_index, target, kind }` cannot be mixed up;
  `(usize, f64)` can silently mean two different things at two call sites.
- Prefer `match` over `if`-chains when branching on an enum; exhaustiveness
  checking catches the variant you forgot when a new one is added. Avoid
  `_ =>` catch-alls for the same reason.
- Derive liberally: `Debug` almost always; `Clone`/`Copy` for small plain
  data; `PartialEq` where tests compare values.
- Newtype wrappers (`struct Meters(f64)`) when mixing up two same-typed
  quantities would be a real bug and the call sites are numerous.

## Functions and APIs

- Accept borrows, return owned: parameters as `&[T]`, `&str`, `&T`; return
  `Vec<T>`, `String`, `T`. Let the caller decide about ownership.
- Magic numbers become named `const`s with a doc comment
  (`const LIMIT_TOLERANCE: f64 = 0.01;`), at the narrowest scope that all
  users share.
- A function does one job. If a loop body needs its own explanatory comment
  block, it probably wants to be a named function whose doc comment is that
  block.
- Keep items private by default. `pub` is a commitment; add it when something
  outside the module actually needs the item, remove it when nothing does.
- `impl` blocks hold behavior that needs `self`; free functions are fine for
  operations on borrowed data with no state.

## Control Flow and Iteration

- Iterator chains (`iter().map().collect()`, `position`, `any`, `find`) over
  hand-rolled loops when they read clearer. Index loops remain idiomatic for
  numeric/matrix code where indices are the domain (aligned vectors, matrix
  rows).
- `if let Some(x) = ...` for a single interesting case; `match` once there are
  two.
- Early `return`/`continue`/`break` over nested conditionals. Guard clauses
  first, happy path unindented.

## Floating-Point and Numerical Code

- Max/min searches over data that can contain NaN use `total_cmp`
  (`max_by(|a, b| a.total_cmp(b))`). A plain `>` fold or `f64::max` silently
  skips NaN, misreporting a solver blowup as a clean zero; `partial_cmp`
  + `unwrap` panics on it.
- Clamp before inverse trig: `x.clamp(-1.0, 1.0).acos()`. Float drift pushes
  dot products and normalized traces epsilon outside the domain, and the raw
  call returns NaN, not an error.
- NaN propagates silently — one `0.0 / 0.0` poisons everything downstream and
  often shows up as truncated or empty output, far from the division. Guard
  denominators near zero, and check `is_finite()` at stage boundaries,
  returning `Err` with context (which item, which stage) instead of letting
  the NaN travel.
- Convergence and acceptance thresholds are relative
  (`eps * value.max(1.0)`), not absolute, when magnitudes vary: an absolute
  `1e-12` margin is below one ULP once values reach `~1e4`, so the comparison
  silently stops meaning anything.
- Verify the invariant on the output, not the algorithm's flag. "Converged"
  does not imply the constraint holds (a relaxed subproblem can converge while
  violating the real limit); check the final result directly and warn when it
  misses.
- `let (s, c) = angle.sin_cos();` when both are needed.
- Module-local `type DMat = nalgebra::DMatrix<f64>;` aliases keep heavy
  generic signatures readable; keep them private to the module.

## Documentation and Comments

- `///` doc comments on every public item: what it does, units/conventions,
  error conditions. First line is a standalone summary sentence.
- Comments explain WHY — the sign convention, the numerical edge case, the
  invariant — never restate WHAT the next line does.
- Document deviations from the obvious (e.g. why a Taylor series replaces the
  closed form near a singularity) at the point of deviation.

## Testing

- Unit tests in `#[cfg(test)] mod tests` at the bottom of the module under
  test, using `use super::*`.
- Extract repeated setup into plain helper functions inside the test module —
  no framework fixtures needed.
- Assert with messages that print the offending values:
  `assert!(x < limit, "joint {} out of limits: {}", i, x)`. A bare failed
  `assert!` tells you nothing.
- Test behavior through the public API (a trajectory stays within limits),
  not implementation details (which rows the working set held).
- Check analytic derivatives against central finite differences — including
  the entries claimed to be zero. This catches sign errors and forgotten
  terms that end-to-end tests absorb as "slow convergence".
- Roundtrip through the inverse operation or an independent implementation
  where one exists (`exp(log(T)) == T` via the library's generic `exp`),
  rather than asserting hand-computed expected values.
- Exercise the singular/edge inputs by name: the θ = π branch, the empty
  series, the zero-length segment, duplicate items. These are where numeric
  code actually breaks, and a test at a generic interior point proves nothing
  about them.
- A test that asserts "the penalty reduces X" must first assert the scenario
  produces X without the penalty (`assert!(baseline > 0.05, "scenario induces
  no X")`) — otherwise it passes vacuously when the setup drifts.
- Inline config/data fixtures in the test (a `const` YAML string) instead of
  reading the shipped asset, so the test checks parsing and behavior, not the
  current tuning of a file that changes for unrelated reasons.

## Tooling

- `cargo fmt` before finishing; never hand-format.
- `cargo clippy --all-targets` and take its advice; allow-list with a
  `#[allow]` + reason only when the lint is genuinely wrong.
- `cargo check --all-targets` so `#[cfg(test)]` code compiles too.
- Keep the build warning-clean; a new warning is a defect, not noise.
