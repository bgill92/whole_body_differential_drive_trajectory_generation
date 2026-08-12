# Project Conventions

## Rust Style

Generic Rust best practices (error handling with `Result`, enums for closed
sets, API and testing idioms) live in
[docs/rust-best-practices.md](docs/rust-best-practices.md). Follow them for
all Rust code in this repo; the project-specific rules below take precedence
where they overlap.

## Rust Edition

**Use edition = "2024" in Cargo.toml** - This is the latest stable Rust edition. Do not change to 2021 or other editions without explicit request.

## Crate Layout

This package has both a library target (`src/lib.rs`, crate name `wbdd`) and a
binary target (`src/main.rs`).

**Never write `mod lib;` in `main.rs`.** Cargo already builds `src/lib.rs` as
its own library target. Declaring it as a module compiles the file a second
time inside the binary crate, producing two independent copies of every type —
same source, incompatible types if they ever meet. rustc warns
`special_module_name`. The binary depends on the library by crate name:

```rust
// src/main.rs
use wbdd::{Config, SolverConfig};
```

The same rule applies to `mod main;` and to any other target reaching into
`lib.rs` by path. Integration tests in `tests/` and downstream crates only ever
see the library target, so anything they need must be public API of `wbdd`.

## Modules and Re-exports

Keep the crate root's public surface flat. Submodules stay private; lift the
types callers need with a re-export at the top of `lib.rs`:

```rust
// src/lib.rs
mod configs;
mod kinematics;

pub use crate::configs::{Config, DifferentialIkConfig, Pose, SolverConfig};
pub use crate::kinematics::{Kinematics, differential_ik, se3_log};
```

Prefer `self::` (or `crate::`) on these paths rather than a bare `use configs::`
— the explicit prefix disambiguates a local module from an external crate of
the same name.

**One public path per type.** `mod foo;` (private) plus a root `pub use` gives
callers exactly one way to name a type: `wbdd::Config`. Writing `pub mod foo;`
*and* re-exporting from it publishes two paths for everything
(`wbdd::configs::Config` and `wbdd::Config`), which then both have to be kept
working. Pick one style — this crate uses private modules with root re-exports.

**Re-export a type from the module that defines it.** Do not `pub use` someone
else's type onward from an unrelated module: `kinematics` importing the config
structs should write a plain `use crate::configs::{...}`, not `pub use`. A
`pub use` there, combined with the root re-export, would give
`DifferentialIkConfig` three public paths and imply the type belongs to
`kinematics`. Plain `use` for consumption, `pub use` only for publishing what
you own.

Re-export deliberately. Exporting a name commits the crate to its signature, so
do not add one to the list until something outside the defining module actually
needs it.

Give a module its own file (`src/configs.rs`) once it outgrows a screen or two.
Inline `mod foo { ... }` blocks are fine for small, tightly-coupled helpers.
Module names are singular by convention (`config`, not `configs`) — this crate
is inconsistent on that point; do not spread it further.

Unit tests live in a `#[cfg(test)] mod tests` at the bottom of the module they
exercise, reaching the code under test through `use super::*`.

## Before Finishing

Run, in order:

```bash
cargo fmt
cargo check --all-targets
cargo test
```

`cargo fmt` is not optional — do not hand-format `use` lists or wrap lines
yourself. rustfmt sorts names inside braces with uppercase before lowercase
(`DifferentialIkConfig, Kinematics, SolverConfig, differential_ik, se3_log`)
and splits a brace list across lines once it passes the width limit.

Use `--all-targets`. Plain `cargo check` skips `#[cfg(test)]` code, so a test
module that no longer compiles goes unnoticed.

The crate builds warning-clean; a new `unused`/`dead_code` warning means either
the code is unreachable or it is wired up wrong.

## Git Workflow

**Never commit or push without explicit request.** Stage changes and wait for
clear user instruction. This repo is your working directory, not an auto-pushed
branch.

Known failure: `configs::tests::config_parses` asserts a goal translation of
`(-1, -1, 1)` while `assets/config.yaml` specifies `[1.0, 1.0, 1.0]`. This
predates the library split. Do not "fix" it by editing whichever side is
convenient — it is unresolved which one is correct.

## Matrix Algebra

**Never change the order of matrix multiplication or division.** Matrices are
non-commutative: `A * B ≠ B * A` in general. A transpose on the wrong side of
a product (`(AB)ᵀ = BᵀAᵀ`, not `AᵀBᵀ`) causes silent type mismatches or runtime
panics when dimensions don't align.
