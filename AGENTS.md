# Project Conventions

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
pub use self::config::{Config, DifferentialIkConfig, SolverConfig};

mod config { /* ... */ }
```

Prefer `self::` (or `crate::`) on these paths rather than a bare `use config::`
— the explicit prefix disambiguates a local module from an external crate of
the same name.

Re-export deliberately. A type left out of the `pub use` list (e.g. `Pose`) is
an internal detail; do not add it to the re-export just because something in
`main.rs` wants it — either use it through the types that expose it, or make
the omission a considered API change.

Give a module its own file (`src/config.rs`) once it outgrows a screen or two.
Inline `mod foo { ... }` blocks are fine for small, tightly-coupled helpers.

## Before Finishing

Run `cargo check` and read the warnings. This crate builds warning-clean apart
from known dead code; a new `unused`/`dead_code` warning means either the code
is unreachable or it is wired up wrong.
