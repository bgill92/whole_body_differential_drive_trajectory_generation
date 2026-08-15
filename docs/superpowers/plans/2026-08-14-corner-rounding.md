# Corner Rounding Implementation Plan

> Executed inline (single-file feature); spec: docs/superpowers/specs/2026-08-14-corner-rounding-design.md

**Goal:** `path.corner_radius` rounds interior waypoint corners with tangent circular-arc fillets.

**Tasks:**

1. `src/configs.rs`: `corner_radius` field (`#[serde(default)]`), `PathPiece`/`PieceKind` types, `rounded_pieces` builder (fillet math per spec, clamp+warn), `sample_rounded` (uniform arc-length sampling, count = 1 + segments·poses_per_segment), `interpolate()` dispatches to it when radius > 0 and ≥3 waypoints — legacy loop untouched otherwise. Tests: right-angle clearance ≈ r(√2−1) + constant orientation, direction continuity bound, oversized-radius clamp, collinear no-op, parse default 0.
2. `assets/config.yaml`: `corner_radius: 0.3` with comment.
3. `cargo test` green (27 prior + new), clippy/fmt clean, `cargo run --release` exit 0, commit on `sqp-trajectory`, push (PR #2 picks it up), then review subagent + fixes.
