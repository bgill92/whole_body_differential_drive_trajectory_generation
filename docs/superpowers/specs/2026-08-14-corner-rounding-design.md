# Path Corner Rounding — Design

Date: 2026-08-14. Status: approved (user selected circular-arc fillet,
clamp+warn on oversized radius, fillet in the corner's own plane).

## Problem

Waypoint paths interpolate as straight segments; at a corner the direction
changes instantly. The SQP trajectory optimizer must then spread a sharp
heading change across knots, which slows or stalls convergence. Rounding the
path itself removes the discontinuity at the source.

## Decisions

- **Geometry**: circular-arc fillet, tangent to both segments (constant
  curvature — friendliest to the diff-drive base; exact tangency).
- **Oversized radius**: clamp the fillet per corner to fit the adjacent
  segments (tangent cut ≤ half of each neighbor), stderr warning; the path
  always builds.
- **3D**: the arc lives in the plane spanned by the two segments (general 3D;
  reduces to the xy case for planar paths).
- **Config**: `path.corner_radius: f64`, `#[serde(default)]` — 0.0 (default)
  preserves the current behavior bit-for-bit via the existing code path.
- **Location**: all in `src/configs.rs` beside `interpolate()`; path
  generation stays with the path config.

## Geometry (interior waypoint B, neighbors A and C)

- `u = (B−A)/|B−A|`, `v = (C−B)/|C−B|`, turn angle `φ = acos(clamp(u·v))`.
- Collinear or reversal corners (`|u×v| < 1e-9`) are skipped (no plane, or no
  fillet exists).
- Tangent cut `d = r·tan(φ/2)`, clamped to `½·min(|AB|, |BC|)`; effective
  radius recomputed as `r_eff = d/tan(φ/2)` after clamping.
- Blend points `P_in = B − u·d`, `P_out = B + v·d`; arc center
  `O = B + normalize(v−u)·r_eff/cos(φ/2)`; arc sampled by rotating
  `P_in − O` about the unit normal `n = normalize(u×v)` through sweep `φ`.

## Sampling and orientation

The rounded path is a list of alternating straight/arc pieces. It is sampled
uniformly by arc length with the same total pose count as the unrounded path:
`1 + segments·poses_per_segment`, endpoints exactly at the first/last
waypoints. Orientation slerps piecewise: waypoint quaternions at waypoints,
segment-fraction quaternions at blend points, arcs slerp between their two
blend orientations. Constant-orientation paths stay constant everywhere.

## Testing

- Right-angle corner, radius r: minimum distance from the corner over the
  sampled poses ≈ r(√2−1); constant-rpy input keeps all orientations equal.
- Direction continuity: max turn between consecutive sampled directions stays
  small (vs. the 90° jump unrounded).
- Oversized radius: builds, warns, clamped clearance ≈ d(√2−1) for the right
  angle with d = half segment.
- Collinear waypoints with nonzero radius: path unchanged (stays on the line).
- `corner_radius` absent/0: existing tests cover the untouched legacy path.

## Out of scope

Per-corner radii, clothoid/curvature-continuous blends (arc is C¹, curvature
still steps at blend points — accept for v1), resampling density changes.
