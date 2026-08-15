/// A pose written as translation + roll-pitch-yaw, so the YAML stays readable.
///
/// `Isometry3` can serde directly (nalgebra's `serde-serialize` feature), but
/// its wire format is a raw quaternion -- nobody hand-writes that.
#[derive(serde::Deserialize)]
pub struct Pose {
    pub xyz: [f64; 3],
    pub rpy: [f64; 3],
}

#[derive(serde::Deserialize)]
pub struct EqualityConstraint {
    /// Joint name (must exist in the serial chain)
    pub joint_name: String,
    /// Target value for this joint
    pub target_value: f64,
}

#[derive(serde::Deserialize)]
pub struct DifferentialIkConfig {
    pub num_steps: usize,
    pub damping_factor: f64,
    pub convergence_threshold: f64,
    /// Optional equality constraints: joints forced to specific values each
    /// iteration. Applied to the first pose solve only (see main.rs).
    #[serde(default)]
    pub equality_constraints: Vec<EqualityConstraint>,
}

/// End-effector path-following mode for the trajectory optimizer.
#[derive(serde::Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum EeTracking {
    /// Weighted quadratic pose-error cost per knot; always feasible.
    Soft,
    /// Linearized pose equality rows per knot; may be infeasible when the
    /// path conflicts with base kinematics.
    Hard,
}

/// Whole-trajectory SQP settings. See
/// docs/superpowers/specs/2026-08-14-sqp-trajectory-optimization-design.md.
#[derive(serde::Deserialize)]
pub struct TrajectoryConfig {
    pub enabled: bool,
    /// Seconds per knot interval (velocities are finite differences over dt).
    pub dt: f64,
    pub ee_tracking: EeTracking,
    /// Soft-mode EE cost weight; unused in hard mode.
    pub ee_weight: f64,
    pub smoothness_weight: f64,
    /// One scalar for all joints; the k crate does not expose URDF velocity
    /// limits, so per-joint limits are deferred until it does.
    pub max_joint_velocity: f64,
    pub sqp_max_iterations: usize,
    /// Per-variable bound on each SQP step, intersected with joint limits.
    pub trust_region: f64,
    /// Converged when the step infinity-norm drops below this.
    pub convergence_step_norm: f64,
    /// Coefficient added to the diagonal of P; keeps P positive definite.
    pub damping: f64,
    /// Planar base joints in x, y, yaw order; validated against the chain.
    pub base_joint_names: Vec<String>,
}

/// A path of waypoint poses, linearly interpolated into a denser pose list.
#[derive(serde::Deserialize)]
pub struct PathConfig {
    pub waypoints: Vec<Pose>,
    /// Interpolated poses added per waypoint segment (endpoints included).
    pub poses_per_segment: usize,
    /// Debug: solve only the first pose of the interpolated path.
    #[serde(default)]
    pub solve_first_pose_only: bool,
    /// Pin the base yaw joint for the first pose so the base x-axis points
    /// along the path direction (first pose toward second interpolated pose).
    /// Seeds the differential-drive base facing the direction of travel.
    #[serde(default)]
    pub align_first_pose_base_yaw: bool,
    /// Fillet radius for rounding interior waypoint corners with tangent
    /// circular arcs; 0 (the default) leaves corners sharp and preserves the
    /// legacy interpolation exactly. Oversized radii are clamped per corner
    /// to fit the adjacent segments, with a stderr warning.
    #[serde(default)]
    pub corner_radius: f64,
}

#[derive(serde::Deserialize)]
pub struct Config {
    pub urdf_path: String,
    pub end_joint: String,
    pub path: PathConfig,
    pub differential_ik: DifferentialIkConfig,
    /// Optional whole-trajectory optimization stage; absent = IK only.
    #[serde(default)]
    pub trajectory: Option<TrajectoryConfig>,
}

impl PathConfig {
    /// Interpolate waypoints into a pose list: translation lerp, rotation
    /// slerp. With a nonzero `corner_radius`, interior corners are replaced by
    /// tangent circular-arc fillets and the path is resampled uniformly by
    /// arc length (same total pose count either way).
    pub fn interpolate(&self) -> Vec<k::Isometry3<f64>> {
        assert!(!self.waypoints.is_empty(), "path.waypoints is empty");
        assert!(self.corner_radius >= 0.0, "path.corner_radius must be >= 0");
        let waypoints: Vec<k::Isometry3<f64>> =
            self.waypoints.iter().map(Pose::to_isometry).collect();

        if self.corner_radius > 0.0 && waypoints.len() >= 3 {
            return sample_rounded(&waypoints, self.corner_radius, self.poses_per_segment);
        }

        let mut poses = vec![waypoints[0]];
        for pair in waypoints.windows(2) {
            for i in 1..=self.poses_per_segment {
                let t = i as f64 / self.poses_per_segment as f64;
                poses.push(pair[0].lerp_slerp(&pair[1], t));
            }
        }
        poses
    }
}

type Vec3 = k::nalgebra::Vector3<f64>;
type Quat = k::nalgebra::UnitQuaternion<f64>;

/// One straight or arc piece of a corner-rounded path. Orientation slerps
/// from `start_rotation` to `end_rotation` along the piece's length.
struct PathPiece {
    length: f64,
    start_rotation: Quat,
    end_rotation: Quat,
    kind: PieceKind,
}

enum PieceKind {
    Line {
        start: Vec3,
        /// Full displacement over the piece (end − start), not normalized.
        direction: Vec3,
    },
    Arc {
        center: Vec3,
        /// Blend-entry point relative to the center; rotated through the
        /// sweep to trace the arc.
        start_offset: Vec3,
        /// Unit rotation axis (normal of the corner's plane).
        axis: Vec3,
        sweep: f64,
    },
}

impl PathPiece {
    fn pose_at(&self, t: f64) -> k::Isometry3<f64> {
        let translation = match &self.kind {
            PieceKind::Line { start, direction } => start + direction * t,
            PieceKind::Arc {
                center,
                start_offset,
                axis,
                sweep,
            } => center + Quat::from_scaled_axis(axis * (sweep * t)) * start_offset,
        };
        k::Isometry3::from_parts(
            translation.into(),
            self.start_rotation.slerp(&self.end_rotation, t),
        )
    }
}

/// Tangent-arc fillet for one interior waypoint.
struct CornerFillet {
    /// Tangent cut distance along both adjacent segments (possibly clamped).
    cut: f64,
    center: Vec3,
    axis: Vec3,
    sweep: f64,
    /// Effective radius after clamping; arc length is `radius * sweep`.
    radius: f64,
}

/// Build the alternating line/arc pieces of the rounded path. Collinear and
/// reversal corners (no plane to fillet in, or no tangent arc exists) are
/// left sharp.
fn rounded_pieces(waypoints: &[k::Isometry3<f64>], radius: f64) -> Vec<PathPiece> {
    let n = waypoints.len();
    let positions: Vec<Vec3> = waypoints.iter().map(|w| w.translation.vector).collect();
    let rotations: Vec<Quat> = waypoints.iter().map(|w| w.rotation).collect();

    let mut fillets: Vec<Option<CornerFillet>> = (0..n).map(|_| None).collect();
    for i in 1..n - 1 {
        let to_here = positions[i] - positions[i - 1];
        let to_next = positions[i + 1] - positions[i];
        let (len_in, len_out) = (to_here.norm(), to_next.norm());
        if len_in < 1e-12 || len_out < 1e-12 {
            continue;
        }
        let u = to_here / len_in;
        let v = to_next / len_out;
        let plane_normal = u.cross(&v);
        if plane_normal.norm() < 1e-9 {
            continue;
        }
        let turn = u.dot(&v).clamp(-1.0, 1.0).acos();
        let half_turn_tan = (turn / 2.0).tan();
        let mut cut = radius * half_turn_tan;
        let max_cut = 0.5 * len_in.min(len_out);
        if cut > max_cut {
            eprintln!(
                "path.corner_radius {radius} too large at waypoint {i}; clamping the fillet to fit"
            );
            cut = max_cut;
        }
        let effective_radius = cut / half_turn_tan;
        // The center sits on the interior bisector, at distance r/cos(φ/2)
        // from the corner, so both segments are tangent at distance `cut`.
        let bisector = (v - u).normalize();
        let center = positions[i] + bisector * (effective_radius / (turn / 2.0).cos());
        fillets[i] = Some(CornerFillet {
            cut,
            center,
            axis: plane_normal.normalize(),
            sweep: turn,
            radius: effective_radius,
        });
    }

    let mut pieces = Vec::new();
    let mut cursor_position = positions[0];
    let mut cursor_rotation = rotations[0];
    for i in 1..n {
        let segment = positions[i] - positions[i - 1];
        let segment_length = segment.norm();
        let cut = fillets[i].as_ref().map_or(0.0, |f| f.cut);
        // The straight run ends at the fillet entry (or the waypoint itself
        // when the corner is sharp or this is the final segment); its end
        // orientation is the segment slerp evaluated at that point.
        let line_end = positions[i] - segment * (cut / segment_length);
        let end_rotation =
            rotations[i - 1].slerp(&rotations[i], (segment_length - cut) / segment_length);
        let line_vector = line_end - cursor_position;
        let line_length = line_vector.norm();
        if line_length > 1e-12 {
            pieces.push(PathPiece {
                length: line_length,
                start_rotation: cursor_rotation,
                end_rotation,
                kind: PieceKind::Line {
                    start: cursor_position,
                    direction: line_vector,
                },
            });
        }
        cursor_position = line_end;
        cursor_rotation = end_rotation;

        if let Some(fillet) = &fillets[i] {
            let next_segment = positions[i + 1] - positions[i];
            let next_length = next_segment.norm();
            let exit_rotation = rotations[i].slerp(&rotations[i + 1], fillet.cut / next_length);
            pieces.push(PathPiece {
                length: fillet.radius * fillet.sweep,
                start_rotation: cursor_rotation,
                end_rotation: exit_rotation,
                kind: PieceKind::Arc {
                    center: fillet.center,
                    start_offset: cursor_position - fillet.center,
                    axis: fillet.axis,
                    sweep: fillet.sweep,
                },
            });
            cursor_position = positions[i] + next_segment * (fillet.cut / next_length);
            cursor_rotation = exit_rotation;
        }
    }
    pieces
}

/// Sample the rounded path uniformly by arc length, matching the unrounded
/// pose count (1 + segments * poses_per_segment), endpoints exact.
fn sample_rounded(
    waypoints: &[k::Isometry3<f64>],
    radius: f64,
    poses_per_segment: usize,
) -> Vec<k::Isometry3<f64>> {
    let pieces = rounded_pieces(waypoints, radius);
    let total_length: f64 = pieces.iter().map(|p| p.length).sum();
    let count = 1 + (waypoints.len() - 1) * poses_per_segment;

    let mut poses = Vec::with_capacity(count);
    for index in 0..count {
        let mut s = total_length * index as f64 / (count - 1) as f64;
        let mut pose = None;
        for piece in &pieces {
            if s <= piece.length {
                pose = Some(piece.pose_at(s / piece.length));
                break;
            }
            s -= piece.length;
        }
        // Float spill past the last piece lands exactly on the path's end.
        poses.push(pose.unwrap_or_else(|| {
            pieces
                .last()
                .expect("rounded path has at least one piece")
                .pose_at(1.0)
        }));
    }
    poses
}

impl Pose {
    pub fn to_isometry(&self) -> k::Isometry3<f64> {
        k::Isometry3::from_parts(
            k::nalgebra::Translation3::new(self.xyz[0], self.xyz[1], self.xyz[2]),
            k::nalgebra::UnitQuaternion::from_euler_angles(self.rpy[0], self.rpy[1], self.rpy[2]),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use k::nalgebra::Vector3;

    // Inline YAML so the test checks parsing, not the current tuning of
    // assets/config.yaml.
    const CONFIG_YAML: &str = "
urdf_path: assets/rox_diff_ur5e.urdf
end_joint: ur5ewrist_3_joint
path:
  poses_per_segment: 2
  waypoints:
    - xyz: [-1.0, -1.0, 1.0]
      rpy: [3.14159265358979, 0.0, 0.0]
    - xyz: [1.0, -1.0, 1.0]
      rpy: [3.14159265358979, 0.0, 0.0]
differential_ik:
  num_steps: 25
  damping_factor: 0.5
  convergence_threshold: 0.01
trajectory:
  enabled: true
  dt: 0.5
  ee_tracking: soft
  ee_weight: 100.0
  smoothness_weight: 1.0
  max_joint_velocity: 1.0
  sqp_max_iterations: 20
  trust_region: 0.2
  convergence_step_norm: 1.0e-4
  damping: 1.0e-3
  base_joint_names: [world_base_link_planar_prismatic_x, world_base_link_planar_prismatic_y, world_base_link_planar_yaw]
";

    #[test]
    fn config_parses() {
        let config: Config = serde_yaml_ng::from_str(CONFIG_YAML).unwrap();
        let goal = config.path.waypoints[0].to_isometry();

        assert_eq!(goal.translation.vector, Vector3::new(-1.0, -1.0, 1.0));
        let (roll, pitch, yaw) = goal.rotation.euler_angles();
        assert!((roll.abs() - std::f64::consts::PI).abs() < 1e-9);
        assert!(pitch.abs() < 1e-9 && yaw.abs() < 1e-9);
        // Optional flags default to off when absent from the YAML.
        assert!(!config.path.solve_first_pose_only);
        assert!(!config.path.align_first_pose_base_yaw);
    }

    #[test]
    fn path_interpolates_linearly() {
        let config: Config = serde_yaml_ng::from_str(CONFIG_YAML).unwrap();
        let poses = config.path.interpolate();

        // 2 waypoints, 2 poses per segment: start + 2 interpolated = 3.
        assert_eq!(poses.len(), 3);
        assert_eq!(
            poses[1].translation.vector,
            Vector3::new(0.0, -1.0, 1.0),
            "midpoint translation wrong"
        );
        assert_eq!(poses[2].translation.vector, Vector3::new(1.0, -1.0, 1.0));
    }

    #[test]
    fn trajectory_config_parses() {
        let config: Config = serde_yaml_ng::from_str(CONFIG_YAML).unwrap();
        let trajectory = config.trajectory.expect("trajectory section missing");
        assert!(trajectory.enabled);
        assert!(matches!(trajectory.ee_tracking, EeTracking::Soft));
        assert_eq!(trajectory.base_joint_names.len(), 3);
        assert_eq!(trajectory.sqp_max_iterations, 20);

        // The section is optional: configs without it parse to None.
        let without = CONFIG_YAML.split("trajectory:").next().unwrap();
        let config: Config = serde_yaml_ng::from_str(without).unwrap();
        assert!(config.trajectory.is_none());

        // The enum rejects unknown modes rather than defaulting.
        let bad = CONFIG_YAML.replace("ee_tracking: soft", "ee_tracking: rigid");
        assert!(serde_yaml_ng::from_str::<Config>(&bad).is_err());

        // corner_radius is optional and defaults to sharp corners.
        assert_eq!(config.path.corner_radius, 0.0);
    }

    // Right-angle corner path used by the rounding tests: along +x, then +y.
    fn right_angle_path(corner_radius: f64) -> PathConfig {
        PathConfig {
            waypoints: vec![
                Pose {
                    xyz: [0.0, 0.0, 0.0],
                    rpy: [0.0, 0.0, 0.0],
                },
                Pose {
                    xyz: [1.0, 0.0, 0.0],
                    rpy: [0.0, 0.0, 0.0],
                },
                Pose {
                    xyz: [1.0, 1.0, 0.0],
                    rpy: [0.0, 0.0, 0.0],
                },
            ],
            poses_per_segment: 50,
            solve_first_pose_only: false,
            align_first_pose_base_yaw: false,
            corner_radius,
        }
    }

    fn min_distance_to(poses: &[k::Isometry3<f64>], point: Vector3<f64>) -> f64 {
        poses
            .iter()
            .map(|p| (p.translation.vector - point).norm())
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn rounded_corner_keeps_clearance_and_orientation() {
        let poses = right_angle_path(0.2).interpolate();

        // Same pose count as the unrounded path, endpoints exact.
        assert_eq!(poses.len(), 101);
        assert!(poses[0].translation.vector.norm() < 1e-12);
        assert!((poses[100].translation.vector - Vector3::new(1.0, 1.0, 0.0)).norm() < 1e-12);

        // The arc's closest approach to a right-angle corner is r(sqrt(2)-1).
        let clearance = min_distance_to(&poses, Vector3::new(1.0, 0.0, 0.0));
        let expected = 0.2 * (2.0_f64.sqrt() - 1.0);
        assert!(
            (clearance - expected).abs() < 5e-3,
            "corner clearance {clearance}, expected ~{expected}"
        );

        // Constant-orientation waypoints must stay constant along the blend.
        for pose in &poses {
            assert!(pose.rotation.angle() < 1e-9, "orientation drifted");
        }
    }

    #[test]
    fn rounded_path_direction_is_continuous() {
        let poses = right_angle_path(0.2).interpolate();
        let mut max_turn: f64 = 0.0;
        for window in poses.windows(3) {
            let a = window[1].translation.vector - window[0].translation.vector;
            let b = window[2].translation.vector - window[1].translation.vector;
            if a.norm() < 1e-12 || b.norm() < 1e-12 {
                continue;
            }
            let cos = (a.dot(&b) / (a.norm() * b.norm())).clamp(-1.0, 1.0);
            max_turn = max_turn.max(cos.acos());
        }
        // Unrounded, the corner is a single ~90 degree jump; rounded, each
        // step turns by at most roughly the arc's angular step.
        assert!(max_turn < 0.15, "max per-step turn {max_turn}");
    }

    #[test]
    fn oversized_radius_clamps_to_fit() {
        let poses = right_angle_path(10.0).interpolate();
        assert_eq!(poses.len(), 101);
        // Clamped cut = half the shorter segment (0.5), so the effective
        // radius is 0.5 and the clearance is 0.5(sqrt(2)-1).
        let clearance = min_distance_to(&poses, Vector3::new(1.0, 0.0, 0.0));
        let expected = 0.5 * (2.0_f64.sqrt() - 1.0);
        assert!(
            (clearance - expected).abs() < 5e-3,
            "clamped clearance {clearance}, expected ~{expected}"
        );
        for pose in &poses {
            assert!(pose.translation.vector.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn collinear_waypoints_are_left_sharp() {
        let mut path = right_angle_path(0.3);
        path.waypoints[2] = Pose {
            xyz: [2.0, 0.0, 0.0],
            rpy: [0.0, 0.0, 0.0],
        };
        let poses = path.interpolate();
        assert_eq!(poses.len(), 101);
        for pose in &poses {
            assert!(
                pose.translation.vector.y.abs() < 1e-9 && pose.translation.vector.z.abs() < 1e-9,
                "collinear path left the line"
            );
        }
    }
}
