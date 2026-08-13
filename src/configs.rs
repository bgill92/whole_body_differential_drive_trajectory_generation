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
    /// Optional equality constraints: joints forced to specific values each iteration
    #[serde(default)]
    pub equality_constraints: Vec<EqualityConstraint>,
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
}

#[derive(serde::Deserialize)]
pub struct Config {
    pub urdf_path: String,
    pub end_joint: String,
    pub path: PathConfig,
    pub differential_ik: DifferentialIkConfig,
}

impl PathConfig {
    /// Interpolate waypoints into a pose list: translation lerp, rotation slerp.
    pub fn interpolate(&self) -> Vec<k::Isometry3<f64>> {
        assert!(!self.waypoints.is_empty(), "path.waypoints is empty");
        let waypoints: Vec<k::Isometry3<f64>> =
            self.waypoints.iter().map(Pose::to_isometry).collect();

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
";

    #[test]
    fn config_parses() {
        let config: Config = serde_yaml_ng::from_str(CONFIG_YAML).unwrap();
        let goal = config.path.waypoints[0].to_isometry();

        assert_eq!(goal.translation.vector, Vector3::new(-1.0, -1.0, 1.0));
        let (roll, pitch, yaw) = goal.rotation.euler_angles();
        assert!((roll.abs() - std::f64::consts::PI).abs() < 1e-9);
        assert!(pitch.abs() < 1e-9 && yaw.abs() < 1e-9);
        // Debug flag defaults to off when absent from the YAML.
        assert!(!config.path.solve_first_pose_only);
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
}
