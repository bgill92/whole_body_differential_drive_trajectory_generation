pub use self::config::{Config, DifferentialIkConfig, SolverConfig};

mod config {
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
    pub struct SolverConfig {
        pub allowable_target_distance: f64,
        pub allowable_target_angle: f64,
        pub jacobian_multiplier: f64,
        pub num_max_try: usize,
    }

    #[derive(serde::Deserialize)]
    pub struct DifferentialIkConfig {
        pub num_steps: usize,
        pub pseudo_inverse_epsilon: f64,
        pub step_size: f64,
    }

    #[derive(serde::Deserialize)]
    pub struct Config {
        pub urdf_path: String,
        pub end_joint: String,
        pub goal: Pose,
        pub solver: SolverConfig,
        pub differential_ik: DifferentialIkConfig,
    }

    impl Pose {
        pub fn to_isometry(&self) -> k::Isometry3<f64> {
            k::Isometry3::from_parts(
                k::nalgebra::Translation3::new(self.xyz[0], self.xyz[1], self.xyz[2]),
                k::nalgebra::UnitQuaternion::from_euler_angles(
                    self.rpy[0],
                    self.rpy[1],
                    self.rpy[2],
                ),
            )
        }
    }
}
