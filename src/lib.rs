mod configs;
mod kinematics;

pub use crate::configs::{Config, DifferentialIkConfig, SolverConfig};
pub use crate::kinematics::{Kinematics, differential_ik};
