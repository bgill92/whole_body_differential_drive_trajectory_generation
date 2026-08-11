mod configs;
mod kinematics;

pub use crate::configs::{Config, DifferentialIkConfig, EqualityConstraint, SolverConfig};
pub use crate::kinematics::{Kinematics, differential_ik};
