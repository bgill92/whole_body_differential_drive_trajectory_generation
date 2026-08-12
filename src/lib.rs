mod active_set;
mod configs;
mod kinematics;

pub use crate::configs::{Config, DifferentialIkConfig, EqualityConstraint};
pub use crate::kinematics::{Kinematics, differential_ik};
