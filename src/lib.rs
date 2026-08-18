mod configs;
mod diagnostics;
mod kinematics;
mod qp;
pub mod trajectory;

pub use crate::configs::{
    Config, DifferentialIkConfig, EeTracking, EqualityConstraint, TrajectoryConfig,
};
pub use crate::diagnostics::{BaseIndices, resolve_base_indices, slip_residual};
pub use crate::kinematics::{Kinematics, differential_ik};
