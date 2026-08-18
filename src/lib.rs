mod configs;
mod diagnostics;
mod kinematics;
mod qp;
pub mod trajectory;

pub use crate::configs::{
    Config, DifferentialIkConfig, EeTracking, EqualityConstraint, TrajectoryConfig,
};
pub use crate::diagnostics::{
    BaseIndices, SLIP_TOLERANCE, SlipSummary, resolve_base_indices, slip_residual, slip_residuals,
    summarize_slip,
};
pub use crate::kinematics::{Kinematics, differential_ik};
