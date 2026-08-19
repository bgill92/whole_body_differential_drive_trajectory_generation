//! Timing harness: full pipeline minus Rerun. Prints per-stage wall time.
//! Run: cargo run --release --example timing [config.yaml]

use std::time::Instant;
use wbdd::{Config, EqualityConstraint, Kinematics, differential_ik, trajectory};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "assets/config.yaml".to_string());
    let mut config: Config = serde_yaml_ng::from_str(&std::fs::read_to_string(&config_path)?)?;
    let mut kinematics = Kinematics::build(config.urdf_path.as_str(), &config.end_joint).unwrap();

    let goal_poses = config.path.interpolate();
    println!("knots: {}", goal_poses.len());

    if config.path.align_first_pose_base_yaw && goal_poses.len() > 1 {
        let direction = goal_poses[1].translation.vector - goal_poses[0].translation.vector;
        config
            .differential_ik
            .equality_constraints
            .push(EqualityConstraint {
                joint_name: "world_base_link_planar_yaw".to_string(),
                target_value: direction.y.atan2(direction.x),
            });
    }

    let t = Instant::now();
    let mut joint_positions = vec![
        differential_ik(&goal_poses[0], &mut kinematics, &config.differential_ik)?
            .pop()
            .unwrap(),
    ];
    config.differential_ik.equality_constraints.clear();
    for goal in &goal_poses[1..] {
        joint_positions.push(
            differential_ik(goal, &mut kinematics, &config.differential_ik)?
                .pop()
                .unwrap(),
        );
    }
    println!("sequential IK: {:.3} s", t.elapsed().as_secs_f64());

    let trajectory_config = config.trajectory.as_ref().expect("trajectory config");
    let t = Instant::now();
    let optimized = trajectory::optimize(
        &goal_poses,
        &mut kinematics,
        trajectory_config,
        &joint_positions,
    )?;
    println!("SQP optimize: {:.3} s", t.elapsed().as_secs_f64());
    println!("optimized knots: {}", optimized.len());
    Ok(())
}
