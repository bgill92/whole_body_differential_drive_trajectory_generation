mod visualization;

use rerun::external::re_importer::UrdfTree;

use rerun::external::re_log;

use wbdd::{Config, EqualityConstraint, Kinematics, differential_ik, trajectory};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    re_log::setup_logging();

    let rec = rerun::RecordingStreamBuilder::new("urdf_view").spawn()?;

    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "assets/config.yaml".to_string());
    let mut config: Config = serde_yaml_ng::from_str(&std::fs::read_to_string(&config_path)?)?;

    let urdf_path = config.urdf_path.as_str();

    let mut kinematics = Kinematics::build(urdf_path, &config.end_joint).unwrap();

    let urdf = UrdfTree::from_file_path(urdf_path, None)?;

    let goal_poses = config.path.interpolate();

    visualization::log_urdf(&rec, urdf_path, &urdf)?;
    visualization::log_path_line(&rec, &goal_poses)?;
    visualization::log_goal_axes(&rec, &goal_poses)?;

    // Seed the differential-drive base facing the direction of travel: pin the
    // planar yaw joint so the base x-axis is parallel to the first path
    // segment, for the first pose only.
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

    // Keep only the converged configuration per pose; the trajectory is never
    // empty (it starts with the seed configuration). Each solve seeds from the
    // previous one — `kinematics` keeps its joint positions between calls.
    let mut joint_positions = vec![
        differential_ik(&goal_poses[0], &mut kinematics, &config.differential_ik)?
            .pop()
            .unwrap(),
    ];

    // Equality constraints (config + yaw pin) apply only to the first pose;
    // subsequent solves run unconstrained.
    config.differential_ik.equality_constraints.clear();

    // Remaining poses skipped when debugging the first pose — still visualized.
    if !config.path.solve_first_pose_only {
        for goal in &goal_poses[1..] {
            joint_positions.push(
                differential_ik(goal, &mut kinematics, &config.differential_ik)?
                    .pop()
                    .unwrap(),
            );
        }
    }

    // Whole-trajectory SQP pass, warm-started from the sequential IK result.
    // Skipped in first-pose debug mode: the optimizer needs every knot.
    if let Some(trajectory_config) = &config.trajectory {
        if trajectory_config.enabled {
            if config.path.solve_first_pose_only {
                eprintln!("trajectory optimization skipped: path.solve_first_pose_only is set");
            } else {
                joint_positions = trajectory::optimize(
                    &goal_poses,
                    &mut kinematics,
                    trajectory_config,
                    &joint_positions,
                )?;
            }
        }
    }

    visualization::log_trajectory(&rec, &urdf, &kinematics.joint_names(), &joint_positions)?;

    // Block until everything (including the ~50 MiB of URDF meshes) reaches the
    // viewer — dropping the stream on exit silently discards unsent data.
    rec.flush_blocking();

    Ok(())
}
