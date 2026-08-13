mod visualization;

use rerun::external::re_importer::UrdfTree;

use rerun::external::re_log;

use wbdd::{Config, Kinematics, differential_ik};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    re_log::setup_logging();

    let rec = rerun::RecordingStreamBuilder::new("urdf_view").spawn()?;

    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "assets/config.yaml".to_string());
    let config: Config = serde_yaml_ng::from_str(&std::fs::read_to_string(&config_path)?)?;

    let urdf_path = config.urdf_path.as_str();

    let mut kinematics = Kinematics::build(urdf_path, &config.end_joint).unwrap();

    let urdf = UrdfTree::from_file_path(urdf_path, None)?;

    let mut goal_poses = config.path.interpolate();

    visualization::log_urdf(&rec, urdf_path, &urdf)?;
    visualization::log_path_line(&rec, &goal_poses)?;
    visualization::log_goal_axes(&rec, &goal_poses)?;

    // Truncate only for solving — every pose above is still visualized.
    if config.path.solve_first_pose_only {
        goal_poses.truncate(1);
    }

    let mut joint_positions = Vec::new();
    for goal in &goal_poses {
        joint_positions.extend(differential_ik(goal, &mut kinematics, &config.differential_ik)?);
    }

    visualization::log_trajectory(&rec, &urdf, &kinematics.joint_names(), &joint_positions)?;

    Ok(())
}
