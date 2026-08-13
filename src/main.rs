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

    let mut goal_poses = config.path.interpolate();

    // Full interpolated path as a translucent green line, logged before the
    // debug truncation so the whole path stays visible.
    let path_points: Vec<[f32; 3]> = goal_poses
        .iter()
        .map(|p| {
            let t = p.translation.vector.cast::<f32>();
            [t.x, t.y, t.z]
        })
        .collect();
    // Pin the line to the world frame; without a CoordinateFrame the entity
    // cannot be resolved once the URDF frame graph starts animating.
    rec.log_static("path_line", &rerun::CoordinateFrame::new("world"))?;
    rec.log_static(
        "path_line",
        &rerun::LineStrips3D::new([path_points])
            .with_colors([rerun::Color::from_unmultiplied_rgba(0, 255, 0, 100)]),
    )?;

    rec.log_file_from_path(urdf_path, None, true)?;

    let urdf = UrdfTree::from_file_path(urdf_path, None)?;

    let mut links: Vec<&str> = vec![urdf.root().name.as_str()];
    links.extend(urdf.joints().map(|j| j.child.link.as_str()));

    for link in links {
        let path = format!("axes/{}", link);
        rec.log_static(path.as_str(), &rerun::CoordinateFrame::new(link))?;
        rec.log_static(path, &rerun::TransformAxes3D::new(0.1))?;
    }

    for (i, goal) in goal_poses.iter().enumerate() {
        // 1. Define where the goal frame sits, relative to an existing frame.
        let goal_translation = goal.translation.vector.cast::<f32>();
        let goal_quaternion = goal.rotation.as_vector().cast::<f32>();
        let frame = format!("goal_frame_{i}");
        rec.log_static(
            format!("goal_transform_{i}"),
            &rerun::Transform3D::new()
                .with_translation([goal_translation.x, goal_translation.y, goal_translation.z])
                // nalgebra stores the quaternion coefficients as [x, y, z, w],
                // the order rerun wants.
                .with_quaternion(rerun::Quaternion::from_xyzw([
                    goal_quaternion[0],
                    goal_quaternion[1],
                    goal_quaternion[2],
                    goal_quaternion[3],
                ]))
                .with_parent_frame("world") // must match a real URDF frame
                .with_child_frame(frame.as_str()),
        )?;

        // 2. Draw axes at that frame — identical to the link loop.
        let path = format!("axes/goal_{i}");
        rec.log_static(path.as_str(), &rerun::CoordinateFrame::new(frame))?;
        rec.log_static(path, &rerun::TransformAxes3D::new(0.1))?;
    }

    // Truncate only for solving — every pose above is still visualized.
    if config.path.solve_first_pose_only {
        goal_poses.truncate(1);
    }

    let mut joint_positions = Vec::new();
    for goal in &goal_poses {
        joint_positions.extend(differential_ik(goal, &mut kinematics, &config.differential_ik)?);
    }

    let names = kinematics.joint_names();

    for (idx, positions) in joint_positions.iter().enumerate() {
        rec.set_time_sequence("step", idx as i64);
        for (name, position) in names.iter().zip(positions) {
            let joint = urdf
                .get_joint_by_name(name)
                .ok_or_else(|| format!("no urdf joint named {name}"))?;
            let joint_transform = urdf.compute_joint_transform(joint, *position, false)?;
            rec.log("/transforms", &joint_transform)?;
        }
    }

    Ok(())
}
