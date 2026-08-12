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

    let goal = config.goal.to_isometry();

    // println!("Jacobian: {}", k::jacobian(&kinematics.serial_chain));

    rec.log_file_from_path(urdf_path, None, true)?;

    let urdf = UrdfTree::from_file_path(urdf_path, None)?;

    let mut links: Vec<&str> = vec![urdf.root().name.as_str()];
    links.extend(urdf.joints().map(|j| j.child.link.as_str()));

    for link in links {
        let path = format!("axes/{}", link);
        rec.log_static(path.as_str(), &rerun::CoordinateFrame::new(link))?;
        rec.log_static(path, &rerun::TransformAxes3D::new(0.1))?;
    }

    // 1. Define where "goal_frame" sits, relative to an existing frame.
    let goal_translation = goal.translation.vector.cast::<f32>();
    let goal_quaternion = goal.rotation.as_vector().cast::<f32>();
    rec.log_static(
        "goal_transform",
        &rerun::Transform3D::new()
            .with_translation([goal_translation.x, goal_translation.y, goal_translation.z])
            // nalgebra stores the quaternion coefficients as [x, y, z, w], the
            // order rerun wants.
            .with_quaternion(rerun::Quaternion::from_xyzw([
                goal_quaternion[0],
                goal_quaternion[1],
                goal_quaternion[2],
                goal_quaternion[3],
            ]))
            .with_parent_frame("world") // must match a real URDF frame
            .with_child_frame("goal_frame"),
    )?;

    // 2. Draw axes at that frame — identical to the link loop.
    rec.log_static("axes/goal", &rerun::CoordinateFrame::new("goal_frame"))?;
    rec.log_static("axes/goal", &rerun::TransformAxes3D::new(0.1))?;

    let joint_positions = differential_ik(&goal, &mut kinematics, &config.differential_ik)?;

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
    // rec.set_time_sequence("step", 0);
    // for (name, position) in names.iter().zip(&positions) {
    //     let joint = urdf
    //         .get_joint_by_name(name)
    //         .ok_or_else(|| format!("no urdf joint named {name}"))?;
    //     let joint_transform = urdf.compute_joint_transform(joint, *position, false)?;
    //     // rec.log("/transforms", &joint_transform)?;
    // }

    Ok(())
}
