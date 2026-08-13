//! Rerun logging helpers for the trajectory viewer.

use rerun::external::re_importer::UrdfTree;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Log the URDF model plus a coordinate-axes gizmo on every link frame.
pub fn log_urdf(rec: &rerun::RecordingStream, urdf_path: &str, urdf: &UrdfTree) -> Result<()> {
    rec.log_file_from_path(urdf_path, None, true)?;

    let mut links: Vec<&str> = vec![urdf.root().name.as_str()];
    links.extend(urdf.joints().map(|j| j.child.link.as_str()));

    for link in links {
        let path = format!("axes/{}", link);
        rec.log_static(path.as_str(), &rerun::CoordinateFrame::new(link))?;
        rec.log_static(path, &rerun::TransformAxes3D::new(0.1))?;
    }
    Ok(())
}

/// Log the interpolated path as a translucent green line in the world frame.
pub fn log_path_line(rec: &rerun::RecordingStream, poses: &[k::Isometry3<f64>]) -> Result<()> {
    let points: Vec<[f32; 3]> = poses
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
        &rerun::LineStrips3D::new([points])
            .with_colors([rerun::Color::from_unmultiplied_rgba(0, 255, 0, 100)]),
    )?;
    Ok(())
}

/// Log a coordinate-axes gizmo at every pose, each in its own world-relative
/// frame.
pub fn log_goal_axes(rec: &rerun::RecordingStream, poses: &[k::Isometry3<f64>]) -> Result<()> {
    for (i, goal) in poses.iter().enumerate() {
        // 1. Define where the goal frame sits, relative to an existing frame.
        let translation = goal.translation.vector.cast::<f32>();
        let quaternion = goal.rotation.as_vector().cast::<f32>();
        let frame = format!("goal_frame_{i}");
        rec.log_static(
            format!("goal_transform_{i}"),
            &rerun::Transform3D::new()
                .with_translation([translation.x, translation.y, translation.z])
                // nalgebra stores the quaternion coefficients as [x, y, z, w],
                // the order rerun wants.
                .with_quaternion(rerun::Quaternion::from_xyzw([
                    quaternion[0],
                    quaternion[1],
                    quaternion[2],
                    quaternion[3],
                ]))
                .with_parent_frame("world") // must match a real URDF frame
                .with_child_frame(frame.as_str()),
        )?;

        // 2. Draw axes at that frame — identical to the link loop.
        let path = format!("axes/goal_{i}");
        rec.log_static(path.as_str(), &rerun::CoordinateFrame::new(frame))?;
        rec.log_static(path, &rerun::TransformAxes3D::new(0.1))?;
    }
    Ok(())
}

/// Animate the joint-position trajectory on the "step" timeline.
pub fn log_trajectory(
    rec: &rerun::RecordingStream,
    urdf: &UrdfTree,
    joint_names: &[String],
    joint_positions: &[Vec<f64>],
) -> Result<()> {
    for (idx, positions) in joint_positions.iter().enumerate() {
        rec.set_time_sequence("step", idx as i64);
        for (name, position) in joint_names.iter().zip(positions) {
            let joint = urdf
                .get_joint_by_name(name)
                .ok_or_else(|| format!("no urdf joint named {name}"))?;
            let joint_transform = urdf.compute_joint_transform(joint, *position, false)?;
            rec.log("/transforms", &joint_transform)?;
        }
    }
    Ok(())
}
