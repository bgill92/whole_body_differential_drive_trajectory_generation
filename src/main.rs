mod visualization;

use rerun::external::re_importer::UrdfTree;

use rerun::external::re_log;

use wbdd::{
    Config, EqualityConstraint, Kinematics, SLIP_TOLERANCE, differential_ik, pose_errors,
    resolve_base_indices, slip_residuals, summarize_slip, trajectory,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    re_log::setup_logging();

    // Spawn the Rerun viewer, or — when WBDD_RRD_PATH is set — write the
    // recording to an .rrd file for offline playback and rendering.
    let rec = match std::env::var("WBDD_RRD_PATH") {
        Ok(path) => rerun::RecordingStreamBuilder::new("urdf_view").save(path)?,
        Err(_) => rerun::RecordingStreamBuilder::new("urdf_view").spawn()?,
    };

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

    let ik_joint_positions = joint_positions.clone();

    // Whole-trajectory SQP pass, warm-started from the sequential IK result.
    // Skipped in first-pose debug mode: the optimizer needs every knot.
    let mut sqp_ran = false;
    if let Some(trajectory_config) = &config.trajectory
        && trajectory_config.enabled
    {
        if config.path.solve_first_pose_only {
            eprintln!("trajectory optimization skipped: path.solve_first_pose_only is set");
        } else {
            joint_positions = trajectory::optimize(
                &goal_poses,
                &mut kinematics,
                trajectory_config,
                &joint_positions,
            )?;
            sqp_ran = true;
        }
    }

    visualization::log_trajectory(&rec, &urdf, &kinematics.joint_names(), &joint_positions)?;

    let base_joint_names: Vec<String> = config
        .trajectory
        .as_ref()
        .map(|t| t.base_joint_names.clone())
        .unwrap_or_else(|| DEFAULT_BASE_JOINT_NAMES.map(String::from).to_vec());

    report_diagnostics(
        &rec,
        "ik",
        &goal_poses,
        &ik_joint_positions,
        &mut kinematics,
        &base_joint_names,
    )?;
    if sqp_ran {
        report_diagnostics(
            &rec,
            "sqp",
            &goal_poses,
            &joint_positions,
            &mut kinematics,
            &base_joint_names,
        )?;
    }

    // Block until everything (including the ~50 MiB of URDF meshes) reaches the
    // viewer — dropping the stream on exit silently discards unsent data.
    rec.flush_blocking()?;

    Ok(())
}

/// Fallback planar base joints when the `trajectory` config section (which
/// names them) is absent.
const DEFAULT_BASE_JOINT_NAMES: [&str; 3] = [
    "world_base_link_planar_prismatic_x",
    "world_base_link_planar_prismatic_y",
    "world_base_link_planar_yaw",
];

/// Compute, log, and print diagnostics for one solved trajectory.
fn report_diagnostics(
    rec: &rerun::RecordingStream,
    label: &str,
    goal_poses: &[k::Isometry3<f64>],
    joint_positions: &[Vec<f64>],
    kinematics: &mut Kinematics,
    base_joint_names: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let errors = pose_errors(goal_poses, joint_positions, kinematics);
    // total_cmp is NaN-robust — a plain `>` fold silently skips NaN, which
    // would misreport a solver blowup as a zero error.
    let (pos_knot, max_pos) = errors
        .iter()
        .enumerate()
        .map(|(i, e)| (i, e.position))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap_or((0, 0.0));
    let (ori_knot, max_ori) = errors
        .iter()
        .enumerate()
        .map(|(i, e)| (i, e.orientation))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap_or((0, 0.0));
    println!(
        "[{label}] pos err max {max_pos:.3e} m (knot {pos_knot}), \
         ori err max {max_ori:.3e} rad (knot {ori_knot})"
    );

    let slips = if joint_positions.len() < 2 {
        println!("[{label}] slip check skipped: fewer than two knots");
        Vec::new()
    } else {
        let base = resolve_base_indices(&kinematics.joint_names(), base_joint_names)?;
        let slips = slip_residuals(joint_positions, &base);
        let summary = summarize_slip(&slips);
        // max_index is None when every interval has exactly zero slip.
        let worst = summary
            .max_index
            .map_or_else(|| "-".to_string(), |i| i.to_string());
        println!(
            "[{label}] slip max {:.3e} m (interval {worst}), {}/{} intervals above {SLIP_TOLERANCE:.0e} m",
            summary.max_abs,
            summary.count_above_tol,
            slips.len(),
        );
        slips
    };

    visualization::log_diagnostics(rec, label, &errors, &slips)?;
    Ok(())
}
