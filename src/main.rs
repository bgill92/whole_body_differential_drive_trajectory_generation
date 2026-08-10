use rerun::external::re_importer::UrdfTree;
use rerun::{CoordinateFrame, TransformAxes3D};

use rerun::external::{anyhow, re_log, urdf_rs};

use k::prelude::*;

/// A pose written as translation + roll-pitch-yaw, so the YAML stays readable.
///
/// `Isometry3` can serde directly (nalgebra's `serde-serialize` feature), but
/// its wire format is a raw quaternion -- nobody hand-writes that.
#[derive(serde::Deserialize)]
struct Pose {
    xyz: [f64; 3],
    rpy: [f64; 3],
}

#[derive(serde::Deserialize)]
struct SolverConfig {
    allowable_target_distance: f64,
    allowable_target_angle: f64,
    jacobian_multiplier: f64,
    num_max_try: usize,
}

#[derive(serde::Deserialize)]
struct DifferentialIkConfig {
    num_steps: usize,
    pseudo_inverse_epsilon: f64,
    step_size: f64,
}

#[derive(serde::Deserialize)]
struct Config {
    urdf_path: String,
    end_joint: String,
    goal: Pose,
    solver: SolverConfig,
    differential_ik: DifferentialIkConfig,
}

impl Pose {
    fn to_isometry(&self) -> k::Isometry3<f64> {
        k::Isometry3::from_parts(
            k::nalgebra::Translation3::new(self.xyz[0], self.xyz[1], self.xyz[2]),
            k::nalgebra::UnitQuaternion::from_euler_angles(self.rpy[0], self.rpy[1], self.rpy[2]),
        )
    }
}

pub struct Kinematics {
    pub chain: k::Chain<f64>,
    pub serial_chain: k::SerialChain<f64>,
    pub solver: k::JacobianIkSolver<f64>,
}

impl Kinematics {
    fn build(
        urdf_path: &str,
        serial_chain_end_joint: &str,
        solver_config: &SolverConfig,
    ) -> Result<Kinematics, &'static str> {
        let chain = k::Chain::<f64>::from_urdf_file(urdf_path).unwrap();
        let end = chain
            .find(serial_chain_end_joint)
            .ok_or("joint_not_found")?;
        let serial_chain = k::SerialChain::from_end(end);
        let solver = k::JacobianIkSolver::new(
            solver_config.allowable_target_distance,
            solver_config.allowable_target_angle,
            solver_config.jacobian_multiplier,
            solver_config.num_max_try,
        );

        Ok(Kinematics {
            chain,
            serial_chain,
            solver,
        })
    }
    fn solve(&self, target_pose: &k::Isometry3<f64>) -> Result<(), k::Error> {
        self.solver.solve(&self.serial_chain, target_pose)?;

        Ok(())
    }
    fn get_serial_chain_joint_names_and_positions(&self) -> (Vec<String>, Vec<f64>) {
        let names: Vec<String> = self
            .serial_chain
            .iter_joints()
            .map(|j| j.name.clone())
            .collect();
        let positions = self.serial_chain.joint_positions();

        (names, positions)
    }
}

/// Matrix logarithm of a homogeneous transform: SE(3) -> se(3).
///
/// Returns the twist `[v; omega]` whose matrix exponential reproduces `pose`.
/// `omega` is the axis-angle vector; `v` is the translation mapped through the
/// inverse left Jacobian, *not* the raw translation.
///
/// Linear-first to match `k::jacobian`, whose rows are `[linear; angular]`.
/// Modern Robotics uses the opposite order (`[omega; v]`) -- reorder when
/// cross-checking against the book.
///
/// The rotation block is taken as-is; a non-orthonormal `pose` yields garbage
/// rather than an error.
fn se3_log(pose: &k::nalgebra::Matrix4<f64>) -> k::nalgebra::Vector6<f64> {
    let rotation =
        k::nalgebra::Rotation3::from_matrix_unchecked(pose.fixed_slice::<3, 3>(0, 0).into_owned());
    let translation = pose.fixed_slice::<3, 1>(0, 3).into_owned();

    let omega = rotation.scaled_axis();
    let theta = omega.norm();
    let omega_hat = omega.cross_matrix();

    // Coefficient on (omega_hat)^2 in V^-1. The closed form
    // 1/theta^2 - (1 + cos)/(2 * theta * sin) is 0/0 at theta = 0, so use its
    // Taylor series near the singularity. Same story at theta = pi, where sin
    // vanishes again -- the series is not valid there, but scaled_axis() keeps
    // theta <= pi and the term stays finite in the limit.
    let coeff = if theta < 1e-6 {
        1.0 / 12.0 + theta * theta / 720.0
    } else {
        1.0 / (theta * theta) - (1.0 + theta.cos()) / (2.0 * theta * theta.sin())
    };

    let v_inv = k::nalgebra::Matrix3::identity() - 0.5 * omega_hat + coeff * omega_hat * omega_hat;
    let v = v_inv * translation;

    k::nalgebra::Vector6::new(v[0], v[1], v[2], omega[0], omega[1], omega[2])
}

fn differential_ik(
    goal_pose: &k::Isometry3<f64>,
    kinematics: &Kinematics,
    config: &DifferentialIkConfig,
) -> Vec<Vec<f64>> {
    let goal_pose = goal_pose.to_homogeneous();

    let mut joint_positions: Vec<Vec<f64>> = vec![];
    joint_positions.push(kinematics.serial_chain.joint_positions());
    for _ in 0..config.num_steps {
        let current_pose = kinematics.serial_chain.end_transform().to_homogeneous();

        let current_pose_inverted = current_pose.try_inverse().unwrap();

        let temp = goal_pose * current_pose_inverted;

        let twist = se3_log(&temp);

        let current_joint_positions =
            k::nalgebra::DVector::from_vec(kinematics.serial_chain.joint_positions());

        let inv_jacobian = k::jacobian(&kinematics.serial_chain)
            .pseudo_inverse(config.pseudo_inverse_epsilon)
            .unwrap();

        // Normal Newton-Raphson, with some step size built in
        // let updated_joint_positions =
        //     current_joint_positions + inv_jacobian * twist * config.step_size;

        let lambda = 0.5;

        let dls_term = k::jacobian(&kinematics.serial_chain).transpose()
            * (k::jacobian(&kinematics.serial_chain)
                * k::jacobian(&kinematics.serial_chain).transpose()
                + f64::powf(lambda, 2.0) * k::nalgebra::DMatrix::identity(6, 6))
            .try_inverse()
            .unwrap();

        let updated_joint_positions = current_joint_positions + dls_term * twist;

        kinematics
            .serial_chain
            .set_joint_positions_unchecked(updated_joint_positions.as_slice());

        println!("next angles={:?}", updated_joint_positions);
        joint_positions.push(kinematics.serial_chain.joint_positions());
    }

    // println!(
    //     "Jacobian size: {:?}",
    //     k::jacobian(&kinematics.serial_chain).shape()
    // );

    // println!("Goal pose matrix: {}", goal_pose);

    println!(
        "Current pose matrix: {}",
        kinematics.serial_chain.end_transform().to_homogeneous()
    );

    // let current_pose_inverted = current_pose.try_inverse().unwrap();

    // println!(
    //     "Inverted current pose times goal pose: {}",
    //     current_pose_inverted * goal_pose
    // );

    // let temp = goal_pose * current_pose_inverted;

    // let twist = se3_log(&temp);

    // println!("twist: {}", twist);

    // println!(
    //     "initial angles={:?}",
    //     kinematics.serial_chain.joint_positions()
    // );

    // let updated_joint_positions =
    //     k::nalgebra::DVector::from_vec(kinematics.serial_chain.joint_positions())
    //         + k::jacobian(&kinematics.serial_chain)
    //             .pseudo_inverse(0.000001)
    //             .unwrap()
    //             * twist;

    // println!("next angles={:?}", updated_joint_positions);
    // let invert_successful = k::nalgebra::try_invert_to(current_pose, &current_pose_inverted);
    joint_positions
}

#[cfg(test)]
mod tests {
    use super::{Config, se3_log};
    use k::nalgebra::{Isometry3, Matrix4, UnitQuaternion, Vector3};

    // The shipped config must stay parseable and produce the pose it spells out.
    #[test]
    fn config_parses() {
        let config: Config =
            serde_yaml_ng::from_str(&std::fs::read_to_string("assets/config.yaml").unwrap())
                .unwrap();
        let goal = config.goal.to_isometry();

        assert_eq!(goal.translation.vector, Vector3::new(-1.0, -1.0, 1.0));
        let (roll, pitch, yaw) = goal.rotation.euler_angles();
        assert!((roll.abs() - std::f64::consts::PI).abs() < 1e-9);
        assert!(pitch.abs() < 1e-9 && yaw.abs() < 1e-9);
    }

    // Cross-check se3_log against nalgebra's general matrix exponential: expm of
    // the hat matrix must reproduce the original homogeneous transform.
    fn assert_roundtrip(pose: &Isometry3<f64>) {
        let twist = se3_log(&pose.to_homogeneous());
        let omega_hat = Vector3::new(twist[3], twist[4], twist[5]).cross_matrix();

        let mut hat = Matrix4::zeros();
        hat.fixed_slice_mut::<3, 3>(0, 0).copy_from(&omega_hat);
        hat[(0, 3)] = twist[0];
        hat[(1, 3)] = twist[1];
        hat[(2, 3)] = twist[2];

        let rebuilt = hat.exp();
        assert!(
            (rebuilt - pose.to_homogeneous()).norm() < 1e-9,
            "roundtrip failed\nexpected:{}\ngot:{}",
            pose.to_homogeneous(),
            rebuilt
        );
    }

    #[test]
    fn log_roundtrips() {
        assert_roundtrip(&Isometry3::identity());
        // Pure translation: rotation is exactly zero, so the small-angle branch runs.
        assert_roundtrip(&Isometry3::translation(1.0, -2.0, 3.0));
        assert_roundtrip(&Isometry3::from_parts(
            Vector3::new(0.3, -0.7, 1.1).into(),
            UnitQuaternion::from_scaled_axis(Vector3::new(0.2, -1.3, 0.5)),
        ));
        // Near the theta = pi singularity.
        assert_roundtrip(&Isometry3::from_parts(
            Vector3::new(-1.0, 4.0, 0.5).into(),
            UnitQuaternion::from_scaled_axis(Vector3::x() * (std::f64::consts::PI - 1e-4)),
        ));
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    re_log::setup_logging();

    let rec = rerun::RecordingStreamBuilder::new("urdf_view").spawn()?;

    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "assets/config.yaml".to_string());
    let config: Config = serde_yaml_ng::from_str(&std::fs::read_to_string(&config_path)?)?;

    let urdf_path = config.urdf_path.as_str();

    let kinematics = Kinematics::build(urdf_path, &config.end_joint, &config.solver).unwrap();

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

    kinematics.serial_chain.update_transforms();

    let joint_positions = differential_ik(&goal, &kinematics, &config.differential_ik);

    // kinematics.solve(&target)?;

    let (names, _) = kinematics.get_serial_chain_joint_names_and_positions();

    for idx in 0..joint_positions.len() {
        rec.set_time_sequence("step", idx as i64);
        let positions = &joint_positions[idx];
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
