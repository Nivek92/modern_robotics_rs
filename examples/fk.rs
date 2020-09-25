use modern_robotics_rs::core::fk_in_space;
use modern_robotics_rs::na::{Matrix4, RowVector4, Vector6};

fn main() {
  let m = Matrix4::from_rows(&[
    RowVector4::new(0., 0., -1., 60.5),
    RowVector4::new(0., 1., 0., -40.),
    RowVector4::new(1., 0., 0., 11.5),
    RowVector4::new(0., 0., 0., 1.),
  ]);

  let s3 = Vector6::from_column_slice(&[-1., 0., 0., 0., 107., -40.]);
  let s2 = Vector6::from_column_slice(&[-1., 0., 0., 0., 0., -10.]);
  let s1 = Vector6::from_column_slice(&[0., 0., 1., 0., 0., 0.]);

  println!(
    "{}",
    fk_in_space(
      &m,
      &vec![s1, s2, s3],
      &vec![
        std::f64::consts::FRAC_PI_2,
        std::f64::consts::FRAC_PI_2,
        std::f64::consts::FRAC_PI_2
      ]
    )
  );
}
