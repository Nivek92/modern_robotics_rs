// use modern_robotics_rs::{
//   fk_in_space, Dynamic, Matrix4, MatrixMN, RowVector1, Vector4, Vector6, U1, U6,
// };

// fn main() {
//   let m = Matrix4::from_columns(&[
//     Vector4::new(0., 0., 1., 0.),
//     Vector4::new(0., 1., 0., 0.),
//     Vector4::new(-1., 0., 0., 0.),
//     Vector4::new(60.5, -40., 11.5, 1.),
//   ]);

//   let s3 = Vector6::from_column_slice(&[-1., 0., 0., 0., 107., -40.]);
//   let s2 = Vector6::from_column_slice(&[-1., 0., 0., 0., 0., -10.]);
//   let s1 = Vector6::from_column_slice(&[0., 0., 1., 0., 0., 0.]);

//   let mut s_list: MatrixMN<f64, U6, Dynamic> = MatrixMN::<f64, U6, Dynamic>::zeros(3);
//   s_list.set_column(0, &s1);
//   s_list.set_column(1, &s2);
//   s_list.set_column(2, &s3);

//   let theta_list = MatrixMN::<f64, Dynamic, U1>::from_rows(&[
//     RowVector1::new(0.),
//     RowVector1::new(std::f64::consts::FRAC_PI_2),
//     RowVector1::new(0.),
//   ]);

//   println!("{}", fk_in_space(m, &s_list, &theta_list.transpose()));
// }

fn main() {}
