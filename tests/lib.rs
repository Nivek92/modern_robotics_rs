#[cfg(test)]
mod tests {

  use modern_robotics_rs::core::*;
  pub use nalgebra::base::dimension::{Dynamic, U1, U3, U4, U6, U8};
  pub use nalgebra::base::{
    Matrix3, Matrix4, Matrix6, MatrixMN, RowVector1, RowVector3, RowVector4, RowVector6,
    RowVectorN, Vector3, Vector4, Vector6,
  };
  pub use nalgebra::geometry::{Isometry3, Translation3, UnitQuaternion};
  pub type RowVector8 = RowVectorN<f32, U8>;

  #[test]
  fn test_columns_to_vec() {
    let columns = MatrixMN::<f32, Dynamic, U1>::from_rows(&[
      RowVector1::new(1.),
      RowVector1::new(2.),
      RowVector1::new(3.),
    ]);

    let vec = vec![1., 2., 3.];

    assert_eq!(columns_to_vec(&columns.transpose()), vec)
  }

  #[test]
  fn test_vec_to_columns() {
    let columns = MatrixMN::<f32, Dynamic, U1>::from_rows(&[
      RowVector1::new(1.),
      RowVector1::new(2.),
      RowVector1::new(3.),
    ]);

    let vec = vec![1., 2., 3.];

    assert_eq!(vec_to_columns(&vec), columns.transpose())
  }

  #[test]
  fn test_near_zero() {
    let x = 1e-5;
    let y = 1e-7;

    assert_eq!(near_zero(x), false);
    assert_eq!(near_zero(y), true);
  }

  #[test]
  fn test_vec_to_so3() {
    let v = Vector3::new(1., 2., 3.);
    let m = Matrix3::from_rows(&[
      RowVector3::new(0., -3., 2.),
      RowVector3::new(3., 0., -1.),
      RowVector3::new(-2., 1., 0.),
    ]);

    assert_eq!(vec_to_so3(&v), m);
  }

  #[test]
  fn test_so3_to_vec() {
    let v = Vector3::new(1., 2., 3.);
    let m = Matrix3::from_rows(&[
      RowVector3::new(0., -3., 2.),
      RowVector3::new(3., 0., -1.),
      RowVector3::new(-2., 1., 0.),
    ]);

    assert_eq!(so3_to_vec(&m), v);
  }

  #[test]
  fn test_axis_ang3() {
    let v = Vector3::new(1., 2., 3.);
    let axis = Vector3::new(0.2672612419124244, 0.5345224838248488, 0.8017837257372732);
    let ang = 3.7416573867739413;

    assert_eq!(axis_ang3(&v), (axis, ang));
  }

  #[test]
  fn test_matrix_exp3() {
    let m = Matrix3::from_rows(&[
      RowVector3::new(0., -3., 2.),
      RowVector3::new(3., 0., -1.),
      RowVector3::new(-2., 1., 0.),
    ]);

    let e = Matrix3::from_rows(&[
      RowVector3::new(-0.694920557641312, 0.7135209905277877, 0.08929285886191213),
      RowVector3::new(
        -0.19200697279199935,
        -0.30378504433947073,
        0.9331923538236468,
      ),
      RowVector3::new(0.6929781677417702, 0.6313496993837179, 0.34810747783026463),
    ]);

    assert_eq!(matrix_exp3(&m), e);
  }

  #[test]
  fn test_matrix_log3() {
    let m = Matrix3::from_rows(&[
      RowVector3::new(0., 0., 1.),
      RowVector3::new(1., 0., 0.),
      RowVector3::new(0., 1., 0.),
    ]);

    let a = 1.2091995761561456;

    let e = Matrix3::from_rows(&[
      RowVector3::new(0., -a, a),
      RowVector3::new(a, 0., -a),
      RowVector3::new(-a, a, 0.),
    ]);

    assert_eq!(matrix_log3(&m), e);
  }

  #[test]
  fn test_rp_to_trans() {
    let r = Matrix3::from_rows(&[
      RowVector3::new(1., 0., 0.),
      RowVector3::new(0., 0., -1.),
      RowVector3::new(0., 1., 0.),
    ]);

    let p = Vector3::new(1., 2., 5.);

    let e = Matrix4::from_rows(&[
      RowVector4::new(0.9999999999999998, 0., 0., 1.),
      RowVector4::new(0., 0., -0.9999999999999998, 2.),
      RowVector4::new(0., 0.9999999999999998, 0., 5.),
      RowVector4::new(0., 0., 0., 1.),
    ]);

    assert_eq!(rp_to_trans(&r, &p), e);
  }

  #[test]
  fn test_trans_to_rp() {
    let t = Matrix4::from_rows(&[
      RowVector4::new(1., 0., 0., 1.),
      RowVector4::new(0., 0., -1., 2.),
      RowVector4::new(0., 1., 0., 5.),
      RowVector4::new(0., 0., 0., 1.),
    ]);

    let r = Matrix3::from_rows(&[
      RowVector3::new(1., 0., 0.),
      RowVector3::new(0., 0., -1.),
      RowVector3::new(0., 1., 0.),
    ]);

    let p = Vector3::new(1., 2., 5.);

    assert_eq!(trans_to_rp(&t), (r, p));
  }

  #[test]
  fn test_trans_inv() {
    let t = Matrix4::from_rows(&[
      RowVector4::new(1., 0., 0., 0.),
      RowVector4::new(0., 0., -1., 0.),
      RowVector4::new(0., 1., 0., 3.),
      RowVector4::new(0., 0., 0., 1.),
    ]);

    let e = Matrix4::from_rows(&[
      RowVector4::new(1., 0., 0., 0.),
      RowVector4::new(0., 0., 1., -3.),
      RowVector4::new(0., -1., 0., 0.),
      RowVector4::new(0., 0., 0., 1.),
    ]);

    assert_eq!(trans_inv(&t), e);
  }

  #[test]
  fn test_vec_to_se3() {
    let v = Vector6::new(1., 2., 3., 4., 5., 6.);

    let m = Matrix4::from_rows(&[
      RowVector4::new(0., -3., 2., 4.),
      RowVector4::new(3., 0., -1., 5.),
      RowVector4::new(-2., 1., 0., 6.),
      RowVector4::new(0., 0., 0., 0.),
    ]);

    assert_eq!(vec_to_se3(&v), m);
  }

  #[test]
  fn test_se3_to_vec() {
    let m = Matrix4::from_rows(&[
      RowVector4::new(0., -3., 2., 4.),
      RowVector4::new(3., 0., -1., 5.),
      RowVector4::new(-2., 1., 0., 6.),
      RowVector4::new(0., 0., 0., 0.),
    ]);

    let v = Vector6::new(1., 2., 3., 4., 5., 6.);

    assert_eq!(se3_to_vec(&m), v);
  }

  #[test]
  fn test_adjoint() {
    let m = Matrix4::from_rows(&[
      RowVector4::new(1., 0., 0., 0.),
      RowVector4::new(0., 0., -1., 0.),
      RowVector4::new(0., 1., 0., 3.),
      RowVector4::new(0., 0., 0., 1.),
    ]);

    let e = Matrix6::from_rows(&[
      RowVector6::new(1., 0., 0., 0., 0., 0.),
      RowVector6::new(0., 0., -1., 0., 0., 0.),
      RowVector6::new(0., 1., 0., 0., 0., 0.),
      RowVector6::new(0., 0., 3., 1., 0., 0.),
      RowVector6::new(3., 0., 0., 0., 0., -1.),
      RowVector6::new(0., 0., 0., 0., 1., 0.),
    ]);

    assert_eq!(adjoint(&m), e);
  }

  #[test]
  fn test_screw_to_axis() {
    let q = Vector3::new(3., 0., 0.);
    let s = Vector3::new(0., 0., 1.);
    let h = 2.;

    let v = Vector6::new(0., 0., 1., 0., -3., 2.);

    assert_eq!(screw_to_axis(&q, &s, h), v);
  }

  #[test]
  fn test_axis_ang6() {
    let v = Vector6::new(1., 0., 0., 1., 2., 3.);
    let axis = Vector6::new(1., 0., 0., 1., 2., 3.);
    let ang = 1.;

    assert_eq!(axis_ang6(&v), (axis, ang));
  }

  #[test]
  fn test_matrix_exp6() {
    let m = Matrix4::from_rows(&[
      RowVector4::new(0., 0., 0., 0.),
      RowVector4::new(0., 0., -1.57079632, 2.35619449),
      RowVector4::new(0., 1.57079632, 0., 2.35619449),
      RowVector4::new(0., 0., 0., 0.),
    ]);

    let e = Matrix4::from_rows(&[
      RowVector4::new(1., 0., 0., 0.),
      RowVector4::new(
        0.,
        0.0000000067948967563680185,
        -1.,
        0.000000010192345190063179,
      ),
      RowVector4::new(0., 1., 0.0000000067948967563680185, 3.0000000025400504),
      RowVector4::new(0., 0., 0., 1.),
    ]);

    assert_eq!(matrix_exp6(&m), e);
  }

  #[test]
  fn test_matrix_log6() {
    let m = Matrix4::from_rows(&[
      RowVector4::new(1., 0., 0., 0.),
      RowVector4::new(0., 0., -1., 0.),
      RowVector4::new(0., 1., 0., 3.),
      RowVector4::new(0., 0., 0., 1.),
    ]);

    let e = Matrix4::from_rows(&[
      RowVector4::new(0., 0., 0., 0.),
      RowVector4::new(0., 0., -1.5707963267948966, 2.356194490192345),
      RowVector4::new(0., 1.5707963267948966, 0., 2.3561944901923453),
      RowVector4::new(0., 0., 0., 0.),
    ]);

    assert_eq!(matrix_log6(&m), e);
  }

  #[test]
  fn test_project_to_so3() {
    let m = Matrix3::from_rows(&[
      RowVector3::new(0.675, 0.150, 0.720),
      RowVector3::new(0.370, 0.771, -0.511),
      RowVector3::new(-0.630, 0.619, 0.472),
    ]);

    let e = Matrix3::from_rows(&[
      RowVector3::new(0.6790113606772366, 0.14894516151550874, 0.7188594514453904),
      RowVector3::new(0.3732070788228454, 0.7731958439349471, -0.5127227937572542),
      RowVector3::new(-0.6321867195597889, 0.616428037797474, 0.46942137342625173),
    ]);

    assert_eq!(project_to_so3(&m), e);
  }

  #[test]
  fn test_project_to_se3() {
    let m = Matrix4::from_rows(&[
      RowVector4::new(0.675, 0.150, 0.720, 1.2),
      RowVector4::new(0.370, 0.771, -0.511, 5.4),
      RowVector4::new(-0.630, 0.619, 0.472, 3.6),
      RowVector4::new(0.003, 0.002, 0.010, 0.9),
    ]);

    let e = Matrix4::from_rows(&[
      RowVector4::new(
        0.6790113606772366,
        0.1489451615155089,
        0.7188594514453908,
        1.2,
      ),
      RowVector4::new(
        0.3732070788228456,
        0.7731958439349473,
        -0.5127227937572543,
        5.4,
      ),
      RowVector4::new(
        -0.632186719559789,
        0.6164280377974745,
        0.46942137342625256,
        3.6,
      ),
      RowVector4::new(0., 0., 0., 1.),
    ]);

    assert_eq!(project_to_se3(&m), e);
  }

  #[test]
  fn test_distance_to_so3() {
    let m = Matrix3::from_rows(&[
      RowVector3::new(1., 0., 0.),
      RowVector3::new(0., 0.1, -0.95),
      RowVector3::new(0., 1., 0.1),
    ]);

    assert_eq!(distance_to_so3(&m), 0.08835298523536149);
  }

  #[test]
  fn test_distance_to_se3() {
    let m = Matrix4::from_rows(&[
      RowVector4::new(1., 0., 0., 1.2),
      RowVector4::new(0., 0.1, -0.95, 1.5),
      RowVector4::new(0., 1., 0.1, -0.9),
      RowVector4::new(0., 0., 0.1, 0.98),
    ]);

    assert_eq!(distance_to_se3(&m), 0.13493053768513638);
  }

  #[test]
  fn test_is_so3() {
    let m = Matrix3::from_rows(&[
      RowVector3::new(1., 0., 0.),
      RowVector3::new(0., 0.1, -0.95),
      RowVector3::new(0., 1., 0.1),
    ]);

    assert_eq!(is_so3(&m), false);
  }

  #[test]
  fn test_is_se3() {
    let m = Matrix4::from_rows(&[
      RowVector4::new(1., 0., 0., 1.2),
      RowVector4::new(0., 0.1, -0.95, 1.5),
      RowVector4::new(0., 1., 0.1, -0.9),
      RowVector4::new(0., 0., 0.1, 0.98),
    ]);

    assert_eq!(is_se3(&m), false);
  }

  #[test]
  fn test_fk_in_body() {
    let m = Matrix4::from_rows(&[
      RowVector4::new(-1., 0., 0., 0.),
      RowVector4::new(0., 1., 0., 6.),
      RowVector4::new(0., 0., -1., 2.),
      RowVector4::new(0., 0., 0., 1.),
    ]);

    let b_list = vec![
      Vector6::new(0., 0., -1., 2., 0., 0.),
      Vector6::new(0., 0., 0., 0., 1., 0.),
      Vector6::new(0., 0., 1., 0., 0., 0.1),
    ];

    let theta_list = vec![std::f32::consts::PI / 2., 3., std::f32::consts::PI];

    let e = Matrix4::from_rows(&[
      RowVector4::new(-0.000000000000000011442377452219667, 1., 0., -5.),
      RowVector4::new(1., 0.000000000000000011442377452219667, 0., 4.),
      RowVector4::new(0., 0., -1., 1.6858407346410207),
      RowVector4::new(0., 0., 0., 1.),
    ]);

    assert_eq!(fk_in_body(&m, &b_list, &theta_list), e);
  }

  #[test]
  fn test_fk_in_space() {
    let m = Matrix4::from_rows(&[
      RowVector4::new(-1., 0., 0., 0.),
      RowVector4::new(0., 1., 0., 6.),
      RowVector4::new(0., 0., -1., 2.),
      RowVector4::new(0., 0., 0., 1.),
    ]);

    let s_list = vec![
      Vector6::new(0., 0., 1., 4., 0., 0.),
      Vector6::new(0., 0., 0., 0., 1., 0.),
      Vector6::new(0., 0., -1., -6., 0., -0.1),
    ];

    let theta_list = vec![std::f32::consts::PI / 2., 3., std::f32::consts::PI];

    let e = Matrix4::from_rows(&[
      RowVector4::new(-0.000000000000000011442377452219667, 1., 0., -5.),
      RowVector4::new(
        1.,
        0.000000000000000011442377452219667,
        0.,
        4.000000000000001,
      ),
      RowVector4::new(0., 0., -1., 1.6858407346410207),
      RowVector4::new(0., 0., 0., 1.),
    ]);

    assert_eq!(fk_in_space(&m, &s_list, &theta_list), e);
  }

  #[test]
  fn test_jacobian_body() {
    let b_list = vec![
      Vector6::new(0., 0., 1., 0., 0.2, 0.2),
      Vector6::new(1., 0., 0., 2., 0., 3.),
      Vector6::new(0., 1., 0., 0., 2., 1.),
      Vector6::new(1., 0., 0., 0.2, 0.3, 0.4),
    ];

    let theta_list = vec![0.2, 1.1, 0.1, 1.2];
    let e = MatrixMN::<f32, Dynamic, U4>::from_rows(&[
      RowVector4::new(-0.04528405057966491, 0.9950041652780258, 0., 1.),
      RowVector4::new(
        0.7435931265563965,
        0.09304864640049498,
        0.3623577544766736,
        0.,
      ),
      RowVector4::new(
        -0.6670971570177624,
        0.03617541267787882,
        -0.9320390859672263,
        0.,
      ),
      RowVector4::new(
        2.3258604714595155,
        1.668090004953633,
        0.5641083080438885,
        0.2,
      ),
      RowVector4::new(
        -1.4432116718196155,
        2.945612749911765,
        1.4330652142884392,
        0.3,
      ),
      RowVector4::new(
        -2.0663956487602655,
        1.8288172246233696,
        -1.5886862785321807,
        0.4,
      ),
    ]);

    assert_eq!(jacobian_body(&b_list, &theta_list), e);
  }

  #[test]
  fn test_jacobian_space() {
    let b_list = vec![
      Vector6::new(0., 0., 1., 0., 0.2, 0.2),
      Vector6::new(1., 0., 0., 2., 0., 3.),
      Vector6::new(0., 1., 0., 0., 2., 1.),
      Vector6::new(1., 0., 0., 0.2, 0.3, 0.4),
    ];

    let theta_list = vec![0.2, 1.1, 0.1, 1.2];
    let e = MatrixMN::<f32, Dynamic, U4>::from_rows(&[
      RowVector4::new(
        0.,
        0.9800665778412416,
        -0.09011563789485476,
        0.957494264730031,
      ),
      RowVector4::new(
        0.,
        0.19866933079506122,
        0.4445543984476258,
        0.28487556541794845,
      ),
      RowVector4::new(1., 0., 0.8912073600614354, -0.04528405057966491),
      RowVector4::new(
        0.,
        1.9521863824506809,
        -2.216352156896298,
        -0.5116153729819477,
      ),
      RowVector4::new(
        0.2,
        0.4365413247037721,
        -2.437125727653321,
        2.7753571339551537,
      ),
      RowVector4::new(
        0.2,
        2.960266133840988,
        3.2357306532803083,
        2.2251244335357394,
      ),
    ]);

    assert_eq!(jacobian_space(&b_list, &theta_list), e);
  }

  #[test]
  fn test_ik_in_body() {
    let m = Matrix4::from_rows(&[
      RowVector4::new(-1., 0., 0., 0.),
      RowVector4::new(0., 1., 0., 6.),
      RowVector4::new(0., 0., -1., 2.),
      RowVector4::new(0., 0., 0., 1.),
    ]);

    let d = Matrix4::from_rows(&[
      RowVector4::new(0., 1., 0., -5.),
      RowVector4::new(1., 0., 0., 4.),
      RowVector4::new(0., 0., -1., 1.6858),
      RowVector4::new(0., 0., 0., 1.),
    ]);

    let b_list = vec![
      Vector6::new(0., 0., -1., 2., 0., 0.),
      Vector6::new(0., 0., 0., 0., 1., 0.),
      Vector6::new(0., 0., 1., 0., 0., 0.1),
    ];

    let theta_list = vec![1.5, 2.5, 3.];

    let e = MatrixMN::<f32, Dynamic, U1>::from_rows(&[
      RowVector1::new(1.5707381937148923),
      RowVector1::new(2.999666997382942),
      RowVector1::new(3.141539129217613),
    ]);

    let w_tolerance = 0.01;
    let v_tolerance = 0.001;

    assert_eq!(
      ik_in_body(&m, &d, &b_list, &theta_list, (w_tolerance, v_tolerance)),
      (e.transpose(), true)
    );
  }

  #[test]
  fn test_ik_in_space() {
    let m = Matrix4::from_rows(&[
      RowVector4::new(-1., 0., 0., 0.),
      RowVector4::new(0., 1., 0., 6.),
      RowVector4::new(0., 0., -1., 2.),
      RowVector4::new(0., 0., 0., 1.),
    ]);

    let d = Matrix4::from_rows(&[
      RowVector4::new(0., 1., 0., -5.),
      RowVector4::new(1., 0., 0., 4.),
      RowVector4::new(0., 0., -1., 1.6858),
      RowVector4::new(0., 0., 0., 1.),
    ]);

    let s_list = vec![
      Vector6::new(0., 0., 1., 4., 0., 0.),
      Vector6::new(0., 0., 0., 0., 1., 0.),
      Vector6::new(0., 0., -1., -6., 0., -0.1),
    ];

    let theta_list = vec![1.5, 2.5, 3.];

    let e = MatrixMN::<f32, Dynamic, U1>::from_rows(&[
      RowVector1::new(1.57073782965672),
      RowVector1::new(2.9996638446725234),
      RowVector1::new(3.141534199856583),
    ]);

    let w_tolerance = 0.01;
    let v_tolerance = 0.001;

    assert_eq!(
      ik_in_space(&m, &d, &s_list, &theta_list, (w_tolerance, v_tolerance)),
      (e.transpose(), true)
    );
  }

  #[test]
  fn test_ad() {
    let v = Vector6::new(1., 2., 3., 4., 5., 6.);

    let e = Matrix6::from_rows(&[
      RowVector6::new(0., -3., 2., 0., 0., 0.),
      RowVector6::new(3., 0., -1., 0., 0., 0.),
      RowVector6::new(-2., 1., 0., 0., 0., 0.),
      RowVector6::new(0., -6., 5., 0., -3., 2.),
      RowVector6::new(6., 0., -4., 3., 0., -1.),
      RowVector6::new(-5., 4., 0., -2., 1., 0.),
    ]);

    assert_eq!(ad(&v), e);
  }

  #[test]
  fn test_cubic_time_scaling() {
    let tf = 2.;
    let t = 0.6;
    let e = 0.21600000000000003;

    assert_eq!(cubic_time_scaling(tf, t), e);
  }

  #[test]
  fn test_quintic_time_scaling() {
    let tf = 2.;
    let t = 0.6;
    let e = 0.16308000000000003;

    assert_eq!(quintic_time_scaling(tf, t), e);
  }

  #[test]
  fn test_joint_trajectory() {
    let theta_start = vec![1., 0., 0., 1., 1., 0.2, 0., 1.];

    let theta_end = vec![1.2, 0.5, 0.6, 1.1, 2., 2., 0.9, 1.];

    let tf = 4.;
    let n = 6;
    let method = TimeScalingMethod::cubic;

    let e = MatrixMN::<f32, Dynamic, U8>::from_rows(&[
      RowVector8::from_row_slice(&[1., 0., 0., 1., 1., 0.2, 0., 1.]),
      RowVector8::from_row_slice(&[
        1.0208,
        0.05200000000000001,
        0.06240000000000001,
        1.0104,
        1.104,
        0.3872000000000001,
        0.09360000000000002,
        1.,
      ]),
      RowVector8::from_row_slice(&[
        1.0704,
        0.17600000000000005,
        0.21120000000000005,
        1.0352000000000001,
        1.352,
        0.8336000000000001,
        0.3168000000000001,
        1.,
      ]),
      RowVector8::from_row_slice(&[
        1.1296,
        0.32400000000000007,
        0.3888000000000001,
        1.0648,
        1.6480000000000001,
        1.3664000000000003,
        0.5832000000000002,
        1.,
      ]),
      RowVector8::from_row_slice(&[
        1.1792,
        0.44800000000000006,
        0.5376000000000001,
        1.0896000000000001,
        1.8960000000000001,
        1.8128000000000002,
        0.8064000000000001,
        1.,
      ]),
      RowVector8::from_row_slice(&[1.2, 0.5, 0.6, 1.1, 2., 2., 0.9, 1.]),
    ]);

    assert_eq!(joint_trajectory(&theta_start, &theta_end, tf, n, method), e)
  }

  #[test]
  fn test_screw_trajectory() {
    let x_start = Matrix4::from_rows(&[
      RowVector4::new(1., 0., 0., 1.),
      RowVector4::new(0., 1., 0., 0.),
      RowVector4::new(0., 0., 1., 1.),
      RowVector4::new(0., 0., 0., 1.),
    ]);

    let x_end = Matrix4::from_rows(&[
      RowVector4::new(0., 0., 1., 0.1),
      RowVector4::new(1., 0., 0., 0.),
      RowVector4::new(0., 1., 0., 4.1),
      RowVector4::new(0., 0., 0., 1.),
    ]);

    let tf = 5.;
    let n = 4;
    let method = TimeScalingMethod::cubic;

    let e1 = Matrix4::from_rows(&[
      RowVector4::new(
        0.9041112663535109,
        -0.25037215420139397,
        0.34626088784788306,
        0.44095776022422584,
      ),
      RowVector4::new(
        0.34626088784788306,
        0.9041112663535109,
        -0.25037215420139397,
        0.5287462373116416,
      ),
      RowVector4::new(
        -0.25037215420139397,
        0.34626088784788306,
        0.9041112663535109,
        1.600666372834503,
      ),
      RowVector4::new(0., 0., 0., 1.),
    ]);
    let e2 = Matrix4::from_rows(&[
      RowVector4::new(
        0.34626088784788267,
        -0.25037215420139375,
        0.9041112663535111,
        -0.1171114382485472,
      ),
      RowVector4::new(
        0.9041112663535111,
        0.34626088784788267,
        -0.25037215420139375,
        0.47274237949393444,
      ),
      RowVector4::new(
        -0.25037215420139375,
        0.9041112663535111,
        0.34626088784788267,
        3.2739986883842422,
      ),
      RowVector4::new(0., 0., 0., 1.),
    ]);

    let e3 = Matrix4::from_rows(&[
      RowVector4::new(
        -0.0000000000000002220446049250313,
        0.0000000000000003885780586188048,
        0.9999999999999998,
        0.10000000000000031,
      ),
      RowVector4::new(
        0.9999999999999998,
        -0.0000000000000002220446049250313,
        0.0000000000000003885780586188048,
        -0.00000000000000011102230246251565,
      ),
      RowVector4::new(
        0.0000000000000003885780586188048,
        0.9999999999999998,
        -0.0000000000000002220446049250313,
        4.1,
      ),
      RowVector4::new(0., 0., 0., 1.),
    ]);

    assert_eq!(
      screw_trajectory(&x_start, &x_end, tf, n, method),
      vec![x_start, e1, e2, e3]
    );
  }

  #[test]
  fn test_cartesian_trajectory() {
    let x_start = Matrix4::from_rows(&[
      RowVector4::new(1., 0., 0., 1.),
      RowVector4::new(0., 1., 0., 0.),
      RowVector4::new(0., 0., 1., 1.),
      RowVector4::new(0., 0., 0., 1.),
    ]);

    let x_end = Matrix4::from_rows(&[
      RowVector4::new(0., 0., 1., 0.1),
      RowVector4::new(1., 0., 0., 0.),
      RowVector4::new(0., 1., 0., 4.1),
      RowVector4::new(0., 0., 0., 1.),
    ]);

    let tf = 5.;
    let n = 4;
    let method = TimeScalingMethod::quintic;

    let e1 = Matrix4::from_rows(&[
      RowVector4::new(
        0.9366247432120836,
        -0.21400107588081244,
        0.27737633266872885,
        0.8111111111111111,
      ),
      RowVector4::new(
        0.27737633266872885,
        0.9366247432120836,
        -0.21400107588081244,
        0.,
      ),
      RowVector4::new(
        -0.21400107588081244,
        0.27737633266872885,
        0.9366247432120836,
        1.6506172839506172,
      ),
      RowVector4::new(0., 0., 0., 1.),
    ]);
    let e2 = Matrix4::from_rows(&[
      RowVector4::new(
        0.2773763326687284,
        -0.21400107588081219,
        0.9366247432120838,
        0.2888888888888889,
      ),
      RowVector4::new(
        0.9366247432120838,
        0.2773763326687284,
        -0.21400107588081219,
        0.,
      ),
      RowVector4::new(
        -0.21400107588081219,
        0.9366247432120838,
        0.2773763326687284,
        3.4493827160493824,
      ),
      RowVector4::new(0., 0., 0., 1.),
    ]);

    let e3 = Matrix4::from_rows(&[
      RowVector4::new(
        -0.0000000000000002220446049250313,
        0.0000000000000003885780586188048,
        0.9999999999999998,
        0.1,
      ),
      RowVector4::new(
        0.9999999999999998,
        -0.0000000000000002220446049250313,
        0.0000000000000003885780586188048,
        0.,
      ),
      RowVector4::new(
        0.0000000000000003885780586188048,
        0.9999999999999998,
        -0.0000000000000002220446049250313,
        4.1,
      ),
      RowVector4::new(0., 0., 0., 1.),
    ]);

    assert_eq!(
      cartesian_trajectory(&x_start, &x_end, tf, n, method),
      vec![x_start, e1, e2, e3]
    );
  }
}
