use na::base::dimension::{Dynamic, U1, U3, U4, U6};
use na::base::{
    Matrix3, Matrix4, Matrix6, MatrixMN, RowVector1, RowVector3, RowVector4, RowVector6, Vector3,
    Vector4, Vector6,
};
use na::geometry::{Isometry3, Translation3, UnitQuaternion};
use nalgebra as na;

fn near_zero(x: f64) -> bool {
    f64::abs(x) < 1e-6
}

#[test]
fn test_near_zero() {
    let x = 1e-5;
    let y = 1e-7;

    assert_eq!(near_zero(x), false);
    assert_eq!(near_zero(y), true);
}

fn vec_to_so3(v: Vector3<f64>) -> Matrix3<f64> {
    Matrix3::from_rows(&[
        RowVector3::new(0., -v[2], v[1]),
        RowVector3::new(v[2], 0., -v[0]),
        RowVector3::new(-v[1], v[0], 0.),
    ])
}

#[test]
fn test_vec_to_so3() {
    let v = Vector3::new(1., 2., 3.);
    let m = Matrix3::from_rows(&[
        RowVector3::new(0., -3., 2.),
        RowVector3::new(3., 0., -1.),
        RowVector3::new(-2., 1., 0.),
    ]);

    assert_eq!(vec_to_so3(v), m);
}

fn so3_to_vec(m: Matrix3<f64>) -> Vector3<f64> {
    Vector3::new(m[(2, 1)], m[(0, 2)], m[(1, 0)])
}

#[test]
fn test_so3_to_vec() {
    let v = Vector3::new(1., 2., 3.);
    let m = Matrix3::from_rows(&[
        RowVector3::new(0., -3., 2.),
        RowVector3::new(3., 0., -1.),
        RowVector3::new(-2., 1., 0.),
    ]);

    assert_eq!(so3_to_vec(m), v);
}

fn axis_ang3(v: Vector3<f64>) -> (Vector3<f64>, f64) {
    (v.normalize(), v.norm())
}

#[test]
fn test_axis_ang3() {
    let v = Vector3::new(1., 2., 3.);
    let axis = Vector3::new(0.2672612419124244, 0.5345224838248488, 0.8017837257372732);
    let ang = 3.7416573867739413;

    assert_eq!(axis_ang3(v), (axis, ang));
}

fn matrix_exp3(m: Matrix3<f64>) -> Matrix3<f64> {
    let omega_theta = so3_to_vec(m);
    if near_zero(omega_theta.norm()) {
        return Matrix3::identity();
    }

    let (_, theta) = axis_ang3(omega_theta);
    let omega_mat = m / theta;

    Matrix3::identity()
        + f64::sin(theta) * omega_mat
        + (1. - f64::cos(theta)) * omega_mat * omega_mat
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

    assert_eq!(matrix_exp3(m), e);
}

fn matrix_log3(r: Matrix3<f64>) -> Matrix3<f64> {
    match (r.trace() - 1.) / 2. {
        a if a >= 1. => Matrix3::zeros(),
        a if a <= -1. => {
            let omega;
            if !near_zero(1. + r[(2, 2)]) {
                omega = (1. / f64::sqrt(2. * (1. + r[(2, 2)])))
                    * Vector3::new(r[(0, 2)], r[(1, 2)], r[(2, 2)]);
            } else if !near_zero(1. + r[(1, 1)]) {
                omega = (1. / f64::sqrt(2. * (1. + r[(1, 1)])))
                    * Vector3::new(r[(0, 1)], r[(1, 1)], r[(2, 1)]);
            } else {
                omega = (1. / f64::sqrt(2. * (1. + r[(0, 0)])))
                    * Vector3::new(r[(0, 0)], r[(1, 0)], r[(2, 0)]);
            }

            vec_to_so3(std::f64::consts::PI * omega)
        }
        a => {
            let theta = f64::acos(a);
            theta / (2.0 * f64::sin(theta)) * (r - r.transpose())
        }
    }
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

    assert_eq!(matrix_log3(m), e);
}

fn rp_to_trans(r: Matrix3<f64>, p: Vector3<f64>) -> Matrix4<f64> {
    let tra = Translation3::from(p);
    let rot = UnitQuaternion::from_matrix(&r);
    let iso = Isometry3::from_parts(tra, rot);

    iso.to_homogeneous()
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

    assert_eq!(rp_to_trans(r, p), e);
}

fn trans_to_rp(t: Matrix4<f64>) -> (Matrix3<f64>, Vector3<f64>) {
    (
        t.fixed_slice::<U3, U3>(0, 0).clone_owned(),
        t.fixed_slice::<U3, U1>(0, 3).clone_owned(),
    )
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

    assert_eq!(trans_to_rp(t), (r, p));
}

fn trans_inv(t: Matrix4<f64>) -> Matrix4<f64> {
    let (r, p) = trans_to_rp(t);
    let tra = Translation3::from(p);
    let rot = UnitQuaternion::from_matrix(&r);

    Isometry3::from_parts(tra, rot).inverse().to_homogeneous()
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
        RowVector4::new(0.9999999999999998, 0., 0., 0.),
        RowVector4::new(0., 0., 0.9999999999999998, -2.9999999999999996),
        RowVector4::new(
            0.,
            -0.9999999999999998,
            0.,
            -0.0000000000000004440892098500626,
        ),
        RowVector4::new(0., 0., 0., 1.),
    ]);

    assert_eq!(trans_inv(t), e);
}

fn vec_to_se3(v: Vector6<f64>) -> Matrix4<f64> {
    let mut m = Matrix4::zeros();
    m.fixed_slice_mut::<U3, U3>(0, 0)
        .copy_from(&vec_to_so3(Vector3::new(v[0], v[1], v[2])));
    m.fixed_slice_mut::<U3, U1>(0, 3)
        .copy_from(&Vector3::new(v[3], v[4], v[5]));
    m
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

    assert_eq!(vec_to_se3(v), m);
}

fn se_to_vec(m: Matrix4<f64>) -> Vector6<f64> {
    Vector6::new(
        m[(2, 1)],
        m[(0, 2)],
        m[(1, 0)],
        m[(0, 3)],
        m[(1, 3)],
        m[(2, 3)],
    )
}

#[test]
fn test_se_to_vec() {
    let m = Matrix4::from_rows(&[
        RowVector4::new(0., -3., 2., 4.),
        RowVector4::new(3., 0., -1., 5.),
        RowVector4::new(-2., 1., 0., 6.),
        RowVector4::new(0., 0., 0., 0.),
    ]);

    let v = Vector6::new(1., 2., 3., 4., 5., 6.);

    assert_eq!(se_to_vec(m), v);
}

fn adjoint(t: Matrix4<f64>) -> Matrix6<f64> {
    let (r, p) = trans_to_rp(t);

    let mut m = Matrix6::zeros();
    m.fixed_slice_mut::<U3, U3>(0, 0).copy_from(&r);
    m.fixed_slice_mut::<U3, U3>(3, 0)
        .copy_from(&(vec_to_so3(p) * r));
    m.fixed_slice_mut::<U3, U3>(3, 3).copy_from(&r);
    m
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

    assert_eq!(adjoint(m), e);
}

fn screw_to_axis(point: Vector3<f64>, screw: Vector3<f64>, pitch: f64) -> Vector6<f64> {
    let mut v = Vector6::zeros();
    v.fixed_slice_mut::<U3, U1>(0, 0).copy_from(&screw);
    v.fixed_slice_mut::<U3, U1>(3, 0)
        .copy_from(&(point.cross(&screw) + screw * pitch));

    v
}

#[test]
fn test_screw_to_axis() {
    let q = Vector3::new(3., 0., 0.);
    let s = Vector3::new(0., 0., 1.);
    let h = 2.;

    let v = Vector6::new(0., 0., 1., 0., -3., 2.);

    assert_eq!(screw_to_axis(q, s, h), v);
}

fn axis_ang6(v: Vector6<f64>) -> (Vector6<f64>, f64) {
    let mut theta = Vector3::new(v[0], v[1], v[2]).norm();

    if near_zero(theta) {
        theta = Vector3::new(v[3], v[4], v[5]).norm();
    }

    (v / theta, theta)
}

#[test]
fn test_axis_ang6() {
    let v = Vector6::new(1., 0., 0., 1., 2., 3.);
    let axis = Vector6::new(1., 0., 0., 1., 2., 3.);
    let ang = 1.;

    assert_eq!(axis_ang6(v), (axis, ang));
}

fn matrix_exp6(m: Matrix4<f64>) -> Matrix4<f64> {
    let r = m.fixed_slice::<U3, U3>(0, 0).clone_owned();
    let v = m.fixed_slice::<U3, U1>(0, 3).clone_owned();
    let omega_theta = so3_to_vec(r);

    let mut mm = Matrix4::identity();

    if near_zero(omega_theta.norm()) {
        mm.fixed_slice_mut::<U3, U1>(0, 3)
            .copy_from(&(m.fixed_slice::<U3, U1>(0, 3).clone_owned()));
        println!("{}", mm);
        return mm;
    }

    let (_, theta) = axis_ang3(omega_theta);
    let omega_mat = r / theta;

    let t = (Matrix3::identity() * theta
        + (1. - f64::cos(theta)) * omega_mat
        + (theta - f64::sin(theta)) * (omega_mat * omega_mat))
        * (v / theta);

    mm.fixed_slice_mut::<U3, U3>(0, 0)
        .copy_from(&(matrix_exp3(r)));
    mm.fixed_slice_mut::<U3, U1>(0, 3).copy_from(&t);
    mm
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

    assert_eq!(matrix_exp6(m), e);
}

fn matrix_log6(m: Matrix4<f64>) -> Matrix4<f64> {
    let (r, p) = trans_to_rp(m);
    let omega_mat = matrix_log3(r);

    if omega_mat == Matrix3::zeros() {
        let mut mm = Matrix4::zeros();
        mm.fixed_slice_mut::<U3, U1>(0, 3)
            .copy_from(&m.fixed_slice::<U3, U1>(0, 3).clone_owned());
        return mm;
    }

    let theta = f64::acos(r.trace() - 1.) / 2.;

    let mut mm = Matrix4::zeros();
    mm.fixed_slice_mut::<U3, U3>(0, 0).copy_from(&omega_mat);
    mm.fixed_slice_mut::<U3, U1>(0, 3).copy_from(
        &((Matrix3::identity() - omega_mat / 2.
            + (1. / theta - 1. / f64::tan(theta / 2.) / 2.) * (omega_mat * omega_mat) / theta)
            * m.fixed_slice::<U3, U1>(0, 3).clone_owned()),
    );

    mm
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
        RowVector4::new(0., 1.5707963267948966, 0., 2.376713387622238),
        RowVector4::new(0., 0., 0., 0.),
    ]);

    assert_eq!(matrix_log6(m), e);
}

fn project_to_so3(m: Matrix3<f64>) -> Matrix3<f64> {
    let svd = m.svd(true, true);
    let mut r = svd.u.expect("Could not extract u from SVD.")
        * svd.v_t.expect("Could not extract v_t from SVD.");

    let rows = r.nrows();
    let cols = svd.singular_values[2] as usize;

    let r_neg = -r.slice((0, 0), (rows, cols)).clone_owned();

    if r.determinant() < 0. {
        r.slice_mut((0, 0), (rows, cols)).copy_from(&r_neg);
    }

    r
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

    assert_eq!(project_to_so3(m), e);
}

fn project_to_se3(m: Matrix4<f64>) -> Matrix4<f64> {
    rp_to_trans(
        project_to_so3(m.fixed_slice::<U3, U3>(0, 0).clone_owned()),
        m.fixed_slice::<U3, U1>(0, 3).clone_owned(),
    )
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

    assert_eq!(project_to_se3(m), e);
}

fn distance_to_so3(m: Matrix3<f64>) -> f64 {
    if m.determinant() <= 0. {
        return 1e+9;
    }

    (m.transpose() * m - Matrix3::identity()).norm()
}

#[test]
fn test_distance_to_so3() {
    let m = Matrix3::from_rows(&[
        RowVector3::new(1., 0., 0.),
        RowVector3::new(0., 0.1, -0.95),
        RowVector3::new(0., 1., 0.1),
    ]);

    assert_eq!(distance_to_so3(m), 0.08835298523536149);
}

fn distance_to_se3(m: Matrix4<f64>) -> f64 {
    let r = m.fixed_slice::<U3, U3>(0, 0).clone_owned();
    if r.determinant() <= 0. {
        return 1e+9;
    }

    let mut mm = Matrix4::zeros();
    let row = m.fixed_slice::<U1, U4>(3, 0).clone_owned();

    mm.fixed_slice_mut::<U3, U3>(0, 0)
        .copy_from(&(r.transpose() * r));
    mm.fixed_slice_mut::<U1, U4>(3, 0).copy_from(&row);

    (mm - Matrix4::identity()).norm()
}

#[test]
fn test_distance_to_se3() {
    let m = Matrix4::from_rows(&[
        RowVector4::new(1., 0., 0., 1.2),
        RowVector4::new(0., 0.1, -0.95, 1.5),
        RowVector4::new(0., 1., 0.1, -0.9),
        RowVector4::new(0., 0., 0.1, 0.98),
    ]);

    assert_eq!(distance_to_se3(m), 0.13493053768513638);
}

fn is_so3(m: Matrix3<f64>) -> bool {
    f64::abs(distance_to_so3(m)) < 1e-3
}

#[test]
fn test_is_so3() {
    let m = Matrix3::from_rows(&[
        RowVector3::new(1., 0., 0.),
        RowVector3::new(0., 0.1, -0.95),
        RowVector3::new(0., 1., 0.1),
    ]);

    assert_eq!(is_so3(m), false);
}

fn is_se3(m: Matrix4<f64>) -> bool {
    f64::abs(distance_to_se3(m)) < 1e-3
}

#[test]
fn test_is_se3() {
    let m = Matrix4::from_rows(&[
        RowVector4::new(1., 0., 0., 1.2),
        RowVector4::new(0., 0.1, -0.95, 1.5),
        RowVector4::new(0., 1., 0.1, -0.9),
        RowVector4::new(0., 0., 0.1, 0.98),
    ]);

    assert_eq!(is_se3(m), false);
}

fn fk_in_body(
    m: Matrix4<f64>,
    b_list: MatrixMN<f64, U6, Dynamic>,
    theta_list: MatrixMN<f64, U1, Dynamic>,
) -> Matrix4<f64> {
    let mut t = m.clone();

    for i in 0..theta_list.ncols() {
        let theta = theta_list[i];
        let col = b_list.column(i).clone_owned();
        let scaled_col = col * theta;
        let screw_mat = vec_to_se3(scaled_col);
        let transformation = matrix_exp6(screw_mat);

        println!("T: {}", transformation);

        t = t * transformation;
    }
    t
}

#[test]
fn test_fk_in_body() {
    let m = Matrix4::from_rows(&[
        RowVector4::new(-1., 0., 0., 0.),
        RowVector4::new(0., 1., 0., 6.),
        RowVector4::new(0., 0., -1., 2.),
        RowVector4::new(0., 0., 0., 1.),
    ]);

    let b_list = MatrixMN::<f64, Dynamic, U6>::from_rows(&[
        RowVector6::new(0., 0., -1., 2., 0., 0.),
        RowVector6::new(0., 0., 0., 0., 1., 0.),
        RowVector6::new(0., 0., 1., 0., 0., 0.1),
    ]);

    let theta_list = MatrixMN::<f64, Dynamic, U1>::from_rows(&[
        RowVector1::new(std::f64::consts::PI / 2.),
        RowVector1::new(3.),
        RowVector1::new(std::f64::consts::PI),
    ]);

    let e = Matrix4::from_rows(&[
        RowVector4::new(-0.000000000000000011442377452219667, 1., 0., -5.),
        RowVector4::new(1., 0.000000000000000011442377452219667, 0., 4.),
        RowVector4::new(0., 0., -1., 1.6858407346410207),
        RowVector4::new(0., 0., 0., 1.),
    ]);

    assert_eq!(fk_in_body(m, b_list.transpose(), theta_list.transpose()), e);
}

fn fk_in_space(
    m: Matrix4<f64>,
    s_list: MatrixMN<f64, U6, Dynamic>,
    theta_list: MatrixMN<f64, U1, Dynamic>,
) -> Matrix4<f64> {
    let mut t = m.clone();

    for i in 0..theta_list.ncols() {
        let theta = theta_list[theta_list.ncols() - i - 1];
        let col = s_list.column(theta_list.ncols() - i - 1).clone_owned();
        let scaled_col = col * theta;
        let screw_mat = vec_to_se3(scaled_col);
        let transformation = matrix_exp6(screw_mat);

        t = transformation * t;
    }
    t
}

#[test]
fn test_fk_in_space() {
    let m = Matrix4::from_rows(&[
        RowVector4::new(-1., 0., 0., 0.),
        RowVector4::new(0., 1., 0., 6.),
        RowVector4::new(0., 0., -1., 2.),
        RowVector4::new(0., 0., 0., 1.),
    ]);

    let s_list = MatrixMN::<f64, Dynamic, U6>::from_rows(&[
        RowVector6::new(0., 0., 1., 4., 0., 0.),
        RowVector6::new(0., 0., 0., 0., 1., 0.),
        RowVector6::new(0., 0., -1., -6., 0., -0.1),
    ]);

    let theta_list = MatrixMN::<f64, Dynamic, U1>::from_rows(&[
        RowVector1::new(std::f64::consts::PI / 2.),
        RowVector1::new(3.),
        RowVector1::new(std::f64::consts::PI),
    ]);

    let e = Matrix4::from_rows(&[
        RowVector4::new(-0.000000000000000011442377452219667, 1., 0., -5.),
        RowVector4::new(1., 0.000000000000000011442377452219667, 0., 4.000000000000001),
        RowVector4::new(0., 0., -1., 1.6858407346410207),
        RowVector4::new(0., 0., 0., 1.),
    ]);

    assert_eq!(
        fk_in_space(m, s_list.transpose(), theta_list.transpose()),
        e
    );
}

// fn inverse_kinematics(target: &Matrix4<f64>, theta: &mut Vector3<f64>) {
//     let inverse_jacobian = Matrix3::identity();
//     let mut error = target - forward_kinematics(theta);
//     loop {
//         if error > 1e-12 {
//             theta += (inverse_jacobian * error);
//             error = (target - forward_kinematics(theta)).norm();
//         }
//     }
// }

// fn main() {
//     let m = Matrix4::from_columns(&[
//         Vector4::new(0., 0., 1., 0.),
//         Vector4::new(0., 1., 0., 0.),
//         Vector4::new(-1., 0., 0., 0.),
//         Vector4::new(60.5, -40., 11.5, 1.),
//     ]);

//     let s3 = Vector6::from_column_slice(&[-1., 0., 0., 0., 107., -40.]);

//     let s2 = Vector6::from_column_slice(&[-1., 0., 0., 0., 0., -10.]);

//     let s1 = Vector6::from_column_slice(&[0., 0., 1., 0., 0., 0.]);
//     let theta = Vector3::new(0., 90., 0.);

//     println!("{}", forward_kinematics(&theta))
// }
