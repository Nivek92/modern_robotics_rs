use na::base::dimension::{Dynamic, U1, U3, U4, U6, U8};
use na::base::{
    Matrix3, Matrix4, Matrix6, MatrixMN, RowVector1, RowVector3, RowVector4, RowVector6,
    RowVectorN, Vector3, Vector4, Vector6,
};
use na::geometry::{Isometry3, Translation3, UnitQuaternion};
use nalgebra as na;

fn near_zero(x: f64) -> bool {
    f64::abs(x) < 1e-6
}

type RowVector8 = RowVectorN<f64, U8>;

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
    let rt = r.transpose();

    let mut m = Matrix4::identity();
    m.fixed_slice_mut::<U3, U3>(0, 0).copy_from(&rt);
    m.fixed_slice_mut::<U3, U1>(0, 3).copy_from(&(-rt * p));
    m
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

fn se3_to_vec(m: Matrix4<f64>) -> Vector6<f64> {
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
fn test_se3_to_vec() {
    let m = Matrix4::from_rows(&[
        RowVector4::new(0., -3., 2., 4.),
        RowVector4::new(3., 0., -1., 5.),
        RowVector4::new(-2., 1., 0., 6.),
        RowVector4::new(0., 0., 0., 0.),
    ]);

    let v = Vector6::new(1., 2., 3., 4., 5., 6.);

    assert_eq!(se3_to_vec(m), v);
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
    let (r, _) = trans_to_rp(m);
    let omega_mat = matrix_log3(r);

    if omega_mat == Matrix3::zeros() {
        let mut mm = Matrix4::zeros();
        mm.fixed_slice_mut::<U3, U1>(0, 3)
            .copy_from(&m.fixed_slice::<U3, U1>(0, 3).clone_owned());
        return mm;
    }

    let theta = f64::acos((r.trace() - 1.) / 2.);

    let mut mm = Matrix4::zeros();
    mm.fixed_slice_mut::<U3, U3>(0, 0).copy_from(&omega_mat);
    mm.fixed_slice_mut::<U3, U1>(0, 3).copy_from(
        &(((Matrix3::identity() - omega_mat / 2.)
            + (1. / theta - 1. / f64::tan(theta / 2.) / 2.) * ((omega_mat * omega_mat) / theta))
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
        RowVector4::new(0., 1.5707963267948966, 0., 2.3561944901923453),
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
    b_list: &MatrixMN<f64, U6, Dynamic>,
    theta_list: &MatrixMN<f64, U1, Dynamic>,
) -> Matrix4<f64> {
    let mut t = m.clone();

    for i in 0..theta_list.ncols() {
        let theta = theta_list[i];
        let col = b_list.column(i).clone_owned();
        let scaled_col = col * theta;
        let screw_mat = vec_to_se3(scaled_col);
        let transformation = matrix_exp6(screw_mat);

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

    assert_eq!(
        fk_in_body(m, &b_list.transpose(), &theta_list.transpose()),
        e
    );
}

fn fk_in_space(
    m: Matrix4<f64>,
    s_list: &MatrixMN<f64, U6, Dynamic>,
    theta_list: &MatrixMN<f64, U1, Dynamic>,
) -> Matrix4<f64> {
    let mut t = m.clone();

    for i in (0..theta_list.ncols()).rev() {
        let theta = theta_list[i];
        let col = s_list.column(i).clone_owned();
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
        RowVector4::new(
            1.,
            0.000000000000000011442377452219667,
            0.,
            4.000000000000001,
        ),
        RowVector4::new(0., 0., -1., 1.6858407346410207),
        RowVector4::new(0., 0., 0., 1.),
    ]);

    assert_eq!(
        fk_in_space(m, &s_list.transpose(), &theta_list.transpose()),
        e
    );
}

fn jacobian_body(
    b_list: &MatrixMN<f64, U6, Dynamic>,
    theta_list: &MatrixMN<f64, U1, Dynamic>,
) -> MatrixMN<f64, U6, Dynamic> {
    let mut jb = b_list.clone();
    let mut t = Matrix4::identity();

    for i in (0..theta_list.ncols() - 1).rev() {
        t = t * matrix_exp6(vec_to_se3(
            b_list.column(i + 1).clone_owned() * -theta_list[i + 1],
        ));
        jb.set_column(i, &(adjoint(t) * b_list.column(i).clone_owned()));
    }

    jb
}

#[test]
fn test_jacobian_body() {
    let b_list = MatrixMN::<f64, Dynamic, U6>::from_rows(&[
        RowVector6::new(0., 0., 1., 0., 0.2, 0.2),
        RowVector6::new(1., 0., 0., 2., 0., 3.),
        RowVector6::new(0., 1., 0., 0., 2., 1.),
        RowVector6::new(1., 0., 0., 0.2, 0.3, 0.4),
    ]);

    let theta_list = MatrixMN::<f64, Dynamic, U1>::from_rows(&[
        RowVector1::new(0.2),
        RowVector1::new(1.1),
        RowVector1::new(0.1),
        RowVector1::new(1.2),
    ]);

    let e = MatrixMN::<f64, Dynamic, U4>::from_rows(&[
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

    assert_eq!(
        jacobian_body(&b_list.transpose(), &theta_list.transpose()),
        e
    );
}

fn jacobian_space(
    s_list: &MatrixMN<f64, U6, Dynamic>,
    theta_list: &MatrixMN<f64, U1, Dynamic>,
) -> MatrixMN<f64, U6, Dynamic> {
    let mut js = s_list.clone();
    let mut t = Matrix4::identity();

    for i in 1..theta_list.ncols() {
        t = t * matrix_exp6(vec_to_se3(
            s_list.column(i - 1).clone_owned() * theta_list[i - 1],
        ));
        js.set_column(i, &(adjoint(t) * s_list.column(i).clone_owned()));
    }

    js
}

#[test]
fn test_jacobian_space() {
    let b_list = MatrixMN::<f64, Dynamic, U6>::from_rows(&[
        RowVector6::new(0., 0., 1., 0., 0.2, 0.2),
        RowVector6::new(1., 0., 0., 2., 0., 3.),
        RowVector6::new(0., 1., 0., 0., 2., 1.),
        RowVector6::new(1., 0., 0., 0.2, 0.3, 0.4),
    ]);

    let theta_list = MatrixMN::<f64, Dynamic, U1>::from_rows(&[
        RowVector1::new(0.2),
        RowVector1::new(1.1),
        RowVector1::new(0.1),
        RowVector1::new(1.2),
    ]);

    let e = MatrixMN::<f64, Dynamic, U4>::from_rows(&[
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

    assert_eq!(
        jacobian_space(&b_list.transpose(), &theta_list.transpose()),
        e
    );
}

fn ik_in_body(
    m: Matrix4<f64>,
    d: Matrix4<f64>,
    b_list: MatrixMN<f64, U6, Dynamic>,
    theta_list: MatrixMN<f64, U1, Dynamic>,
    tolerance: (f64, f64),
) -> (MatrixMN<f64, U1, Dynamic>, bool) {
    let (w_tolerance, v_tolerance) = tolerance;
    let max_iterations = 20;

    let mut i = 0;
    let mut joint_configuration = theta_list.clone();

    let mut t = fk_in_body(m, &b_list, &joint_configuration);
    let mut vb = se3_to_vec(matrix_log6(trans_inv(t) * d));

    let mut e = Vector3::new(vb[0], vb[1], vb[2]).norm() > w_tolerance
        || Vector3::new(vb[3], vb[4], vb[5]).norm() > v_tolerance;

    while e && i < max_iterations {
        let pseudo_inverse = jacobian_body(&b_list, &joint_configuration)
            .pseudo_inverse(1e-15)
            .expect("Could not calculate the pseudo inverse.");
        joint_configuration = joint_configuration + (pseudo_inverse * vb).transpose();
        i += 1;
        t = fk_in_body(m, &b_list, &joint_configuration);
        vb = se3_to_vec(matrix_log6(trans_inv(t) * d));
        e = Vector3::new(vb[0], vb[1], vb[2]).norm() > w_tolerance
            || Vector3::new(vb[3], vb[4], vb[5]).norm() > v_tolerance;
    }
    (joint_configuration, !e)
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

    let b_list = MatrixMN::<f64, Dynamic, U6>::from_rows(&[
        RowVector6::new(0., 0., -1., 2., 0., 0.),
        RowVector6::new(0., 0., 0., 0., 1., 0.),
        RowVector6::new(0., 0., 1., 0., 0., 0.1),
    ]);

    let theta_list = MatrixMN::<f64, Dynamic, U1>::from_rows(&[
        RowVector1::new(1.5),
        RowVector1::new(2.5),
        RowVector1::new(3.),
    ]);

    let e = MatrixMN::<f64, Dynamic, U1>::from_rows(&[
        RowVector1::new(1.5707381937148923),
        RowVector1::new(2.999666997382942),
        RowVector1::new(3.141539129217613),
    ]);

    let w_tolerance = 0.01;
    let v_tolerance = 0.001;

    assert_eq!(
        ik_in_body(
            m,
            d,
            b_list.transpose(),
            theta_list.transpose(),
            (w_tolerance, v_tolerance)
        ),
        (e.transpose(), true)
    );
}

fn ik_in_space(
    m: Matrix4<f64>,
    d: Matrix4<f64>,
    s_list: MatrixMN<f64, U6, Dynamic>,
    theta_list: MatrixMN<f64, U1, Dynamic>,
    tolerance: (f64, f64),
) -> (MatrixMN<f64, U1, Dynamic>, bool) {
    let (w_tolerance, v_tolerance) = tolerance;
    let max_iterations = 20;

    let mut i = 0;
    let mut joint_configuration = theta_list.clone();

    let mut t = fk_in_space(m, &s_list, &joint_configuration);
    let mut vs = adjoint(t) * se3_to_vec(matrix_log6(trans_inv(t) * d));

    let mut e = Vector3::new(vs[0], vs[1], vs[2]).norm() > w_tolerance
        || Vector3::new(vs[3], vs[4], vs[5]).norm() > v_tolerance;

    while e && i < max_iterations {
        let pseudo_inverse = jacobian_space(&s_list, &joint_configuration)
            .pseudo_inverse(1e-15)
            .expect("Could not calculate the pseudo inverse.");
        joint_configuration = joint_configuration + (pseudo_inverse * vs).transpose();
        i += 1;
        t = fk_in_space(m, &s_list, &joint_configuration);
        vs = adjoint(t) * se3_to_vec(matrix_log6(trans_inv(t) * d));
        e = Vector3::new(vs[0], vs[1], vs[2]).norm() > w_tolerance
            || Vector3::new(vs[3], vs[4], vs[5]).norm() > v_tolerance;
    }
    (joint_configuration, !e)
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

    let s_list = MatrixMN::<f64, Dynamic, U6>::from_rows(&[
        RowVector6::new(0., 0., 1., 4., 0., 0.),
        RowVector6::new(0., 0., 0., 0., 1., 0.),
        RowVector6::new(0., 0., -1., -6., 0., -0.1),
    ]);

    let theta_list = MatrixMN::<f64, Dynamic, U1>::from_rows(&[
        RowVector1::new(1.5),
        RowVector1::new(2.5),
        RowVector1::new(3.),
    ]);

    let e = MatrixMN::<f64, Dynamic, U1>::from_rows(&[
        RowVector1::new(1.57073782965672),
        RowVector1::new(2.9996638446725234),
        RowVector1::new(3.141534199856583),
    ]);

    let w_tolerance = 0.01;
    let v_tolerance = 0.001;

    assert_eq!(
        ik_in_space(
            m,
            d,
            s_list.transpose(),
            theta_list.transpose(),
            (w_tolerance, v_tolerance)
        ),
        (e.transpose(), true)
    );
}

fn ad(v: Vector6<f64>) -> Matrix6<f64> {
    let m1 = vec_to_so3(Vector3::new(v[0], v[1], v[2]));
    let m2 = vec_to_so3(Vector3::new(v[3], v[4], v[5]));

    let mut m = Matrix6::zeros();

    m.fixed_slice_mut::<U3, U3>(0, 0).copy_from(&m1);
    m.fixed_slice_mut::<U3, U3>(3, 0).copy_from(&m2);
    m.fixed_slice_mut::<U3, U3>(3, 3).copy_from(&m1);
    m
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

    assert_eq!(ad(v), e);
}

fn cubic_time_scaling(tf: f64, t: f64) -> f64 {
    3. * f64::powi(1. * t / tf, 2) - 2. * f64::powi(1. * t / tf, 3)
}

#[test]
fn test_cubic_time_scaling() {
    let tf = 2.;
    let t = 0.6;
    let e = 0.21600000000000003;

    assert_eq!(cubic_time_scaling(tf, t), e);
}

fn quintic_time_scaling(tf: f64, t: f64) -> f64 {
    10. * f64::powi(1. * t / tf, 3) - 15. * f64::powi(1. * t / tf, 4)
        + 6. * f64::powi(1. * t / tf, 5)
}

#[test]
fn test_v() {
    let tf = 2.;
    let t = 0.6;
    let e = 0.16308000000000003;

    assert_eq!(quintic_time_scaling(tf, t), e);
}

enum TimeScalingMethod {
    cubic,
    quintic,
}

fn joint_trajectory(
    theta_start: MatrixMN<f64, U1, Dynamic>,
    theta_end: MatrixMN<f64, U1, Dynamic>,
    tf: f64,
    n: u32,
    method: TimeScalingMethod,
) -> MatrixMN<f64, Dynamic, Dynamic> {
    let time_gap = tf / (n - 1) as f64;
    let mut trajectory: MatrixMN<f64, Dynamic, Dynamic> =
        MatrixMN::<f64, Dynamic, Dynamic>::zeros(theta_start.ncols(), n as usize);

    let scaling = match method {
        TimeScalingMethod::cubic => cubic_time_scaling,
        TimeScalingMethod::quintic => quintic_time_scaling,
    };

    for i in 0..n {
        let s = scaling(tf, time_gap * i as f64);
        trajectory.set_column(
            i as usize,
            &((s * &theta_end + (1. - s) * &theta_start).transpose()),
        );
    }

    trajectory.transpose()
}

#[test]
fn test_joint_trajectory() {
    let theta_start = MatrixMN::<f64, Dynamic, U1>::from_rows(&[
        RowVector1::new(1.),
        RowVector1::new(0.),
        RowVector1::new(0.),
        RowVector1::new(1.),
        RowVector1::new(1.),
        RowVector1::new(0.2),
        RowVector1::new(0.),
        RowVector1::new(1.),
    ]);

    let theta_end = MatrixMN::<f64, Dynamic, U1>::from_rows(&[
        RowVector1::new(1.2),
        RowVector1::new(0.5),
        RowVector1::new(0.6),
        RowVector1::new(1.1),
        RowVector1::new(2.),
        RowVector1::new(2.),
        RowVector1::new(0.9),
        RowVector1::new(1.),
    ]);

    let tf = 4.;
    let n = 6;
    let method = TimeScalingMethod::cubic;

    let e = MatrixMN::<f64, Dynamic, U8>::from_rows(&[
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

    assert_eq!(
        joint_trajectory(
            theta_start.transpose(),
            theta_end.transpose(),
            tf,
            n,
            method
        ),
        e
    )
}

fn screw_trajectory(
    x_start: Matrix4<f64>,
    x_end: Matrix4<f64>,
    tf: f64,
    n: u32,
    method: TimeScalingMethod,
) -> Vec<Matrix4<f64>> {
    let time_gap = tf / (n - 1) as f64;
    let mut trajectory: Vec<Matrix4<f64>> = vec![Matrix4::zeros(); n as usize];

    let scaling = match method {
        TimeScalingMethod::cubic => cubic_time_scaling,
        TimeScalingMethod::quintic => quintic_time_scaling,
    };

    for i in 0..n as usize {
        let s = scaling(tf, time_gap * i as f64);
        let a = trans_inv(x_start);
        let b = a * x_end;
        let c = matrix_log6(b) * s;
        trajectory[i] = x_start * matrix_exp6(c);
    }

    trajectory
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
        screw_trajectory(x_start, x_end, tf, n, method),
        vec![x_start, e1, e2, e3]
    );
}

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
