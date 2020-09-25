pub mod na {
    pub use nalgebra::base::dimension::{Dynamic, U1, U3, U4, U6, U8};
    pub use nalgebra::base::{
        Matrix3, Matrix4, Matrix6, MatrixMN, RowVector1, RowVector3, RowVector4, RowVector6,
        RowVectorN, Vector3, Vector4, Vector6,
    };
    pub use nalgebra::geometry::{Isometry3, Translation3, UnitQuaternion};
    pub type RowVector8 = RowVectorN<f64, U8>;
}

pub mod core {

    use super::na::*;

    pub fn columns_to_vec(columns: &MatrixMN<f64, U1, Dynamic>) -> Vec<f64> {
        let mut vec = vec![];

        for col in columns.iter() {
            vec.push(*col);
        }

        vec
    }

    pub fn vec_to_columns(vec: &Vec<f64>) -> MatrixMN<f64, U1, Dynamic> {
        MatrixMN::<f64, U1, Dynamic>::from_column_slice(&vec)
    }

    pub fn near_zero(x: f64) -> bool {
        f64::abs(x) < 1e-6
    }

    pub fn vec_to_so3(v: &Vector3<f64>) -> Matrix3<f64> {
        Matrix3::from_rows(&[
            RowVector3::new(0., -v[2], v[1]),
            RowVector3::new(v[2], 0., -v[0]),
            RowVector3::new(-v[1], v[0], 0.),
        ])
    }

    pub fn so3_to_vec(m: &Matrix3<f64>) -> Vector3<f64> {
        Vector3::new(m[(2, 1)], m[(0, 2)], m[(1, 0)])
    }

    pub fn axis_ang3(v: &Vector3<f64>) -> (Vector3<f64>, f64) {
        (v.normalize(), v.norm())
    }

    pub fn matrix_exp3(m: &Matrix3<f64>) -> Matrix3<f64> {
        let omega_theta = so3_to_vec(m);
        if near_zero(omega_theta.norm()) {
            return Matrix3::identity();
        }

        let (_, theta) = axis_ang3(&omega_theta);
        let omega_mat = m / theta;

        Matrix3::identity()
            + f64::sin(theta) * omega_mat
            + (1. - f64::cos(theta)) * omega_mat * omega_mat
    }

    pub fn matrix_log3(r: &Matrix3<f64>) -> Matrix3<f64> {
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

                vec_to_so3(&(std::f64::consts::PI * omega))
            }
            a => {
                let theta = f64::acos(a);
                theta / (2.0 * f64::sin(theta)) * (r - r.transpose())
            }
        }
    }

    pub fn rp_to_trans(r: &Matrix3<f64>, p: &Vector3<f64>) -> Matrix4<f64> {
        let mut m = Matrix4::identity();

        m.fixed_slice_mut::<U3, U3>(0, 0).copy_from(r);
        m.fixed_slice_mut::<U3, U1>(0, 3).copy_from(p);
        m
    }

    pub fn trans_to_rp(t: &Matrix4<f64>) -> (Matrix3<f64>, Vector3<f64>) {
        (
            t.fixed_slice::<U3, U3>(0, 0).clone_owned(),
            t.fixed_slice::<U3, U1>(0, 3).clone_owned(),
        )
    }

    pub fn trans_inv(t: &Matrix4<f64>) -> Matrix4<f64> {
        let (r, p) = trans_to_rp(&t);
        let rt = r.transpose();

        let mut m = Matrix4::identity();
        m.fixed_slice_mut::<U3, U3>(0, 0).copy_from(&rt);
        m.fixed_slice_mut::<U3, U1>(0, 3).copy_from(&(-rt * p));
        m
    }

    pub fn vec_to_se3(v: &Vector6<f64>) -> Matrix4<f64> {
        let mut m = Matrix4::zeros();
        m.fixed_slice_mut::<U3, U3>(0, 0)
            .copy_from(&vec_to_so3(&Vector3::new(v[0], v[1], v[2])));
        m.fixed_slice_mut::<U3, U1>(0, 3)
            .copy_from(&Vector3::new(v[3], v[4], v[5]));
        m
    }

    pub fn se3_to_vec(m: &Matrix4<f64>) -> Vector6<f64> {
        Vector6::new(
            m[(2, 1)],
            m[(0, 2)],
            m[(1, 0)],
            m[(0, 3)],
            m[(1, 3)],
            m[(2, 3)],
        )
    }

    pub fn adjoint(t: &Matrix4<f64>) -> Matrix6<f64> {
        let (r, p) = trans_to_rp(&t);

        let mut m = Matrix6::zeros();
        m.fixed_slice_mut::<U3, U3>(0, 0).copy_from(&r);
        m.fixed_slice_mut::<U3, U3>(3, 0)
            .copy_from(&(vec_to_so3(&p) * r));
        m.fixed_slice_mut::<U3, U3>(3, 3).copy_from(&r);
        m
    }

    pub fn screw_to_axis(point: &Vector3<f64>, screw: &Vector3<f64>, pitch: f64) -> Vector6<f64> {
        let mut v = Vector6::zeros();
        v.fixed_slice_mut::<U3, U1>(0, 0).copy_from(&screw);
        v.fixed_slice_mut::<U3, U1>(3, 0)
            .copy_from(&(point.cross(screw) + screw * pitch));

        v
    }

    pub fn axis_ang6(v: &Vector6<f64>) -> (Vector6<f64>, f64) {
        let mut theta = Vector3::new(v[0], v[1], v[2]).norm();

        if near_zero(theta) {
            theta = Vector3::new(v[3], v[4], v[5]).norm();
        }

        (v / theta, theta)
    }

    pub fn matrix_exp6(m: &Matrix4<f64>) -> Matrix4<f64> {
        let r = m.fixed_slice::<U3, U3>(0, 0).clone_owned();
        let v = m.fixed_slice::<U3, U1>(0, 3).clone_owned();
        let omega_theta = so3_to_vec(&r);

        let mut mm = Matrix4::identity();

        if near_zero(omega_theta.norm()) {
            mm.fixed_slice_mut::<U3, U1>(0, 3)
                .copy_from(&(m.fixed_slice::<U3, U1>(0, 3).clone_owned()));
            return mm;
        }

        let (_, theta) = axis_ang3(&omega_theta);
        let omega_mat = r / theta;

        let t = (Matrix3::identity() * theta
            + (1. - f64::cos(theta)) * omega_mat
            + (theta - f64::sin(theta)) * (omega_mat * omega_mat))
            * (v / theta);

        mm.fixed_slice_mut::<U3, U3>(0, 0)
            .copy_from(&(matrix_exp3(&r)));
        mm.fixed_slice_mut::<U3, U1>(0, 3).copy_from(&t);
        mm
    }

    pub fn matrix_log6(m: &Matrix4<f64>) -> Matrix4<f64> {
        let (r, _) = trans_to_rp(m);
        let omega_mat = matrix_log3(&r);

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
                + (1. / theta - 1. / f64::tan(theta / 2.) / 2.)
                    * ((omega_mat * omega_mat) / theta))
                * m.fixed_slice::<U3, U1>(0, 3).clone_owned()),
        );

        mm
    }

    pub fn project_to_so3(m: &Matrix3<f64>) -> Matrix3<f64> {
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

    pub fn project_to_se3(m: &Matrix4<f64>) -> Matrix4<f64> {
        rp_to_trans(
            &project_to_so3(&m.fixed_slice::<U3, U3>(0, 0).clone_owned()),
            &m.fixed_slice::<U3, U1>(0, 3).clone_owned(),
        )
    }

    pub fn distance_to_so3(m: &Matrix3<f64>) -> f64 {
        if m.determinant() <= 0. {
            return 1e+9;
        }

        (m.transpose() * m - Matrix3::identity()).norm()
    }

    pub fn distance_to_se3(m: &Matrix4<f64>) -> f64 {
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

    pub fn is_so3(m: &Matrix3<f64>) -> bool {
        f64::abs(distance_to_so3(m)) < 1e-3
    }

    pub fn is_se3(m: &Matrix4<f64>) -> bool {
        f64::abs(distance_to_se3(m)) < 1e-3
    }

    pub fn fk_in_body(
        m: &Matrix4<f64>,
        b_list: &Vec<Vector6<f64>>,
        theta_list: &Vec<f64>,
    ) -> Matrix4<f64> {
        let mut t = m.clone();

        for i in 0..theta_list.len() {
            let transformation =
                matrix_exp6(&(vec_to_se3(&(b_list[i].clone_owned() * theta_list[i]))));
            t = t * transformation;
        }

        t
    }

    pub fn fk_in_space(
        m: &Matrix4<f64>,
        s_list: &Vec<Vector6<f64>>,
        theta_list: &Vec<f64>,
    ) -> Matrix4<f64> {
        let mut t = m.clone();

        for i in (0..theta_list.len()).rev() {
            let transformation =
                matrix_exp6(&(vec_to_se3(&(s_list[i].clone_owned() * theta_list[i]))));

            t = transformation * t;
        }

        t
    }

    pub fn jacobian_body(
        b_list: &Vec<Vector6<f64>>,
        theta_list: &Vec<f64>,
    ) -> MatrixMN<f64, U6, Dynamic> {
        let mut jb = MatrixMN::<f64, U6, Dynamic>::from_columns(b_list);
        let mut t = Matrix4::identity();

        for i in (0..theta_list.len() - 1).rev() {
            t = t * matrix_exp6(&vec_to_se3(
                &(b_list[i + 1].clone_owned() * -theta_list[i + 1]),
            ));
            jb.set_column(i, &(adjoint(&t) * b_list[i].clone_owned()));
        }

        jb
    }

    pub fn jacobian_space(
        s_list: &Vec<Vector6<f64>>,
        theta_list: &Vec<f64>,
    ) -> MatrixMN<f64, U6, Dynamic> {
        let mut js = MatrixMN::<f64, U6, Dynamic>::from_columns(s_list);
        let mut t = Matrix4::identity();

        for i in 1..theta_list.len() {
            t = t * matrix_exp6(&vec_to_se3(
                &(s_list[i - 1].clone_owned() * theta_list[i - 1]),
            ));
            js.set_column(i, &(adjoint(&t) * s_list[i].clone_owned()));
        }

        js
    }

    pub fn ik_in_body(
        m: &Matrix4<f64>,
        d: &Matrix4<f64>,
        b_list: &Vec<Vector6<f64>>,
        theta_list: &Vec<f64>,
        tolerance: (f64, f64),
    ) -> (MatrixMN<f64, U1, Dynamic>, bool) {
        let (w_tolerance, v_tolerance) = tolerance;
        let max_iterations = 20;

        let mut i = 0;
        let mut joint_configuration = theta_list.clone();

        let mut t = fk_in_body(m, &b_list, &joint_configuration);
        let mut vb = se3_to_vec(&matrix_log6(&(trans_inv(&t) * d)));

        let mut e = Vector3::new(vb[0], vb[1], vb[2]).norm() > w_tolerance
            || Vector3::new(vb[3], vb[4], vb[5]).norm() > v_tolerance;

        while e && i < max_iterations {
            let pseudo_inverse = jacobian_body(&b_list, &joint_configuration)
                .pseudo_inverse(1e-15)
                .expect("Could not calculate the pseudo inverse.");
            joint_configuration = columns_to_vec(
                &(vec_to_columns(&joint_configuration) + (pseudo_inverse * vb).transpose()),
            );
            i += 1;
            t = fk_in_body(m, &b_list, &joint_configuration);
            vb = se3_to_vec(&matrix_log6(&(trans_inv(&t) * d)));
            e = Vector3::new(vb[0], vb[1], vb[2]).norm() > w_tolerance
                || Vector3::new(vb[3], vb[4], vb[5]).norm() > v_tolerance;
        }
        (vec_to_columns(&joint_configuration), !e)
    }

    pub fn ik_in_space(
        m: &Matrix4<f64>,
        d: &Matrix4<f64>,
        s_list: &Vec<Vector6<f64>>,
        theta_list: &Vec<f64>,
        tolerance: (f64, f64),
    ) -> (MatrixMN<f64, U1, Dynamic>, bool) {
        let (w_tolerance, v_tolerance) = tolerance;
        let max_iterations = 20;

        let mut i = 0;
        let mut joint_configuration = theta_list.clone();

        let mut t = fk_in_space(m, &s_list, &joint_configuration);
        let mut vs = adjoint(&t) * se3_to_vec(&matrix_log6(&(trans_inv(&t) * d)));

        let mut e = Vector3::new(vs[0], vs[1], vs[2]).norm() > w_tolerance
            || Vector3::new(vs[3], vs[4], vs[5]).norm() > v_tolerance;

        while e && i < max_iterations {
            let pseudo_inverse = jacobian_space(&s_list, &joint_configuration)
                .pseudo_inverse(1e-15)
                .expect("Could not calculate the pseudo inverse.");
            joint_configuration = columns_to_vec(
                &(vec_to_columns(&joint_configuration) + (pseudo_inverse * vs).transpose()),
            );
            i += 1;
            t = fk_in_space(m, &s_list, &joint_configuration);
            vs = adjoint(&t) * se3_to_vec(&matrix_log6(&(trans_inv(&t) * d)));
            e = Vector3::new(vs[0], vs[1], vs[2]).norm() > w_tolerance
                || Vector3::new(vs[3], vs[4], vs[5]).norm() > v_tolerance;
        }
        (vec_to_columns(&joint_configuration), !e)
    }

    pub fn ad(v: &Vector6<f64>) -> Matrix6<f64> {
        let m1 = vec_to_so3(&Vector3::new(v[0], v[1], v[2]));
        let m2 = vec_to_so3(&Vector3::new(v[3], v[4], v[5]));

        let mut m = Matrix6::zeros();

        m.fixed_slice_mut::<U3, U3>(0, 0).copy_from(&m1);
        m.fixed_slice_mut::<U3, U3>(3, 0).copy_from(&m2);
        m.fixed_slice_mut::<U3, U3>(3, 3).copy_from(&m1);
        m
    }

    pub fn cubic_time_scaling(tf: f64, t: f64) -> f64 {
        3. * f64::powi(1. * t / tf, 2) - 2. * f64::powi(1. * t / tf, 3)
    }

    pub fn quintic_time_scaling(tf: f64, t: f64) -> f64 {
        10. * f64::powi(1. * t / tf, 3) - 15. * f64::powi(1. * t / tf, 4)
            + 6. * f64::powi(1. * t / tf, 5)
    }

    pub enum TimeScalingMethod {
        cubic,
        quintic,
    }

    pub fn joint_trajectory(
        theta_start: &Vec<f64>,
        theta_end: &Vec<f64>,
        tf: f64,
        n: u32,
        method: TimeScalingMethod,
    ) -> MatrixMN<f64, Dynamic, Dynamic> {
        let time_gap = tf / (n - 1) as f64;
        let mut trajectory: MatrixMN<f64, Dynamic, Dynamic> =
            MatrixMN::<f64, Dynamic, Dynamic>::zeros(theta_start.len(), n as usize);

        let scaling = match method {
            TimeScalingMethod::cubic => cubic_time_scaling,
            TimeScalingMethod::quintic => quintic_time_scaling,
        };

        for i in 0..n {
            let s = scaling(tf, time_gap * i as f64);
            trajectory.set_column(
                i as usize,
                &((s * &vec_to_columns(theta_end) + (1. - s) * &vec_to_columns(theta_start))
                    .transpose()),
            );
        }

        trajectory.transpose()
    }

    pub fn screw_trajectory(
        x_start: &Matrix4<f64>,
        x_end: &Matrix4<f64>,
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
            let c = matrix_log6(&b) * s;
            trajectory[i] = x_start * matrix_exp6(&c);
        }

        trajectory
    }

    pub fn cartesian_trajectory(
        x_start: &Matrix4<f64>,
        x_end: &Matrix4<f64>,
        tf: f64,
        n: u32,
        method: TimeScalingMethod,
    ) -> Vec<Matrix4<f64>> {
        let time_gap = tf / (n - 1) as f64;
        let mut trajectory: Vec<Matrix4<f64>> = vec![Matrix4::identity(); n as usize];

        let (r_start, p_start) = trans_to_rp(x_start);
        let (r_end, p_end) = trans_to_rp(x_end);

        let scaling = match method {
            TimeScalingMethod::cubic => cubic_time_scaling,
            TimeScalingMethod::quintic => quintic_time_scaling,
        };

        for i in 0..n as usize {
            let s = scaling(tf, time_gap * i as f64);

            trajectory[i].fixed_slice_mut::<U3, U3>(0, 0).copy_from(
                &(r_start * matrix_exp3(&(matrix_log3(&(r_start.transpose() * r_end)) * s))),
            );
            trajectory[i]
                .fixed_slice_mut::<U3, U1>(0, 3)
                .copy_from(&(s * p_end + (1. - s) * p_start));
        }

        trajectory
    }
}
