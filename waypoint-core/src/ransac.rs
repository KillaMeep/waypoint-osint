//! Fundamental-matrix RANSAC inlier counting, equivalent to
//! `cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)` as used by
//! `verify_utils.match_pair`: 7-point minimal solver, models scored by
//! max(sq. distance to epipolar line in either image) <= threshold^2,
//! adaptive iteration count, best-consensus model wins.

use crate::util::Rng;

type Pt = [f64; 2];

/// Symmetric eigen-decomposition (cyclic Jacobi). Returns eigenvalues and
/// eigenvectors (columns of the returned row-major matrix).
fn jacobi_eigen(a: &mut [[f64; 9]; 9]) -> ([f64; 9], [[f64; 9]; 9]) {
    let mut v = [[0.0; 9]; 9];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _sweep in 0..60 {
        let mut off = 0.0;
        for i in 0..9 {
            for j in 0..i {
                off += a[i][j] * a[i][j];
            }
        }
        if off < 1e-28 {
            break;
        }
        for p in 0..8 {
            for q in p + 1..9 {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..9 {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..9 {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for k in 0..9 {
                    let (vkp, vkq) = (v[k][p], v[k][q]);
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut w = [0.0; 9];
    for i in 0..9 {
        w[i] = a[i][i];
    }
    (w, v)
}

fn det3(m: &[f64; 9]) -> f64 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6])
}

/// Real roots of c3 x^3 + c2 x^2 + c1 x + c0.
fn solve_cubic(c: [f64; 4]) -> Vec<f64> {
    let [c3, c2, c1, c0] = c;
    let scale = c.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if scale == 0.0 {
        return vec![];
    }
    if c3.abs() < 1e-12 * scale {
        // quadratic
        if c2.abs() < 1e-12 * scale {
            return if c1.abs() > 0.0 { vec![-c0 / c1] } else { vec![] };
        }
        let disc = c1 * c1 - 4.0 * c2 * c0;
        if disc < 0.0 {
            return vec![];
        }
        let sq = disc.sqrt();
        return vec![(-c1 + sq) / (2.0 * c2), (-c1 - sq) / (2.0 * c2)];
    }
    let (a, b, cc) = (c2 / c3, c1 / c3, c0 / c3);
    let q = (a * a - 3.0 * b) / 9.0;
    let r = (2.0 * a * a * a - 9.0 * a * b + 27.0 * cc) / 54.0;
    let q3 = q * q * q;
    if r * r < q3 {
        let t = (r / q3.sqrt()).clamp(-1.0, 1.0).acos();
        let m = -2.0 * q.sqrt();
        let two_pi = 2.0 * std::f64::consts::PI;
        vec![m * (t / 3.0).cos() - a / 3.0, m * ((t + two_pi) / 3.0).cos() - a / 3.0, m * ((t - two_pi) / 3.0).cos() - a / 3.0]
    } else {
        let e = -(r.abs() + (r * r - q3).sqrt()).cbrt() * r.signum();
        let f = if e != 0.0 { q / e } else { 0.0 };
        vec![(e + f) - a / 3.0]
    }
}

fn normalize(pts: &[Pt]) -> (Vec<Pt>, [f64; 9]) {
    let n = pts.len() as f64;
    let (cx, cy) = (pts.iter().map(|p| p[0]).sum::<f64>() / n, pts.iter().map(|p| p[1]).sum::<f64>() / n);
    let md = pts.iter().map(|p| ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt()).sum::<f64>() / n;
    let s = if md > 1e-12 { std::f64::consts::SQRT_2 / md } else { 1.0 };
    (pts.iter().map(|p| [(p[0] - cx) * s, (p[1] - cy) * s]).collect(), [s, 0.0, -cx * s, 0.0, s, -cy * s, 0.0, 0.0, 1.0])
}

fn matmul3(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut o = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            o[i * 3 + j] = (0..3).map(|k| a[i * 3 + k] * b[k * 3 + j]).sum();
        }
    }
    o
}

fn transpose3(a: &[f64; 9]) -> [f64; 9] {
    [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]
}

/// Seven-point algorithm: up to three candidate fundamental matrices.
pub fn seven_point(p1: &[Pt], p2: &[Pt]) -> Vec<[f64; 9]> {
    debug_assert!(p1.len() == 7 && p2.len() == 7);
    let (n1, t1) = normalize(p1);
    let (n2, t2) = normalize(p2);
    let mut ata = [[0.0; 9]; 9];
    for i in 0..7 {
        let (x1, y1, x2, y2) = (n1[i][0], n1[i][1], n2[i][0], n2[i][1]);
        let row = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1.0];
        for a in 0..9 {
            for b in 0..9 {
                ata[a][b] += row[a] * row[b];
            }
        }
    }
    let (w, v) = jacobi_eigen(&mut ata);
    let mut order: Vec<usize> = (0..9).collect();
    order.sort_by(|&a, &b| w[a].total_cmp(&w[b]));
    let col = |k: usize| -> [f64; 9] {
        let mut o = [0.0; 9];
        for r in 0..9 {
            o[r] = v[r][order[k]];
        }
        o
    };
    let (f1, f2) = (col(0), col(1));
    // det(F2 + a (F1 - F2)) as a cubic in a, via samples at a = 0, 1, -1, 2.
    let at = |a: f64| -> f64 {
        let mut m = [0.0; 9];
        for i in 0..9 {
            m[i] = f2[i] + a * (f1[i] - f2[i]);
        }
        det3(&m)
    };
    let (d0, d1, dm1, d2) = (at(0.0), at(1.0), at(-1.0), at(2.0));
    let c0 = d0;
    let c2 = (d1 + dm1) / 2.0 - d0;
    // d1 = c3 + c2 + c1 + c0 ; d2 = 8c3 + 4c2 + 2c1 + c0.
    let s1 = d1 - c0 - c2; // = c3 + c1
    let s2 = d2 - c0 - 4.0 * c2; // = 8 c3 + 2 c1
    let c3 = (s2 - 2.0 * s1) / 6.0;
    let c1 = s1 - c3;
    let mut out = vec![];
    for a in solve_cubic([c3, c2, c1, c0]) {
        let mut fnorm = [0.0; 9];
        for i in 0..9 {
            fnorm[i] = f2[i] + a * (f1[i] - f2[i]);
        }
        let f = matmul3(&matmul3(&transpose3(&t2), &fnorm), &t1);
        if f.iter().any(|x| !x.is_finite()) {
            continue;
        }
        let scale = if f[8].abs() > 1e-12 { 1.0 / f[8] } else { 1.0 / f.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300) };
        out.push(f.map(|x| x * scale));
    }
    out
}

/// Per-point error: max of squared point-to-epipolar-line distances in both images.
fn point_error(f: &[f64; 9], p1: &Pt, p2: &Pt) -> f64 {
    let (x1, y1, x2, y2) = (p1[0], p1[1], p2[0], p2[1]);
    let a = f[0] * x1 + f[1] * y1 + f[2];
    let b = f[3] * x1 + f[4] * y1 + f[5];
    let c = f[6] * x1 + f[7] * y1 + f[8];
    let s2 = 1.0 / (a * a + b * b);
    let d2 = x2 * a + y2 * b + c;
    let a = f[0] * x2 + f[3] * y2 + f[6];
    let b = f[1] * x2 + f[4] * y2 + f[7];
    let c = f[2] * x2 + f[5] * y2 + f[8];
    let s1 = 1.0 / (a * a + b * b);
    let d1 = x1 * a + y1 * b + c;
    (d1 * d1 * s1).max(d2 * d2 * s2)
}

pub fn count_inliers(f: &[f64; 9], p1: &[Pt], p2: &[Pt], threshold: f64) -> Vec<bool> {
    let t2 = threshold * threshold;
    p1.iter().zip(p2).map(|(a, b)| point_error(f, a, b) <= t2).collect()
}

fn update_num_iters(confidence: f64, outlier_ratio: f64, model_points: i32, max_iters: usize) -> usize {
    let num = (1.0 - confidence).max(f64::MIN_POSITIVE).ln();
    let denom_arg = 1.0 - (1.0 - outlier_ratio.clamp(0.0, 1.0)).powi(model_points);
    let denom = if denom_arg <= f64::MIN_POSITIVE { f64::MIN_POSITIVE } else { denom_arg }.ln();
    if denom >= 0.0 {
        return max_iters;
    }
    let n = num / denom;
    if n < 0.0 || n > max_iters as f64 {
        max_iters
    } else {
        n.round().max(1.0) as usize
    }
}

/// Returns the number of inliers of the best model (0 if none was found).
/// Deterministic for a given seed.
pub fn fundamental_ransac_inliers(pts1: &[[f32; 2]], pts2: &[[f32; 2]], threshold: f64, confidence: f64, max_iters: usize, seed: u64) -> usize {
    let n = pts1.len().min(pts2.len());
    if n < 7 {
        return 0;
    }
    let p1: Vec<Pt> = pts1[..n].iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
    let p2: Vec<Pt> = pts2[..n].iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
    if n == 7 {
        return seven_point(&p1, &p2).iter().map(|f| count_inliers(f, &p1, &p2, threshold).iter().filter(|b| **b).count()).max().unwrap_or(0);
    }
    let mut rng = Rng::new(seed);
    let mut best = 0usize;
    let mut niters = max_iters;
    let mut it = 0usize;
    while it < niters {
        it += 1;
        let idx = rng.choice_no_replace(n, 7);
        let s1: Vec<Pt> = idx.iter().map(|&i| p1[i]).collect();
        let s2: Vec<Pt> = idx.iter().map(|&i| p2[i]).collect();
        for f in seven_point(&s1, &s2) {
            let good = count_inliers(&f, &p1, &p2, threshold).iter().filter(|b| **b).count();
            if good > best && good >= 7 {
                best = good;
                niters = update_num_iters(confidence, (n - good) as f64 / n as f64, 7, niters);
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic two-view scene with a known F and check that the
    /// planted inliers are found while gross outliers are rejected.
    #[test]
    fn finds_planted_inliers() {
        let mut rng = Rng::new(12345);
        let f01 = |r: &mut Rng| (r.next_u64() % 10_000) as f64 / 10_000.0;
        // camera 2 = camera 1 translated along x and rotated slightly about y
        let (fx, cx, cy) = (800.0, 512.0, 384.0);
        let ang: f64 = 0.05;
        let (c, s) = (ang.cos(), ang.sin());
        let t = [0.5, 0.05, 0.02];
        let mut a = vec![];
        let mut b = vec![];
        for _ in 0..120 {
            let x = [(f01(&mut rng) - 0.5) * 4.0, (f01(&mut rng) - 0.5) * 3.0, 4.0 + f01(&mut rng) * 6.0];
            a.push([(fx * x[0] / x[2] + cx) as f32, (fx * x[1] / x[2] + cy) as f32]);
            let x2 = [c * x[0] + s * x[2] + t[0], x[1] + t[1], -s * x[0] + c * x[2] + t[2]];
            b.push([(fx * x2[0] / x2[2] + cx) as f32, (fx * x2[1] / x2[2] + cy) as f32]);
        }
        for i in 100..120 {
            b[i] = [(f01(&mut rng) * 1000.0) as f32, (f01(&mut rng) * 700.0) as f32];
        }
        let inl = fundamental_ransac_inliers(&a, &b, 3.0, 0.99, 1000, 7);
        assert!((98..=110).contains(&inl), "inliers = {inl}");
    }
}
