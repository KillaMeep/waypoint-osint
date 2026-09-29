//! Geographic helpers: haversine distance and the DBSCAN clustering used to
//! group PLONK samples (semantics follow scikit-learn's `DBSCAN(metric='haversine')`).

use crate::util::py_round;

pub const EARTH_RADIUS_KM: f64 = 6371.0;
pub const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// Great-circle distance in metres (same formula as `google_sv_refine._haversine_m`).
pub fn haversine_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (lat1, lon1, lat2, lon2) = (lat1.to_radians(), lon1.to_radians(), lat2.to_radians(), lon2.to_radians());
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    EARTH_RADIUS_M * 2.0 * a.sqrt().asin()
}

/// Initial bearing in degrees [0, 360) (same as `sun_refine._bearing`).
pub fn bearing_deg(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (lat1, lon1, lat2, lon2) = (lat1.to_radians(), lon1.to_radians(), lat2.to_radians(), lon2.to_radians());
    let dlon = lon2 - lon1;
    let x = dlon.sin() * lat2.cos();
    let y = lat1.cos() * lat2.sin() - lat1.sin() * lat2.cos() * dlon.cos();
    (x.atan2(y).to_degrees() + 360.0).rem_euclid(360.0)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Cluster {
    pub lat: f64,
    pub lon: f64,
    pub lat_std: f64,
    pub lon_std: f64,
    pub count: usize,
    pub weight: f64,
}

fn to_unit(lat_deg: f32, lon_deg: f32) -> [f64; 3] {
    // np.radians on float32 input stays in float32; mirror that so points
    // sitting right on the eps boundary fall the same way.
    let k = std::f32::consts::PI / 180.0;
    let (la, lo) = ((lat_deg * k) as f64, (lon_deg * k) as f64);
    [la.cos() * lo.cos(), la.cos() * lo.sin(), la.sin()]
}

/// DBSCAN over lat/lon degrees with haversine distance.
/// Returns a label per point (-1 = noise). `eps_rad` is the neighbourhood
/// radius in radians on the unit sphere; a point is a neighbour when its
/// distance is `<= eps` and counts itself (`min_samples` includes the point).
pub fn dbscan_haversine(coords: &[[f32; 2]], eps_rad: f64, min_samples: usize) -> Vec<i32> {
    let n = coords.len();
    let pts: Vec<[f64; 3]> = coords.iter().map(|c| to_unit(c[0], c[1])).collect();
    // chord length for angular distance eps: 2 sin(eps/2); compare squared chords.
    let chord2 = (2.0 * (eps_rad / 2.0).sin()).powi(2);
    let within = |a: &[f64; 3], b: &[f64; 3]| {
        let d0 = a[0] - b[0];
        let d1 = a[1] - b[1];
        let d2 = a[2] - b[2];
        d0 * d0 + d1 * d1 + d2 * d2 <= chord2
    };

    // Pass 1: core points, with early exit once min_samples neighbours are seen.
    let is_core: Vec<bool> = (0..n)
        .map(|i| {
            let mut c = 0usize;
            for j in 0..n {
                if within(&pts[i], &pts[j]) {
                    c += 1;
                    if c >= min_samples {
                        return true;
                    }
                }
            }
            false
        })
        .collect();

    // Pass 2: expansion. A point is labelled the moment a core point reaches it
    // (equivalent to sklearn's stack order: a border point joins the first
    // cluster that reaches it), and only still-unlabelled points are scanned.
    let mut labels = vec![-1i32; n];
    let mut unlabeled: Vec<usize> = (0..n).collect();
    let mut pos: Vec<usize> = (0..n).collect(); // position of each point in `unlabeled`
    let remove = |unlabeled: &mut Vec<usize>, pos: &mut Vec<usize>, v: usize| {
        let p = pos[v];
        let last = *unlabeled.last().unwrap();
        unlabeled.swap_remove(p);
        pos[last] = p;
    };
    let mut label_num = 0i32;
    let mut stack: Vec<usize> = Vec::new();
    for i in 0..n {
        if labels[i] != -1 || !is_core[i] {
            continue;
        }
        labels[i] = label_num;
        remove(&mut unlabeled, &mut pos, i);
        stack.push(i);
        while let Some(c) = stack.pop() {
            let mut k = 0;
            while k < unlabeled.len() {
                let v = unlabeled[k];
                if within(&pts[c], &pts[v]) {
                    labels[v] = label_num;
                    // swap_remove moved another point into slot k; do not advance.
                    let last = *unlabeled.last().unwrap();
                    unlabeled.swap_remove(k);
                    if last != v {
                        pos[last] = k;
                    }
                    if is_core[v] {
                        stack.push(v);
                    }
                } else {
                    k += 1;
                }
            }
        }
        label_num += 1;
    }
    labels
}

/// Group sampled lat/lon points into geographic clusters (port of
/// `plonk_core.cluster_samples`). Returns (top_k clusters, noise_frac).
pub fn cluster_samples(
    coords: &[[f32; 2]],
    cluster_radius_km: f64,
    min_cluster_frac: f64,
    top_k: usize,
) -> (Vec<Cluster>, f64) {
    let n = coords.len();
    if n == 0 {
        return (vec![], 0.0);
    }
    let min_samples = 3usize.max((n as f64 * min_cluster_frac) as usize);
    let eps_rad = cluster_radius_km / EARTH_RADIUS_KM;
    let labels = dbscan_haversine(coords, eps_rad, min_samples);

    let n_clusters = labels.iter().copied().max().map_or(0, |m| (m + 1).max(0)) as usize;
    let mut clusters = Vec::new();
    for label in 0..n_clusters as i32 {
        let members: Vec<&[f32; 2]> = coords.iter().zip(&labels).filter(|(_, l)| **l == label).map(|(c, _)| c).collect();
        let m = members.len() as f64;
        let mean = |k: usize| members.iter().map(|c| c[k] as f64).sum::<f64>() / m;
        let std = |k: usize, mu: f64| (members.iter().map(|c| (c[k] as f64 - mu).powi(2)).sum::<f64>() / m).sqrt();
        let (lat, lon) = (mean(0), mean(1));
        clusters.push(Cluster {
            lat,
            lon,
            lat_std: std(0, lat),
            lon_std: std(1, lon),
            count: members.len(),
            weight: py_round(m / n as f64, 3),
        });
    }
    clusters.sort_by(|a, b| b.count.cmp(&a.count)); // stable, like list.sort
    let noise = labels.iter().filter(|l| **l == -1).count() as f64;
    clusters.truncate(top_k);
    (clusters, py_round(noise / n as f64, 3))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_blobs_and_noise() {
        let mut pts = vec![];
        for i in 0..50 {
            pts.push([10.0 + (i % 5) as f32 * 0.01, 20.0 + (i / 5) as f32 * 0.01]);
        }
        for i in 0..30 {
            pts.push([-30.0 + (i % 5) as f32 * 0.01, 100.0 + (i / 5) as f32 * 0.01]);
        }
        pts.push([60.0, -60.0]);
        let (cl, noise) = cluster_samples(&pts, 100.0, 0.03, 5);
        assert_eq!(cl.len(), 2);
        assert_eq!(cl[0].count, 50);
        assert_eq!(cl[1].count, 30);
        assert!((noise - 0.012).abs() < 1e-9);
    }

    #[test]
    fn bearing_north_east() {
        assert!((bearing_deg(0.0, 0.0, 1.0, 0.0) - 0.0).abs() < 1e-9);
        assert!((bearing_deg(0.0, 0.0, 0.0, 1.0) - 90.0).abs() < 1e-9);
    }
}
