//! Deterministic parity tests against data recorded from the Python
//! implementation (tools/parity/ref_*.py). Fixtures live in tests/fixtures
//! (small committed subset) or, when present, tools/parity/ref (full set).

use std::path::PathBuf;

use serde_json::Value;
use waypoint_core::{astral, geo, ransac, sun};

/// The test photos live outside the repo: set WAYPOINT_TEST_IMAGES to the folder holding test_pano.jpg.
fn test_image(name: &str) -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var_os("WAYPOINT_TEST_IMAGES")?).join(format!("test_{name}.jpg"));
    p.is_file().then_some(p)
}

fn find(name: &str) -> Option<PathBuf> {
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    [here.join("tests/fixtures").join(name), here.join("../tools/parity/ref").join(name)].into_iter().find(|p| p.exists())
}

fn json(name: &str) -> Option<Value> {
    find(name).map(|p| serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap())
}

fn npy_f32(name: &str) -> Option<(Vec<usize>, Vec<f32>)> {
    let p = find(name)?;
    let f = npyz::NpyFile::new(std::io::BufReader::new(std::fs::File::open(p).unwrap())).unwrap();
    let shape = f.shape().iter().map(|d| *d as usize).collect();
    Some((shape, f.into_vec::<f32>().unwrap()))
}

fn parse_iso_us(s: &str) -> i64 {
    // 2024-01-01T12:34:56.123456+00:00
    let (date, rest) = s.split_once('T').unwrap();
    let mut d = date.split('-').map(|x| x.parse::<i64>().unwrap());
    let (y, m, dd) = (d.next().unwrap(), d.next().unwrap(), d.next().unwrap());
    let time = rest.split('+').next().unwrap();
    let (hms, frac) = time.split_once('.').unwrap_or((time, "0"));
    let mut t = hms.split(':').map(|x| x.parse::<i64>().unwrap());
    let (h, mi, se) = (t.next().unwrap(), t.next().unwrap(), t.next().unwrap());
    let us: i64 = format!("{frac:0<6}").parse().unwrap();
    let day = astral::days_from_civil(y as i32, m as u32, dd as u32);
    (day * 86_400 + h * 3600 + mi * 60 + se) * 1_000_000 + us
}

#[test]
fn astral_matches_python() {
    let Some(v) = json("sun_ref.json") else { eprintln!("skipped: no sun_ref.json"); return };
    let rows = v["astral"].as_array().unwrap();
    let (mut n_ok, mut n_err, mut max_dt_us, mut max_az) = (0, 0, 0i64, 0f64);
    for r in rows {
        let (lat, lon) = (r["lat"].as_f64().unwrap(), r["lon"].as_f64().unwrap());
        let d = r["date"].as_str().unwrap();
        let mut p = d.split('-').map(|x| x.parse::<i32>().unwrap());
        let day = astral::days_from_civil(p.next().unwrap(), p.next().unwrap() as u32, p.next().unwrap() as u32);
        let got = astral::sun_rise_set(lat, lon, day);
        if r.get("error").is_some() {
            assert!(got.is_none(), "python raised for {lat},{lon},{d} but rust returned {got:?}");
            n_err += 1;
            continue;
        }
        let (rise, set) = got.unwrap_or_else(|| panic!("rust failed where python succeeded: {lat},{lon},{d}"));
        let (pr, ps) = (parse_iso_us(r["sunrise"].as_str().unwrap()), parse_iso_us(r["sunset"].as_str().unwrap()));
        max_dt_us = max_dt_us.max((rise - pr).abs()).max((set - ps).abs());
        // azimuth evaluated at the PYTHON instants so time error doesn't compound
        let (a1, a2) = (astral::azimuth(lat, lon, pr), astral::azimuth(lat, lon, ps));
        max_az = max_az.max((a1 - r["az_sunrise"].as_f64().unwrap()).abs()).max((a2 - r["az_sunset"].as_f64().unwrap()).abs());
        n_ok += 1;
    }
    eprintln!("astral: {n_ok} ok rows, {n_err} polar/error rows, max time diff {max_dt_us} us, max azimuth diff {max_az:e} deg");
    assert!(max_dt_us <= 1_000, "sunrise/sunset differ by {max_dt_us} us");
    assert!(max_az < 1e-6, "azimuth differs by {max_az}");
}

#[test]
fn best_match_matches_python() {
    let Some(v) = json("sun_ref.json") else { eprintln!("skipped: no sun_ref.json"); return };
    let (mut same, mut tie_diff, mut total) = (0, 0, 0);
    for r in v["best_match"].as_array().unwrap() {
        let bearings: Vec<f64> = r["bearings"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        let months: Vec<u32> = match r["season"].as_str().unwrap() {
            "green" => (4..10).collect(),
            "brown" => vec![9, 10, 11, 12, 1],
            "snow" => vec![11, 12, 1, 2, 3],
            _ => (1..13).collect(),
        };
        let dates = sun::sample_dates(&months);
        let got = sun::best_match_for_candidate(r["lat"].as_f64().unwrap(), r["lon"].as_f64().unwrap(), &dates, &bearings, r["offset"].as_f64().unwrap());
        total += 1;
        let want = &r["result"];
        match (got, want.is_null()) {
            (None, true) => same += 1,
            (Some(g), false) => {
                // the minimal error must agree exactly; date/heading may legitimately differ on ties
                assert_eq!(g.error, want["error"].as_f64().unwrap(), "error differs: {r}");
                if g.date == want["date"].as_str().unwrap() && g.camera_heading == want["camera_heading"].as_i64().unwrap() && g.event == want["event"].as_str().unwrap() {
                    same += 1;
                } else {
                    tie_diff += 1;
                }
            }
            (g, w) => panic!("presence mismatch rust={} python_null={w}", g.is_some()),
        }
    }
    eprintln!("best_match: {total} cases, {same} identical, {tie_diff} same error but different tie-break");
    assert!(same * 10 >= total * 9, "too many tie-break differences: {tie_diff}/{total}");
}

#[test]
fn image_stats_match_python() {
    let Some(v) = json("sun_ref.json") else { eprintln!("skipped: no sun_ref.json"); return };
    for name in ["pano"] {
        let Some(path) = test_image(name) else {
            eprintln!("skipped {name}: set WAYPOINT_TEST_IMAGES");
            continue;
        };
        let img = image::open(path).unwrap().to_rgb8();
        let want = &v["images"][name];
        let s = sun::estimate_season(&img, 40.0);
        assert_eq!(s.label, want["season"].as_str().unwrap(), "{name} season");
        assert!((s.confidence - want["season_conf"].as_f64().unwrap()).abs() <= 0.011, "{name} season conf {} vs {}", s.confidence, want["season_conf"]);
        assert_eq!(s.dates.len() as u64, want["n_dates"].as_u64().unwrap());
        let (g, gc) = sun::estimate_golden_hour(&img);
        assert_eq!(g, want["golden"].as_bool().unwrap(), "{name} golden");
        assert!((gc - want["golden_conf"].as_f64().unwrap()).abs() <= 0.011, "{name} golden conf {gc} vs {}", want["golden_conf"]);
        let (off, oc) = sun::estimate_sun_bearing_offset(&img, 65.0).unwrap();
        let (woff, woc) = (want["offset"].as_f64().unwrap(), want["offset_conf"].as_f64().unwrap());
        eprintln!("{name}: offset rust {off:.4} python {woff:.4}; conf {oc:.4} vs {woc:.4}");
        assert!((off - woff).abs() < 0.1, "{name} offset {off} vs {woff}");
        assert!((oc - woc).abs() < 0.02, "{name} offset conf {oc} vs {woc}");
    }
}

#[test]
fn clustering_matches_python() {
    let Some(want) = json("plonk_clusters.json") else { eprintln!("skipped: no plonk_clusters.json"); return };
    let mut checked = 0;
    for (key, file) in [("pano_s1", "pano_samples_s1.npy"), ("pano_s2", "pano_samples_s2.npy"), ("pano_big", "pano_big_samples.npy")] {
        let Some((shape, flat)) = npy_f32(file) else { continue };
        assert_eq!(shape[1], 2);
        let pts: Vec<[f32; 2]> = flat.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
        let t = std::time::Instant::now();
        let (clusters, noise) = geo::cluster_samples(&pts, 100.0, 0.03, 5);
        eprintln!("{key}: {} samples clustered in {:?}", pts.len(), t.elapsed());
        let w = &want[key];
        let wc = w["clusters"].as_array().unwrap();
        assert_eq!(clusters.len(), wc.len(), "{key} cluster count");
        assert!((noise - w["noise_frac"].as_f64().unwrap()).abs() < 1e-9, "{key} noise {noise}");
        for (c, p) in clusters.iter().zip(wc) {
            assert_eq!(c.count as u64, p["count"].as_u64().unwrap(), "{key} count");
            assert_eq!(c.weight, p["weight"].as_f64().unwrap(), "{key} weight");
            for (a, b, n) in [(c.lat, "lat", 1e-4), (c.lon, "lon", 1e-4), (c.lat_std, "lat_std", 1e-4), (c.lon_std, "lon_std", 1e-4)] {
                let pv = p[b].as_f64().unwrap();
                assert!((a - pv).abs() < n, "{key} {b}: rust {a} python {pv}");
            }
        }
        checked += 1;
    }
    assert!(checked > 0 || find("pano_samples_s1.npy").is_none());
}

#[test]
fn ransac_matches_opencv_masks() {
    let mut checked = 0;
    for name in ["match_pano__self_crop.npz"] {
        let Some(p) = find(name) else { continue };
        let mut z = npyz::npz::NpzArchive::open(&p).unwrap();
        let mut get = |k: &str| -> (Vec<u64>, Vec<f32>) {
            let f = z.by_name(k).unwrap().unwrap();
            let shape = f.shape().to_vec();
            (shape, f.into_vec::<f32>().unwrap())
        };
        let (_, kp1) = get("kp1");
        let (_, kp2) = get("kp2");
        let idxs: Vec<i64> = z.by_name("idxs").unwrap().unwrap().into_vec::<i64>().unwrap();
        let mask: Vec<u8> = z.by_name("mask").unwrap().unwrap().into_vec::<u8>().unwrap();
        let n = idxs.len() / 2;
        let pts1: Vec<[f32; 2]> = (0..n).map(|i| { let k = idxs[2 * i] as usize; [kp1[2 * k], kp1[2 * k + 1]] }).collect();
        let pts2: Vec<[f32; 2]> = (0..n).map(|i| { let k = idxs[2 * i + 1] as usize; [kp2[2 * k], kp2[2 * k + 1]] }).collect();
        let want: usize = mask.iter().map(|m| *m as usize).sum();
        let got = ransac::fundamental_ransac_inliers(&pts1, &pts2, 3.0, 0.99, 1000, 0x5EED);
        eprintln!("{name}: {n} matches, opencv inliers {want}, rust inliers {got}");
        let tol = ((want as f64) * 0.05).ceil() as usize + 1;
        assert!(got.abs_diff(want) <= tol, "{name}: rust {got} vs opencv {want}");
        checked += 1;
    }
    let _ = checked;
}
