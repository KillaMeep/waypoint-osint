//! Sun-position refinement (port of `sun_refine.py`): season / golden-hour /
//! sun-bearing estimates from image colour statistics, then a plausibility
//! check of each candidate cluster against nearby OSM road bearings.

use std::collections::BTreeSet;

use image::RgbImage;
use serde_json::{json, Map, Value};

use crate::astral;
use crate::geo::bearing_deg;
use crate::net;
use crate::util::{py_round, Cancel, Result};

/// (month numbers) per label, northern hemisphere.
fn season_months(label: &str) -> Vec<u32> {
    match label {
        "green" => (4..10).collect(),
        "brown" => vec![9, 10, 11, 12, 1],
        "snow" => vec![11, 12, 1, 2, 3],
        _ => (1..13).collect(),
    }
}

fn days_in_month(y: i32, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 {
                29
            } else {
                28
            }
        }
    }
}

/// `_sample_dates`: every 6th day of each listed month of 2024, months in the given order.
pub fn sample_dates(months: &[u32]) -> Vec<astral::Day> {
    let mut out = vec![];
    for &m in months {
        let mut d = 1;
        while d <= days_in_month(2024, m) {
            out.push(astral::days_from_civil(2024, m, d));
            d += 6;
        }
    }
    out
}

fn iso_date(day: astral::Day) -> String {
    let (y, m, d) = astral::civil_from_days(day);
    format!("{y:04}-{m:02}-{d:02}")
}

/// matplotlib.colors.rgb_to_hsv on one float32 pixel, in float32 like numpy.
fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let s = if max > 0.0 { delta / max } else { 0.0 };
    let mut h = 0.0f32;
    if delta > 0.0 {
        if r == max {
            h = (g - b) / delta;
        }
        if g == max {
            h = 2.0 + (b - r) / delta;
        }
        if b == max {
            h = 4.0 + (r - g) / delta;
        }
    }
    let h = (h / 6.0).rem_euclid(1.0); // numpy % is floor-mod
    (h, s, max)
}

pub struct Season {
    pub label: &'static str,
    pub dates: Vec<astral::Day>,
    pub confidence: f64,
}

/// Classify ground/vegetation colour in the lower frame into green/brown/snow.
pub fn estimate_season(img: &RgbImage, lat_hint: f64) -> Season {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let start = (h as f64 * 0.55) as usize;
    let raw = img.as_raw();
    let (mut n_veg, mut sum_hue, mut sum_sat, mut sum_val) = (0usize, 0.0f64, 0.0f64, 0.0f64);
    let mut n_white = 0usize;
    let total = (h - start) * w;
    for y in start..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let (r, g, b) = (raw[i] as f32 / 255.0, raw[i + 1] as f32 / 255.0, raw[i + 2] as f32 / 255.0);
            let (hh, s, v) = rgb_to_hsv(r, g, b);
            let hue = hh * 360.0;
            if s > 0.15f32 {
                n_veg += 1;
                sum_hue += hue as f64;
                sum_sat += s as f64;
                sum_val += v as f64;
            }
            if s < 0.1f32 && v > 0.75f32 {
                n_white += 1;
            }
        }
    }
    let shift = |months: Vec<u32>| -> Vec<u32> {
        if lat_hint < 0.0 {
            months.into_iter().map(|m| ((m - 1 + 6) % 12) + 1).collect()
        } else {
            months
        }
    };
    if n_veg < 50 {
        let months = shift(season_months("unknown"));
        return Season { label: "unknown", dates: sample_dates(&months), confidence: 0.0 };
    }
    let nv = n_veg as f64;
    let (mean_hue, mean_sat, mean_val) = (sum_hue / nv, sum_sat / nv, sum_val / nv);
    let white_frac = n_white as f64 / total as f64;

    let (label, confidence) = if white_frac > 0.4 {
        ("snow", white_frac.min(0.8))
    } else if (70.0..=170.0).contains(&mean_hue) && mean_sat > 0.2 {
        ("green", mean_sat.min(0.8))
    } else if (20.0..70.0).contains(&mean_hue) {
        ("brown", (mean_sat + (1.0 - mean_val) * 0.3).min(0.8))
    } else {
        ("unknown", 0.0)
    };
    let months = shift(season_months(label));
    Season { label, dates: sample_dates(&months), confidence: py_round(confidence, 2) }
}

/// Is this plausibly a sunrise/sunset shot? Returns (is_golden, confidence).
pub fn estimate_golden_hour(img: &RgbImage) -> (bool, f64) {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let rows = (h as f64 * 0.6) as usize;
    let raw = img.as_raw();
    let (mut warm, mut blue) = (0usize, 0usize);
    for y in 0..rows {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let (r, b) = (raw[i] as f32 / 255.0, raw[i + 2] as f32 / 255.0);
            if r - b > 0.08f32 {
                warm += 1;
            }
            if b - r > 0.08f32 {
                blue += 1;
            }
        }
    }
    let n = (rows * w) as f64;
    let (warm_frac, blue_frac) = (warm as f64 / n, blue as f64 / n);
    let golden = warm_frac > 0.25 && warm_frac > blue_frac;
    let conf = if golden { py_round(warm_frac.min(0.9), 2) } else { py_round(blue_frac.min(0.9), 2) };
    (golden, conf)
}

/// numpy.percentile(x, q) with the default linear interpolation.
fn percentile(scores: &mut [f32], q: f64) -> f64 {
    let n = scores.len();
    let virtual_idx = (n - 1) as f64 * (q / 100.0);
    let prev = virtual_idx.floor() as usize;
    let next = (prev + 1).min(n - 1);
    let gamma = virtual_idx - prev as f64;
    scores.select_nth_unstable_by(prev, |a, b| a.total_cmp(b));
    let a = scores[prev];
    let b = if next == prev {
        a
    } else {
        // smallest element of the right partition
        scores[next..].iter().copied().fold(f32::INFINITY, f32::min)
    };
    // numpy: a + (b - a) * gamma with the difference taken in float32
    a as f64 + ((b - a) as f64) * gamma
}

/// Horizontal angular offset of the brightest+warmest sky region from the
/// camera axis. Returns None when nothing qualifies.
pub fn estimate_sun_bearing_offset(img: &RgbImage, fov_deg: f64) -> Option<(f64, f64)> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let rows = (h as f64 * 0.6) as usize;
    if rows == 0 {
        return None;
    }
    let raw = img.as_raw();
    let mut scores: Vec<f32> = Vec::with_capacity(rows * w);
    for y in 0..rows {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let (r, g, b) = (raw[i] as f32, raw[i + 1] as f32, raw[i + 2] as f32);
            let brightness = (r + g + b) / 3.0;
            let warmth = (r - b).clamp(0.0, 255.0);
            scores.push(brightness * 0.5 + warmth * 0.5);
        }
    }
    let mut tmp = scores.clone();
    let threshold = percentile(&mut tmp, 99.0);
    drop(tmp);
    let (mut count, mut sum_x) = (0u64, 0u64);
    for y in 0..rows {
        for x in 0..w {
            if scores[y * w + x] as f64 >= threshold {
                count += 1;
                sum_x += x as u64;
            }
        }
    }
    if count == 0 {
        return None;
    }
    let centroid_x = sum_x as f64 / count as f64;
    let offset = ((centroid_x / w as f64) - 0.5) * fov_deg;
    let concentration = count as f64 / (rows * w) as f64;
    let confidence = (1.0 - concentration * 20.0).clamp(0.1, 1.0);
    Some((offset, confidence))
}

fn signed_diff(a: f64, b: f64) -> f64 {
    (a - b + 180.0).rem_euclid(360.0) - 180.0
}

pub struct SunMatch {
    pub error: f64,
    pub date: String,
    pub event: &'static str,
    pub sun_azimuth: f64,
    pub camera_heading: i64,
}

/// Sweep sunrise/sunset azimuths over `dates` and road-facing hypotheses and
/// return the best (lowest-error) explanation of the observed sun bearing.
pub fn best_match_for_candidate(lat: f64, lon: f64, dates: &[astral::Day], road_bearings: &[f64], observed_offset_deg: f64) -> Option<SunMatch> {
    if road_bearings.is_empty() {
        return None;
    }
    let mut headings = BTreeSet::new();
    for b in road_bearings {
        let r = b.round_ties_even() as i64; // Python round() is half-even
        headings.insert(r);
        headings.insert((r + 180).rem_euclid(360));
    }
    let mut best: Option<SunMatch> = None;
    for &d in dates {
        let Some((rise, set)) = astral::sun_rise_set(lat, lon, d) else { continue };
        for (event, t) in [("sunrise", rise), ("sunset", set)] {
            let az = astral::azimuth(lat, lon, t);
            for &heading in &headings {
                let expected = signed_diff(az, heading as f64);
                let error = signed_diff(expected, observed_offset_deg).abs();
                // Python compares the raw error against the already-rounded best error.
                if best.as_ref().map_or(true, |bm| error < bm.error) {
                    best = Some(SunMatch {
                        error: py_round(error, 1),
                        date: iso_date(d),
                        event,
                        sun_azimuth: py_round(az, 1),
                        camera_heading: heading,
                    });
                }
            }
        }
    }
    best
}

/// Bearings (mod 180) of every OSM highway segment within `radius_m`.
pub fn nearby_road_bearings(lat: f64, lon: f64, radius_m: u32, cancel: &Cancel, log: &dyn Fn(&str)) -> Result<Vec<f64>> {
    let ql = format!("\n    [out:json][timeout:15];\n    way(around:{radius_m},{lat},{lon})[highway];\n    out geom;\n    ");
    let Some(data) = net::overpass_query(&ql, 30, 3, cancel, log)? else { return Ok(vec![]) };
    let mut bearings = vec![];
    for el in data.get("elements").and_then(Value::as_array).into_iter().flatten() {
        let Some(geom) = el.get("geometry").and_then(Value::as_array) else { continue };
        for pair in geom.windows(2) {
            let g = |p: &Value, k: &str| p.get(k).and_then(Value::as_f64);
            if let (Some(la1), Some(lo1), Some(la2), Some(lo2)) = (g(&pair[0], "lat"), g(&pair[0], "lon"), g(&pair[1], "lat"), g(&pair[1], "lon")) {
                bearings.push(bearing_deg(la1, lo1, la2, lo2).rem_euclid(180.0));
            }
        }
    }
    Ok(bearings)
}

/// `refine_clusters`: annotates each cluster (a JSON object) with
/// `sun_evidence` and returns the metadata object, or None when the image
/// doesn't look like a sunrise/sunset shot.
pub fn refine_clusters(clusters: &mut [Value], img: &RgbImage, fov_deg: f64, cancel: &Cancel, log: &dyn Fn(&str)) -> Result<Option<Value>> {
    let lat_hint = clusters.first().and_then(|c| c["lat"].as_f64()).unwrap_or(40.0);
    let season = estimate_season(img, lat_hint);
    let (is_golden, golden_conf) = estimate_golden_hour(img);
    let offset = estimate_sun_bearing_offset(img, fov_deg);

    log(&format!("Season estimate: {} (confidence {}) -> {} sample dates", season.label, season.confidence, season.dates.len()));
    log(&format!("Golden-hour estimate: {is_golden} (confidence {golden_conf})"));
    let Some((offset_deg, offset_conf)) = offset else {
        log("No bright sky region detected, skipping sun-based refinement.");
        return Ok(None);
    };
    if !is_golden {
        log("Image does not look like a sunrise/sunset shot, sun-bearing refinement is unreliable here, skipping.");
        return Ok(None);
    }
    log(&format!("Estimated sun/glow bearing offset from center: {offset_deg:+.1}\u{b0} (confidence {offset_conf:.2}, assumed FOV {fov_deg}\u{b0})"));

    // Deliberately no re-sort by sun error: see sun_refine.py. A high error
    // is evidence against a candidate, a low error is weak confirmation.
    for cluster in clusters.iter_mut() {
        let (lat, lon) = (cluster["lat"].as_f64().unwrap_or(0.0), cluster["lon"].as_f64().unwrap_or(0.0));
        let bearings = nearby_road_bearings(lat, lon, 300, cancel, log)?;
        let n_distinct = bearings.iter().map(|b| b.round_ties_even() as i64).collect::<BTreeSet<_>>().len();
        let m = best_match_for_candidate(lat, lon, &season.dates, &bearings, offset_deg);
        let evidence = match m {
            Some(m) => {
                let reliability = if n_distinct > 4 {
                    "low (many road orientations nearby, fit is easy)"
                } else {
                    "moderate (few road orientations, fit is more constrained)"
                };
                json!({
                    "error": m.error, "date": m.date, "event": m.event, "sun_azimuth": m.sun_azimuth,
                    "camera_heading": m.camera_heading, "n_road_bearings": n_distinct, "reliability": reliability,
                })
            }
            None => json!({ "error": null, "note": "no usable road data nearby" }),
        };
        if let Some(o) = cluster.as_object_mut() {
            o.insert("sun_evidence".into(), evidence);
        }
    }

    let mut meta = Map::new();
    meta.insert("season".into(), json!(season.label));
    meta.insert("season_confidence".into(), json!(season.confidence));
    meta.insert("golden_hour".into(), json!(is_golden));
    meta.insert("golden_hour_confidence".into(), json!(golden_conf));
    meta.insert("observed_offset_deg".into(), json!(py_round(offset_deg, 1)));
    meta.insert("offset_confidence".into(), json!(py_round(offset_conf, 2)));
    Ok(Some(Value::Object(meta)))
}
