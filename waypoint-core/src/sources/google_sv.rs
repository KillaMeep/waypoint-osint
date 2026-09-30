//! Google Street View retrieval (port of `google_sv_refine.py`): the
//! undocumented GeoPhotoService.SingleImageSearch endpoint for panorama ids
//! and the streetviewpixels tile server for pixels. No key; the tile server
//! only wants an Origin/Referer that looks like Google Maps.

use std::sync::OnceLock;

use image::{imageops, RgbImage};
use regex::Regex;
use serde_json::Value;

use crate::geo::haversine_m;
use crate::net;
use crate::retrieval::{cluster_radius_km, score_and_verify, ClusterInfo, Ctx, Progress, Scorable};
use crate::util::{par_for_each_completed, Result, Rng};

const TILE_HEADERS: [(&str, &str); 3] = [
    ("origin", "https://www.google.com"),
    ("referer", "https://www.google.com/"),
    ("user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"),
];
const IMGX: u32 = 4;
const IMGY: u32 = 2; // panorama tile grid at zoom=2
const TILE_SIZE: u32 = 512;
const MAX_WORKERS: usize = 20;
const ROAD_SAMPLE_SPACING_M: f64 = 60.0;

pub struct Pano {
    pub panoid: String,
    pub lat: f64,
    pub lon: f64,
}

fn panoids_url(lat: f64, lon: f64) -> String {
    format!(
        "https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch\
         ?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{lat}!4d{lon}!2d50!3m10!2m2!1sen!2sGB\
         !9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2&callback=_xdc_._v2mub5"
    )
}

fn tile_url(panoid: &str, x: u32, y: u32) -> String {
    format!("https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile&panoid={panoid}&x={x}&y={y}&zoom=2&nbt=1&fover=2")
}

/// Points sampled every `spacing_m` along nearby OSM roads (panoramas only
/// exist along roads). Shrinks the radius when the Overpass query is too heavy.
pub fn road_sample_points(ctx: &Ctx, lat: f64, lon: f64, radius_km: f64, spacing_m: f64, max_points: usize) -> Result<Vec<(f64, f64)>> {
    let mut data: Option<Value> = None;
    let mut tried = radius_km;
    for _ in 0..3 {
        let radius_m = (tried * 1000.0) as i64;
        let ql = format!("\n        [out:json][timeout:25];\n        way(around:{radius_m},{lat},{lon})[highway];\n        out geom;\n        ");
        data = net::overpass_query(&ql, 30, 3, ctx.cancel, ctx.log)?;
        if data.is_some() {
            if tried < radius_km {
                (ctx.log)(&format!("Google SV: road query succeeded at reduced radius {tried:.1}km (original {radius_km:.1}km was too heavy)"));
            }
            break;
        }
        if tried <= 1.0 {
            break;
        }
        tried = (tried / 2.0).max(1.0);
    }
    let Some(data) = data else { return Ok(vec![]) };

    let mut points = vec![];
    for el in data.get("elements").and_then(Value::as_array).into_iter().flatten() {
        let Some(geom) = el.get("geometry").and_then(Value::as_array) else { continue };
        for pair in geom.windows(2) {
            let g = |p: &Value, k: &str| p.get(k).and_then(Value::as_f64);
            let (Some(la1), Some(lo1), Some(la2), Some(lo2)) = (g(&pair[0], "lat"), g(&pair[0], "lon"), g(&pair[1], "lat"), g(&pair[1], "lon")) else { continue };
            let seg_len_m = haversine_m(la1, lo1, la2, lo2);
            let n = 1usize.max((seg_len_m / spacing_m).floor() as usize);
            for i in 0..n {
                let t = i as f64 / n as f64; // np.linspace(0, 1, n, endpoint=False)
                points.push((la1 + t * (la2 - la1), lo1 + t * (lo2 - lo1)));
            }
        }
    }
    if points.len() > max_points {
        let mut rng = Rng::from_time();
        let idx = rng.choice_no_replace(points.len(), max_points);
        points = idx.into_iter().map(|i| points[i]).collect();
    }
    Ok(points)
}

fn re_ids() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r#""([A-Za-z0-9_-]{22})""#).unwrap())
}

/// Extract (panoid, lat, lon) triples from the endpoint's JS-wrapped response.
pub fn parse_panoids(text: &str, lat: f64, lon: f64) -> Vec<Pano> {
    static LATLON: OnceLock<Regex> = OnceLock::new();
    let latlon = LATLON.get_or_init(|| Regex::new(r"\[null,null,(-?\d+\.\d+),(-?\d+\.\d+)").unwrap());
    let mut out = vec![];
    for cap in re_ids().captures_iter(text) {
        let panoid = cap[1].to_string();
        // Python: re.findall('"ID".+?\[null,null,LAT,LON', text) -> first quoted occurrence
        // of the id that has the pattern later on the same line (`.` stops at newlines).
        let needle = format!("\"{panoid}\"");
        let (mut plat, mut plon) = (lat, lon);
        'occ: for (pos, _) in text.match_indices(&needle) {
            let rest = &text[pos + needle.len()..];
            let line = rest.split('\n').next().unwrap_or("");
            let mut chars = line.char_indices();
            if chars.next().is_none() {
                continue; // `.+?` needs at least one character
            }
            let skip = chars.next().map_or(line.len(), |(i, _)| i);
            if let Some(m) = latlon.captures(&line[skip.min(line.len())..]) {
                // the pattern may also start at offset 1 exactly, which `skip` covers
                if let (Ok(a), Ok(b)) = (m[1].parse(), m[2].parse()) {
                    plat = a;
                    plon = b;
                    break 'occ;
                }
            }
        }
        out.push(Pano { panoid, lat: plat, lon: plon });
    }
    out
}

fn query_panoids(agent: &ureq::Agent, lat: f64, lon: f64) -> Vec<Pano> {
    match net::get_bytes(agent, &panoids_url(lat, lon), &[("Accept", "*/*")]) {
        Ok(b) => parse_panoids(&String::from_utf8_lossy(&b), lat, lon),
        Err(_) => vec![],
    }
}

pub fn search_panoramas(ctx: &Ctx, lat: f64, lon: f64, radius_km: f64, max_images: usize, on_progress: &mut dyn FnMut(usize, usize)) -> Result<Vec<Pano>> {
    let points = road_sample_points(ctx, lat, lon, radius_km, ROAD_SAMPLE_SPACING_M, 400)?;
    if points.is_empty() {
        (ctx.log)("Google SV: no road geometry found nearby, cannot sample panoids.");
        return Ok(vec![]);
    }
    (ctx.log)(&format!("Google SV: probing {} road-sampled points within {radius_km:.1}km ({MAX_WORKERS} concurrent)...", points.len()));
    let agent = net::agent(MAX_WORKERS, 10);
    let total = points.len();
    let mut seen = std::collections::HashSet::new();
    let mut results = vec![];
    par_for_each_completed(points, MAX_WORKERS, ctx.cancel, |(la, lo)| query_panoids(&agent, la, lo), |done, _i, panos| {
        on_progress(done, total);
        for p in panos {
            if seen.insert(p.panoid.clone()) {
                results.push(p);
            }
        }
    })?;
    (ctx.log)(&format!("Google SV: found {} distinct panoramas", results.len()));
    results.truncate(max_images);
    Ok(results)
}

fn download_tile(agent: &ureq::Agent, x: u32, y: u32, panoid: &str) -> Option<Vec<u8>> {
    let url = tile_url(panoid, x, y);
    for _ in 0..2 {
        if let Ok(b) = net::get_bytes(agent, &url, &TILE_HEADERS) {
            return Some(b);
        }
    }
    None
}

/// Download and stitch a panorama's tiles (missing tiles stay black).
pub fn download_panorama(agent: &ureq::Agent, cancel: &crate::util::Cancel, panoid: &str) -> Option<RgbImage> {
    let coords: Vec<(u32, u32)> = (0..IMGX).flat_map(|x| (0..IMGY).map(move |y| (x, y))).collect();
    let mut tiles: Vec<(u32, u32, Vec<u8>)> = vec![];
    let _ = par_for_each_completed(coords, (IMGX * IMGY) as usize, cancel, |(x, y)| (x, y, download_tile(agent, x, y, panoid)), |_n, _i, (x, y, data)| {
        if let Some(d) = data {
            tiles.push((x, y, d));
        }
    });
    if tiles.is_empty() {
        return None;
    }
    let mut pano = RgbImage::new(IMGX * TILE_SIZE, IMGY * TILE_SIZE);
    for (x, y, data) in tiles {
        if let Ok(t) = super::mapillary::decode_rgb(&data) {
            imageops::replace(&mut pano, &t, (x * TILE_SIZE) as i64, (y * TILE_SIZE) as i64);
        }
    }
    Some(pano)
}

pub fn refine_with_google_sv(ctx: &Ctx, cluster: &ClusterInfo, radius_km: Option<f64>, max_images: usize, verify: bool, verify_top_n: usize, on_progress: Progress) -> Result<Vec<Value>> {
    // Cap lower than Mapillary: roads exist everywhere, so a wide radius buys
    // little while making the Overpass query heavy enough to trip rate limits.
    let radius_km = radius_km.unwrap_or_else(|| cluster_radius_km(cluster, 3.0, 8.0));
    let panos = {
        let mut cb = |i: usize, n: usize| on_progress("search", i, n);
        search_panoramas(ctx, cluster.lat, cluster.lon, radius_km, max_images, &mut cb)?
    };
    if panos.is_empty() {
        return Ok(vec![]);
    }
    (ctx.log)(&format!("Google SV: downloading+stitching {} panoramas...", panos.len()));
    let agent = net::agent(64, 10);
    let total = panos.len();
    let mut images: Vec<Option<RgbImage>> = (0..total).map(|_| None).collect();
    let ids: Vec<String> = panos.iter().map(|p| p.panoid.clone()).collect();
    par_for_each_completed(ids, 8, ctx.cancel, |id| download_panorama(&agent, ctx.cancel, &id), |done, i, img| {
        images[i] = img;
        on_progress("download", done, total);
    })?;
    if images.iter().all(Option::is_none) {
        return Ok(vec![]);
    }
    let cands: Vec<Scorable> = panos
        .into_iter()
        .zip(images)
        .filter_map(|(p, img)| {
            img.map(|image| Scorable {
                lat: p.lat,
                lon: p.lon,
                url: ("street_view_url", format!("https://www.google.com/maps?q=&layer=c&cbll={},{}", p.lat, p.lon)),
                image,
            })
        })
        .collect();
    score_and_verify(ctx, cands, verify, verify_top_n, 5, on_progress)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_panoids_and_coordinates() {
        let text = r#"_xdc_._v2mub5 && _xdc_._v2mub5( [[0],[null,"TestPanoId_0123456789a"],[[null,null,48.8583701,2.2944813],[null,null,1.0]]] )"#;
        let p = parse_panoids(text, 1.0, 2.0);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].panoid, "TestPanoId_0123456789a");
        assert!((p[0].lat - 48.8583701).abs() < 1e-9);
        assert!((p[0].lon - 2.2944813).abs() < 1e-9);
    }
}
