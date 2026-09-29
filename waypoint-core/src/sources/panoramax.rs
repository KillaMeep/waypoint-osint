//! Panoramax retrieval (port of `panoramax_refine.py`): open STAC API, no key.

use serde_json::Value;

use super::mapillary::download_all;
use crate::net;
use crate::retrieval::{cluster_radius_km, score_and_verify, ClusterInfo, Ctx, Progress, Scorable};
use crate::util::Result;

const SEARCH_URL: &str = "https://api.panoramax.xyz/api/search";

struct Found {
    id: String,
    lat: f64,
    lon: f64,
    thumb_url: String,
}

fn bbox_for_radius(lat: f64, lon: f64, radius_km: f64) -> (f64, f64, f64, f64) {
    let lat_span = radius_km / 111.0;
    let lon_span = radius_km / (111.0 * lat.to_radians().cos().max(0.1));
    (lon - lon_span, lat - lat_span, lon + lon_span, lat + lat_span)
}

fn search_nearby_images(ctx: &Ctx, lat: f64, lon: f64, radius_km: f64, max_images: usize) -> Vec<Found> {
    let (min_lon, min_lat, max_lon, max_lat) = bbox_for_radius(lat, lon, radius_km);
    let bbox = format!("{min_lon},{min_lat},{max_lon},{max_lat}");
    let limit = max_images.min(100).to_string(); // STAC page size cap on this instance
    let agent = net::agent(1, 20);
    let data = match net::get_json(&agent, SEARCH_URL, &[("bbox", bbox.as_str()), ("limit", limit.as_str())]) {
        Ok(v) => v,
        Err(e) => {
            (ctx.log)(&format!("Panoramax search failed: {e}"));
            return vec![];
        }
    };
    let mut results = vec![];
    for feat in data.get("features").and_then(Value::as_array).into_iter().flatten() {
        let coords = feat.pointer("/geometry/coordinates").and_then(Value::as_array);
        let href = |k: &str| feat.pointer(&format!("/assets/{k}/href")).and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string);
        let thumb = href("sd").or_else(|| href("thumb"));
        let (Some(coords), Some(thumb), Some(id)) = (coords, thumb, feat.get("id").and_then(Value::as_str)) else { continue };
        let (Some(lo), Some(la)) = (coords.first().and_then(Value::as_f64), coords.get(1).and_then(Value::as_f64)) else { continue };
        results.push(Found { id: id.to_string(), lat: la, lon: lo, thumb_url: thumb });
    }
    (ctx.log)(&format!("Panoramax: found {} candidate images", results.len()));
    results.truncate(max_images);
    results
}

pub fn refine_with_panoramax(ctx: &Ctx, cluster: &ClusterInfo, radius_km: Option<f64>, max_images: usize, verify: bool, verify_top_n: usize, on_progress: Progress) -> Result<Vec<Value>> {
    let radius_km = radius_km.unwrap_or_else(|| cluster_radius_km(cluster, 3.0, 15.0));
    let found = search_nearby_images(ctx, cluster.lat, cluster.lon, radius_km, max_images);
    ctx.cancel.check()?;
    if found.is_empty() {
        return Ok(vec![]);
    }
    (ctx.log)(&format!("Panoramax: downloading {} images (20 concurrent)...", found.len()));
    let images = {
        let mut cb = |i: usize, n: usize| on_progress("download", i, n);
        download_all(ctx, found.iter().map(|f| f.thumb_url.clone()).collect(), &mut cb)?
    };
    let cands: Vec<Scorable> = found
        .into_iter()
        .zip(images)
        .filter_map(|(f, img)| {
            img.map(|image| Scorable {
                lat: f.lat,
                lon: f.lon,
                url: ("panoramax_url", format!("https://api.panoramax.xyz/#focus=pic&pic={}", f.id)),
                image,
            })
        })
        .collect();
    score_and_verify(ctx, cands, verify, verify_top_n, 5, on_progress)
}
