//! Mapillary retrieval (port of `mapillary_refine.py`). Needs the user's API token.

use std::collections::HashSet;

use image::RgbImage;
use serde_json::Value;

use crate::net;
use crate::retrieval::{cluster_radius_km, score_and_verify, ClusterInfo, Ctx, Progress, Scorable};
use crate::util::{par_for_each_completed, Result};

const IMAGES_URL: &str = "https://graph.mapillary.com/images";
const BBOX_DEG: f64 = 0.008; // Mapillary rejects bbox queries wider than 0.01 degrees
const MAX_WORKERS: usize = 20;
const SEARCH_WORKERS: usize = 256;

pub struct Found {
    pub id: String,
    pub lat: f64,
    pub lon: f64,
    pub thumb_url: String,
}

/// Cover a circular search area with bbox tiles: (min_lat, min_lon, max_lat, max_lon).
pub fn bbox_tiles(lat: f64, lon: f64, radius_km: f64) -> Vec<(f64, f64, f64, f64)> {
    let lat_span = radius_km / 111.0;
    let lon_span = radius_km / (111.0 * lat.to_radians().cos().max(0.1));
    let n_lat = 1usize.max(((2.0 * lat_span) / BBOX_DEG).ceil() as usize);
    let n_lon = 1usize.max(((2.0 * lon_span) / BBOX_DEG).ceil() as usize);
    let mut tiles = Vec::with_capacity(n_lat * n_lon);
    for i in 0..n_lat {
        for j in 0..n_lon {
            let tile_lat = lat - lat_span + i as f64 * BBOX_DEG;
            let tile_lon = lon - lon_span + j as f64 * BBOX_DEG;
            tiles.push((tile_lat, tile_lon, tile_lat + BBOX_DEG, tile_lon + BBOX_DEG));
        }
    }
    tiles
}

fn query_tile(agent: &ureq::Agent, tile: (f64, f64, f64, f64), token: &str, per_tile_limit: usize) -> Vec<Value> {
    let (min_lat, min_lon, max_lat, max_lon) = tile;
    let bbox = format!("{min_lon},{min_lat},{max_lon},{max_lat}");
    let limit = per_tile_limit.to_string();
    let q = [
        ("access_token", token),
        ("fields", "id,thumb_1024_url,computed_geometry"),
        ("bbox", bbox.as_str()),
        ("limit", limit.as_str()),
    ];
    match net::get_json(agent, IMAGES_URL, &q) {
        Ok(v) => v.get("data").and_then(Value::as_array).cloned().unwrap_or_default(),
        Err(_) => vec![], // tile failures are silent, as in Python
    }
}

pub fn search_nearby_images(ctx: &Ctx, lat: f64, lon: f64, radius_km: f64, token: &str, max_images: usize, on_progress: &mut dyn FnMut(usize, usize)) -> Result<Vec<Found>> {
    let tiles = bbox_tiles(lat, lon, radius_km);
    (ctx.log)(&format!("Mapillary: scanning {} tiles within {radius_km:.1}km of ({lat:.4},{lon:.4}) ({SEARCH_WORKERS} concurrent)...", tiles.len()));
    let agent = net::agent(SEARCH_WORKERS, 15);
    let total = tiles.len();
    let mut seen: HashSet<String> = HashSet::new();
    let mut results: Vec<Found> = vec![];
    par_for_each_completed(tiles, SEARCH_WORKERS, ctx.cancel, |t| query_tile(&agent, t, token, 5), |done, _i, items| {
        on_progress(done, total);
        for item in items {
            let Some(id) = item.get("id").and_then(Value::as_str) else { continue };
            if !seen.insert(id.to_string()) {
                continue;
            }
            let coords = item.pointer("/computed_geometry/coordinates").and_then(Value::as_array);
            let thumb = item.get("thumb_1024_url").and_then(Value::as_str).filter(|s| !s.is_empty());
            let (Some(coords), Some(thumb)) = (coords, thumb) else { continue };
            let (Some(lo), Some(la)) = (coords.first().and_then(Value::as_f64), coords.get(1).and_then(Value::as_f64)) else { continue };
            results.push(Found { id: id.to_string(), lat: la, lon: lo, thumb_url: thumb.to_string() });
        }
    })?;
    (ctx.log)(&format!("Mapillary: found {} candidate images", results.len()));
    results.truncate(max_images);
    Ok(results)
}

pub use crate::imgio::decode_rgb;

/// Download + decode thumbnails concurrently. Returns (index into `urls`, image) for the successes.
pub fn download_all(ctx: &Ctx, urls: Vec<String>, on_progress: &mut dyn FnMut(usize, usize)) -> Result<Vec<Option<RgbImage>>> {
    let total = urls.len();
    let agent = net::agent(MAX_WORKERS, 15);
    let mut images: Vec<Option<RgbImage>> = (0..total).map(|_| None).collect();
    par_for_each_completed(
        urls,
        MAX_WORKERS,
        ctx.cancel,
        |u| net::get_bytes(&agent, &u, &[]).and_then(|b| decode_rgb(&b)).ok(),
        |done, i, img| {
            images[i] = img;
            on_progress(done, total);
        },
    )?;
    Ok(images)
}

pub fn refine_with_retrieval(
    ctx: &Ctx,
    cluster: &ClusterInfo,
    token: &str,
    radius_km: Option<f64>,
    max_images: usize,
    verify: bool,
    verify_top_n: usize,
    on_progress: Progress,
) -> Result<Vec<Value>> {
    let radius_km = radius_km.unwrap_or_else(|| cluster_radius_km(cluster, 3.0, 15.0));
    let found = {
        let mut cb = |i: usize, n: usize| on_progress("search", i, n);
        search_nearby_images(ctx, cluster.lat, cluster.lon, radius_km, token, max_images, &mut cb)?
    };
    if found.is_empty() {
        return Ok(vec![]);
    }
    (ctx.log)(&format!("Mapillary: downloading {} images ({MAX_WORKERS} concurrent)...", found.len()));
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
                url: ("mapillary_url", format!("https://www.mapillary.com/app/?pKey={}&focus=photo", f.id)),
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
    fn tiles_cover_and_stay_under_limit() {
        let t = bbox_tiles(48.86, 2.29, 3.0);
        // 3 km radius at 48.9N: 2*0.027=0.054 deg lat -> 7 tiles; lon span larger
        assert!(t.len() >= 49);
        assert!(t.iter().all(|(a, b, c, d)| (c - a) < 0.01 && (d - b) < 0.01));
    }
}
