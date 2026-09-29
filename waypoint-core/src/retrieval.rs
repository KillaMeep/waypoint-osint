//! Shared retrieval-refinement machinery: embedding similarity ranking and
//! geometric verification of the top candidates (the common tail of
//! `mapillary_refine` / `google_sv_refine` / `panoramax_refine`).

use std::path::Path;

use image::RgbImage;
use serde_json::{json, Map, Value};

use crate::models::Models;
use crate::ransac;
use crate::util::{py_round, Cancel, Error, Result};

/// `on_progress(phase, completed, total)`
pub type Progress<'a> = &'a mut dyn FnMut(&str, usize, usize);

/// Everything a source needs to run.
pub struct Ctx<'a> {
    pub models: &'a dyn Models,
    pub target: &'a Path,
    pub cancel: &'a Cancel,
    pub log: &'a (dyn Fn(&str) + Sync),
}

#[derive(Clone, Copy, Debug)]
pub struct ClusterInfo {
    pub lat: f64,
    pub lon: f64,
    pub lat_std: f64,
    pub lon_std: f64,
}

impl ClusterInfo {
    pub fn from_json(c: &Value) -> Self {
        let f = |k: &str| c.get(k).and_then(Value::as_f64).unwrap_or(0.0);
        ClusterInfo { lat: f("lat"), lon: f("lon"), lat_std: f("lat_std"), lon_std: f("lon_std") }
    }
}

/// Search radius from PLONK's own reported spread for the cluster, clamped.
pub fn cluster_radius_km(c: &ClusterInfo, min_km: f64, max_km: f64) -> f64 {
    let lat_km = c.lat_std * 111.0;
    let lon_km = c.lon_std * 111.0 * c.lat.to_radians().cos().max(0.1);
    lat_km.hypot(lon_km).clamp(min_km, max_km)
}

pub fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb + 1e-8)
}

/// One candidate image with its source metadata, ready to be scored.
pub struct Scorable {
    pub lat: f64,
    pub lon: f64,
    /// (json key, url) of the "open this photo" link.
    pub url: (&'static str, String),
    pub image: RgbImage,
}

/// `verify_utils.match_pair` semantics on top of the model's matches.
pub fn verify_one(models: &dyn Models, target: &Path, cand: &RgbImage) -> (usize, usize) {
    match models.match_pair(target, cand) {
        Ok(m) => {
            if m.n_kp1 < 8 || m.n_kp2 < 8 {
                return (0, 0);
            }
            let total = m.pts1.len();
            if total < 8 {
                return (0, total);
            }
            (ransac::fundamental_ransac_inliers(&m.pts1, &m.pts2, 3.0, 0.99, 1000, 0x5EED), total)
        }
        Err(_) => (0, 0),
    }
}

/// Rank `cands` by embedding similarity to the target, geometrically verify
/// the top `verify_top_n`, and return the best `top_k` as JSON match objects.
pub fn score_and_verify(
    ctx: &Ctx,
    cands: Vec<Scorable>,
    verify: bool,
    verify_top_n: usize,
    top_k: usize,
    on_progress: Progress,
) -> Result<Vec<Value>> {
    ctx.cancel.check()?;
    let target_emb = ctx.models.embed_path(ctx.target)?;
    let refs: Vec<&RgbImage> = cands.iter().map(|c| &c.image).collect();
    let mut embs = Vec::with_capacity(cands.len());
    // Chunked so cancellation is noticed between batches.
    for chunk in refs.chunks(16) {
        ctx.cancel.check()?;
        embs.extend(ctx.models.embed(chunk)?);
    }
    if embs.len() != cands.len() {
        return Err(Error::Msg("embedder returned the wrong number of vectors".into()));
    }

    struct Row {
        sim: f64,
        idx: usize,
        inliers: Option<(usize, usize)>,
    }
    let mut rows: Vec<Row> = embs.iter().enumerate().map(|(i, e)| Row { sim: py_round(cosine_sim(&target_emb, e), 4), idx: i, inliers: None }).collect();
    rows.sort_by(|a, b| b.sim.total_cmp(&a.sim)); // stable, like list.sort(reverse=True) on the key

    if verify && !rows.is_empty() {
        let n = verify_top_n.min(rows.len());
        (ctx.log)(&format!("geometrically verifying top {n} candidates..."));
        for i in 0..n {
            ctx.cancel.check()?;
            let r = &mut rows[i];
            r.inliers = Some(verify_one(ctx.models, ctx.target, &cands[r.idx].image));
            on_progress("verify", i + 1, n);
        }
        // Stable sort by inliers, descending (Python: sort(key=..., reverse=True) keeps ties in order).
        rows[..n].sort_by(|a, b| b.inliers.map_or(0, |x| x.0).cmp(&a.inliers.map_or(0, |x| x.0)));
    }

    Ok(rows
        .into_iter()
        .take(top_k)
        .map(|r| {
            let c = &cands[r.idx];
            let mut m = Map::new();
            m.insert("similarity".into(), json!(r.sim));
            m.insert("lat".into(), json!(c.lat));
            m.insert("lon".into(), json!(c.lon));
            m.insert(c.url.0.into(), json!(c.url.1));
            if let Some((inl, tot)) = r.inliers {
                m.insert("inliers".into(), json!(inl));
                m.insert("total_matches".into(), json!(tot));
            }
            Value::Object(m)
        })
        .collect())
}
