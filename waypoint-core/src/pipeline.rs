//! The pipeline orchestrator (port of `run_pipeline.py`): emits the NDJSON
//! event contract that the renderer consumes, one event at a time.

use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

use crate::geo;
use crate::models::Models;
use crate::net;
use crate::retrieval::{ClusterInfo, Ctx};
use crate::sources::{google_sv, mapillary, panoramax};
use crate::sun;
use crate::util::{now_ts, Cancel, Error, Result};

#[derive(Clone, Debug)]
pub enum Mode {
    One,
    Point,
}

#[derive(Clone, Debug)]
pub struct RunArgs {
    pub mode: Mode,
    pub image_path: PathBuf,
    pub model: String,
    pub num_samples: usize,
    pub num_runs: usize,
    pub cluster_radius_km: f64,
    pub top_k: usize,
    pub no_geocode: bool,
    pub no_sun_refine: bool,
    pub no_verify: bool,
    pub verify_top_n: usize,
    pub mapillary_token: Option<String>,
    pub radius_km: Option<f64>,
    pub max_images: usize,
    pub retrieval_top_clusters: usize,
    pub fov: f64,
    pub lat: Option<f64>,
    pub lon: Option<f64>,
}

impl RunArgs {
    /// The same defaults as `run_pipeline.py`'s argparse.
    pub fn new(mode: Mode, image_path: impl Into<PathBuf>) -> Self {
        RunArgs {
            mode,
            image_path: image_path.into(),
            model: "osv5m".into(),
            num_samples: 512,
            num_runs: 3,
            cluster_radius_km: 100.0,
            top_k: 5,
            no_geocode: false,
            no_sun_refine: false,
            no_verify: false,
            verify_top_n: 15,
            mapillary_token: None,
            radius_km: None,
            max_images: 150,
            retrieval_top_clusters: 1,
            fov: 65.0,
            lat: None,
            lon: None,
        }
    }
}

/// Receives events and log lines from a run.
pub trait Sink: Send + Sync {
    fn event(&self, ev: Value);
    fn log(&self, line: &str);
}

fn emit(sink: &dyn Sink, event: &str, fields: Value) {
    let mut m = Map::new();
    m.insert("event".into(), json!(event));
    m.insert("ts".into(), json!(now_ts()));
    if let Value::Object(f) = fields {
        m.extend(f);
    }
    sink.event(Value::Object(m));
}

struct Run<'a> {
    args: &'a RunArgs,
    models: &'a dyn Models,
    sink: &'a dyn Sink,
    cancel: &'a Cancel,
}

impl<'a> Run<'a> {
    fn ctx<'b>(&'b self, log: &'b (dyn Fn(&str) + Sync)) -> Ctx<'b> {
        Ctx { models: self.models, target: &self.args.image_path, cancel: self.cancel, log }
    }

    /// One retrieval source: stage_start, run (errors other than cancel become
    /// empty results, like the Python `except Exception`), stage_done.
    fn stage<F>(&self, stage: &str, key: &str, cluster: &mut Value, f: F) -> Result<()>
    where
        F: FnOnce(&Ctx, &mut dyn FnMut(&str, usize, usize)) -> Result<Vec<Value>>,
    {
        emit(self.sink, "stage_start", json!({ "stage": stage }));
        let log = |s: &str| self.sink.log(s);
        let ctx = self.ctx(&log);
        let mut progress = |phase: &str, completed: usize, total: usize| {
            emit(self.sink, "progress", json!({ "stage": stage, "phase": phase, "completed": completed, "total": total }));
        };
        let matches = match f(&ctx, &mut progress) {
            Ok(m) => m,
            Err(Error::Cancelled) => return Err(Error::Cancelled),
            Err(e) => {
                self.sink.log(&format!("{stage} stage failed: {e}"));
                vec![]
            }
        };
        let matches = Value::Array(matches);
        cluster[key] = matches.clone();
        emit(self.sink, "stage_done", json!({ "stage": stage, "matches": matches }));
        Ok(())
    }

    fn retrieval_stages(&self, cluster: &mut Value, prefix: &str) -> Result<()> {
        let a = self.args;
        let info = ClusterInfo::from_json(cluster);
        let verify = !a.no_verify;

        match a.mapillary_token.as_deref().filter(|t| !t.is_empty()) {
            Some(token) => {
                let stage = format!("{prefix}mapillary");
                self.stage(&stage, "mapillary_matches", cluster, |ctx, p| {
                    mapillary::refine_with_retrieval(ctx, &info, token, a.radius_km, a.max_images, verify, a.verify_top_n, p)
                })?;
            }
            None => emit(self.sink, "stage_skipped", json!({ "stage": format!("{prefix}mapillary"), "reason": "no token configured" })),
        }

        let stage = format!("{prefix}google_sv");
        self.stage(&stage, "google_sv_matches", cluster, |ctx, p| {
            google_sv::refine_with_google_sv(ctx, &info, a.radius_km, a.max_images, verify, a.verify_top_n, p)
        })?;

        let stage = format!("{prefix}panoramax");
        self.stage(&stage, "panoramax_matches", cluster, |ctx, p| {
            panoramax::refine_with_panoramax(ctx, &info, a.radius_km, a.max_images, verify, a.verify_top_n, p)
        })?;
        Ok(())
    }

    fn cluster_json(c: &geo::Cluster) -> Value {
        json!({ "lat": c.lat, "lon": c.lon, "lat_std": c.lat_std, "lon_std": c.lon_std, "count": c.count, "weight": c.weight })
    }

    fn run_one(&self) -> Result<()> {
        let a = self.args;
        emit(self.sink, "stage_start", json!({ "stage": "model_load" }));
        self.models.load(&a.model)?;
        emit(self.sink, "stage_done", json!({ "stage": "model_load" }));
        self.cancel.check()?;

        emit(self.sink, "stage_start", json!({ "stage": "plonk_sampling" }));
        let mut coords: Vec<[f32; 2]> = vec![];
        for _ in 0..a.num_runs {
            self.cancel.check()?;
            coords.extend(self.models.sample(&a.image_path, a.num_samples, None)?);
        }
        let total_samples = a.num_samples * a.num_runs;
        let (clusters, noise_frac) = geo::cluster_samples(&coords, a.cluster_radius_km, 0.03, a.top_k);
        let log = |s: &str| self.sink.log(s);
        let mut cl: Vec<Value> = clusters.iter().map(Self::cluster_json).collect();
        for c in cl.iter_mut() {
            self.cancel.check()?;
            let addr = if a.no_geocode { None } else { net::reverse_geocode(c["lat"].as_f64().unwrap_or(0.0), c["lon"].as_f64().unwrap_or(0.0), &log) };
            c["address"] = addr.map_or(Value::Null, Value::String);
        }
        emit(self.sink, "stage_done", json!({ "stage": "plonk_sampling", "clusters": cl, "noise_frac": noise_frac }));

        if cl.is_empty() {
            let result = json!({ "num_samples": total_samples, "num_runs": a.num_runs, "clusters": cl, "noise_frac": noise_frac });
            emit(self.sink, "done", json!({ "result": result }));
            return Ok(());
        }

        if !a.no_sun_refine {
            emit(self.sink, "stage_start", json!({ "stage": "sun_refine" }));
            let meta = match image::open(&a.image_path) {
                Ok(img) => match sun::refine_clusters(&mut cl, &img.to_rgb8(), a.fov, self.cancel, &log) {
                    Ok(m) => m,
                    Err(Error::Cancelled) => return Err(Error::Cancelled),
                    Err(e) => {
                        self.sink.log(&format!("Sun refine failed: {e}"));
                        None
                    }
                },
                Err(e) => {
                    self.sink.log(&format!("Sun refine failed: {e}"));
                    None
                }
            };
            emit(self.sink, "stage_done", json!({ "stage": "sun_refine", "clusters": cl, "meta": meta }));
        }

        let n_stages = a.retrieval_top_clusters.min(cl.len());
        for i in 0..n_stages {
            self.cancel.check()?;
            let (lat, lon, weight) = (cl[i]["lat"].clone(), cl[i]["lon"].clone(), cl[i]["weight"].clone());
            emit(self.sink, "cluster_selected", json!({ "index": i, "lat": lat, "lon": lon, "weight": weight }));
            self.retrieval_stages(&mut cl[i], &format!("c{i}_"))?;
        }

        let result = json!({ "num_samples": total_samples, "num_runs": a.num_runs, "clusters": cl, "noise_frac": noise_frac });
        emit(self.sink, "done", json!({ "result": result }));
        Ok(())
    }

    fn run_point(&self) -> Result<()> {
        let a = self.args;
        let (Some(lat), Some(lon)) = (a.lat, a.lon) else { return Err(Error::Msg("point mode requires --lat and --lon".into())) };
        emit(self.sink, "stage_start", json!({ "stage": "model_load" }));
        self.models.load(&a.model)?;
        emit(self.sink, "stage_done", json!({ "stage": "model_load" }));

        let mut cluster = json!({ "lat": lat, "lon": lon, "lat_std": 0, "lon_std": 0, "weight": 1.0, "count": 0 });
        emit(self.sink, "cluster_selected", json!({ "index": 0, "lat": lat, "lon": lon, "weight": 1.0 }));
        self.retrieval_stages(&mut cluster, "zoom_")?;
        emit(self.sink, "done", json!({ "result": { "clusters": [cluster], "num_samples": 1, "noise_frac": 0.0 } }));
        Ok(())
    }
}

/// Run one pipeline invocation to completion. Failures are reported the way
/// the Python orchestrator did (an `error` event); the returned value is the
/// process exit code the shell should report (0 ok, 1 error, 2 cancelled).
pub fn run(args: &RunArgs, models: &dyn Models, sink: &dyn Sink, cancel: &Cancel) -> i32 {
    let r = Run { args, models, sink, cancel };
    let res = match args.mode {
        Mode::One => r.run_one(),
        Mode::Point => r.run_point(),
    };
    match res {
        Ok(()) => 0,
        Err(Error::Cancelled) => 2,
        Err(Error::Msg(m)) => {
            emit(sink, "error", json!({ "message": m }));
            1
        }
    }
}

/// Convenience for tests and the CLI: is this path an existing file?
pub fn check_image(path: &Path) -> Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        Err(Error::Msg(format!("image not found: {}", path.display())))
    }
}
