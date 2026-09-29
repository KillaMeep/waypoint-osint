//! `Models` implementation that runs the retrieval networks (embedder, DISK,
//! LightGlue) in-process through ONNX Runtime. PLONK sampling is delegated to
//! a [`Sampler`], which is the Python model server until PLONK itself moves to
//! ONNX.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use image::RgbImage;

use crate::imgio;
use crate::models::{Models, PairMatches};
use crate::onnx::{Keypoints, OnnxModels};
use crate::prep;
use crate::util::{par_for_each_completed, Cancel, Error, Result};

pub trait Sampler: Send + Sync {
    fn load(&self, model: &str) -> Result<()>;
    fn sample(&self, path: &Path, batch: usize, seed: Option<u64>) -> Result<Vec<[f32; 2]>>;
}

/// Where the ONNX Runtime library and the exported models live.
pub struct Locations {
    pub dll: PathBuf,
    pub models: PathBuf,
}

/// Env overrides (WAYPOINT_ORT_DLL, WAYPOINT_ONNX_DIR) first, then `<data>/ort` and `<data>/onnx`.
/// None when the runtime or any model file is missing.
pub fn locate(data_dir: &Path) -> Option<Locations> {
    let dll = std::env::var_os("WAYPOINT_ORT_DLL").map(PathBuf::from).unwrap_or_else(|| data_dir.join("ort").join("onnxruntime.dll"));
    let models = std::env::var_os("WAYPOINT_ONNX_DIR").map(PathBuf::from).unwrap_or_else(|| data_dir.join("onnx"));
    let needed = ["streetclip_vision.onnx", "disk_unet.onnx", "lg_front.onnx", "lg_post.onnx", "lg_assign.onnx", "lg_layer8.onnx"];
    (dll.is_file() && needed.iter().all(|n| models.join(n).is_file())).then_some(Locations { dll, models })
}

/// Load the ONNX models. `WAYPOINT_ACCEL` = cpu | dml | cuda forces a provider;
/// otherwise DirectML is tried first and CPU is the fallback.
pub fn open(loc: &Locations, sampler: Arc<dyn Sampler>, log: &dyn Fn(&str)) -> Result<NativeModels> {
    use crate::onnx::{init_runtime, Accel, OnnxModels};
    init_runtime(&loc.dll)?;
    let forced = std::env::var("WAYPOINT_ACCEL").ok().and_then(|s| Accel::parse(&s));
    let order: Vec<Accel> = match forced {
        Some(a) => vec![a],
        None => vec![Accel::DirectMl, Accel::Cpu],
    };
    let mut last_err = None;
    for a in order {
        match OnnxModels::load(&loc.models, a, a) {
            Ok(m) => {
                log(&format!("ONNX Runtime ready on {}", m.accel_desc));
                return Ok(NativeModels::new(m, sampler));
            }
            Err(e) => {
                log(&format!("ONNX Runtime could not use {a:?}: {e}"));
                last_err = Some(e);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| Error::Msg("no execution provider available".into())))
}

pub struct NativeModels {
    pub onnx: OnnxModels,
    sampler: Arc<dyn Sampler>,
    target: Mutex<Option<(PathBuf, Arc<Keypoints>)>>,
}

impl NativeModels {
    pub fn new(onnx: OnnxModels, sampler: Arc<dyn Sampler>) -> Self {
        NativeModels { onnx, sampler, target: Mutex::new(None) }
    }

    fn target_features(&self, path: &Path) -> Result<Arc<Keypoints>> {
        let mut guard = self.target.lock().unwrap();
        if let Some((p, k)) = guard.as_ref() {
            if p == path {
                return Ok(k.clone());
            }
        }
        let img = imgio::open_rgb(path)?;
        let (chw, w, h) = prep::disk_input(&img)?;
        let k = Arc::new(self.onnx.disk_features(chw, w, h)?);
        *guard = Some((path.to_path_buf(), k.clone()));
        Ok(k)
    }
}

impl Models for NativeModels {
    fn load(&self, model: &str) -> Result<()> {
        self.sampler.load(model)
    }

    fn sample(&self, path: &Path, batch: usize, seed: Option<u64>) -> Result<Vec<[f32; 2]>> {
        self.sampler.sample(path, batch, seed)
    }

    fn embed_path(&self, path: &Path) -> Result<Vec<f32>> {
        let img = imgio::open_rgb(path)?;
        let px = prep::clip_pixel_values(&img)?;
        self.onnx.embed_pixels(px, 1)?.into_iter().next().ok_or_else(|| Error::Msg("embedder returned nothing".into()))
    }

    fn embed(&self, images: &[&RgbImage]) -> Result<Vec<Vec<f32>>> {
        // Preprocessing (Pillow-exact resize) is the CPU-heavy part: do it on several threads.
        let cancel = Cancel::new();
        let mut prepared: Vec<Option<Vec<f32>>> = (0..images.len()).map(|_| None).collect();
        par_for_each_completed(images.iter().enumerate().collect(), 8, &cancel, |(_, img)| prep::clip_pixel_values(img).ok(), |_n, i, px| prepared[i] = px)?;
        let mut out = Vec::with_capacity(images.len());
        for px in prepared {
            let px = px.ok_or_else(|| Error::Msg("could not preprocess an image".into()))?;
            out.push(self.onnx.embed_pixels(px, 1)?.remove(0));
        }
        Ok(out)
    }

    fn match_pair(&self, target: &Path, candidate: &RgbImage) -> Result<PairMatches> {
        let a = self.target_features(target)?;
        let (chw, w, h) = prep::disk_input(candidate)?;
        let b = self.onnx.disk_features(chw, w, h)?;
        if a.xy.len() < 8 || b.xy.len() < 8 {
            return Ok(PairMatches { pts1: vec![], pts2: vec![], n_kp1: a.xy.len(), n_kp2: b.xy.len() });
        }
        self.onnx.lightglue(&a, &b)
    }
}
