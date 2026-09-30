//! `Models` implementation that runs every network in-process through ONNX
//! Runtime: DISK, LightGlue, and a PLONK variant's step graph plus its image
//! encoder (StreetCLIP or DINOv2), which also provides the retrieval
//! embedding. A variant runs here only when all of its files are installed;
//! otherwise callers run the whole pipeline on the Python model server, so the
//! embedding used for retrieval always matches the one PLONK was trained on.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use image::RgbImage;

use crate::imgio;
use crate::models::{Models, PairMatches};
use crate::onnx::{Accel, Encoder as EncoderSession, Keypoints, OnnxModels};
use crate::plonk::{variant, Encoder, PlonkStep, Variant};
use crate::prep;
use crate::util::{par_for_each_completed, Cancel, Error, Result};

/// Where the ONNX Runtime library and the exported models live.
pub struct Locations {
    pub dll: PathBuf,
    pub models: PathBuf,
}

impl Locations {
    /// Every file of this PLONK variant (`osv5m`, `yfcc`, `inat`) is installed.
    pub fn has(&self, model: &str) -> bool {
        variant(model).is_some_and(|v| v.files().iter().all(|f| self.models.join(f).is_file()))
    }

    /// Keys of the installed PLONK variants.
    pub fn installed(&self) -> Vec<&'static str> {
        crate::plonk::VARIANTS.iter().filter(|v| self.has(v.key)).map(|v| v.key).collect()
    }
}

/// The files every native run needs, whichever PLONK variant it uses.
pub const RETRIEVAL_FILES: [&str; 14] = [
    "streetclip_vision.onnx",
    "disk_unet.onnx",
    "lg_front.onnx",
    "lg_layer0.onnx",
    "lg_layer1.onnx",
    "lg_layer2.onnx",
    "lg_layer3.onnx",
    "lg_layer4.onnx",
    "lg_layer5.onnx",
    "lg_layer6.onnx",
    "lg_layer7.onnx",
    "lg_layer8.onnx",
    "lg_post.onnx",
    "lg_assign.onnx",
];

/// Env overrides (WAYPOINT_ORT_DLL, WAYPOINT_ONNX_DIR) first, then `<data>/ort` and `<data>/onnx`.
/// None when the runtime or any shared model file is missing.
pub fn locate(data_dir: &Path) -> Option<Locations> {
    let dll = std::env::var_os("WAYPOINT_ORT_DLL").map(PathBuf::from).unwrap_or_else(|| data_dir.join("ort").join("onnxruntime.dll"));
    let models = std::env::var_os("WAYPOINT_ONNX_DIR").map(PathBuf::from).unwrap_or_else(|| data_dir.join("onnx"));
    if !(dll.is_file() && RETRIEVAL_FILES.iter().all(|n| models.join(n).is_file())) {
        return None;
    }
    Some(Locations { dll, models })
}

/// Load the ONNX models, with PLONK variant `model` ready to run.
/// `WAYPOINT_ACCEL` = cpu | dml | cuda forces a provider; otherwise DirectML
/// is tried first and CPU is the fallback.
pub fn open(loc: &Locations, model: &str, log: &dyn Fn(&str)) -> Result<NativeModels> {
    use crate::onnx::init_runtime;
    let v = variant(model).ok_or_else(|| Error::Msg(format!("unknown PLONK model '{model}'")))?;
    if !loc.has(v.key) {
        return Err(Error::Msg(format!("the {} model is not installed", v.label)));
    }
    init_runtime(&loc.dll)?;
    let forced = std::env::var("WAYPOINT_ACCEL").ok().and_then(|s| Accel::parse(&s));
    let order: Vec<Accel> = match forced {
        Some(a) => vec![a],
        None => vec![Accel::DirectMl, Accel::Cpu],
    };
    let mut last_err = None;
    for a in order {
        let loaded = OnnxModels::load(&loc.models, a, a).and_then(|m| Ok((m, Active::load(&loc.models, v, a)?)));
        match loaded {
            Ok((m, active)) => {
                log(&format!("ONNX Runtime ready on {} + PLONK {}", m.accel_desc, v.label));
                return Ok(NativeModels::new(m, loc.models.clone(), a, active));
            }
            Err(e) => {
                log(&format!("ONNX Runtime could not use {a:?}: {e}"));
                last_err = Some(e);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| Error::Msg("no execution provider available".into())))
}

/// The PLONK variant in use: its step graph and, unless it is StreetCLIP
/// (always loaded), its image encoder.
struct Active {
    variant: &'static Variant,
    step: Arc<PlonkStep>,
    encoder: Option<Arc<EncoderSession>>,
}

impl Active {
    fn load(dir: &Path, v: &'static Variant, accel: Accel) -> Result<Self> {
        let step = Arc::new(PlonkStep::load(dir, v, accel)?);
        let encoder = match v.encoder {
            Encoder::StreetClip => None,
            other => Some(Arc::new(EncoderSession::load(&dir.join(other.file()), accel)?)),
        };
        Ok(Active { variant: v, step, encoder })
    }
}

pub struct NativeModels {
    pub onnx: OnnxModels,
    dir: PathBuf,
    accel: Accel,
    active: Mutex<Active>,
    target: Mutex<Option<(PathBuf, Arc<Keypoints>)>>,
    /// Conditioning embedding of the target, per (image, variant key).
    target_emb: Mutex<Option<(PathBuf, &'static str, Arc<Vec<f32>>)>>,
}

impl NativeModels {
    fn new(onnx: OnnxModels, dir: PathBuf, accel: Accel, active: Active) -> Self {
        NativeModels { onnx, dir, accel, active: Mutex::new(active), target: Mutex::new(None), target_emb: Mutex::new(None) }
    }

    /// The active variant's step graph and encoder (cheap clones).
    fn current(&self) -> (&'static Variant, Arc<PlonkStep>, Option<Arc<EncoderSession>>) {
        let a = self.active.lock().unwrap();
        (a.variant, a.step.clone(), a.encoder.clone())
    }

    /// One embedding with `encoder` (None: StreetCLIP).
    fn embed_one(&self, encoder: &Option<Arc<EncoderSession>>, img: &RgbImage) -> Result<Vec<f32>> {
        let e = match encoder {
            Some(enc) => enc.embed_pixels(prep::dinov2_pixel_values(img)?, 1)?,
            None => self.onnx.embed_pixels(prep::clip_pixel_values(img)?, 1)?,
        };
        e.into_iter().next().ok_or_else(|| Error::Msg("embedder returned nothing".into()))
    }

    /// PLONK conditioning embedding of the target, computed once per image and variant.
    fn target_embedding(&self, path: &Path) -> Result<Arc<Vec<f32>>> {
        let (v, _, _) = self.current();
        let mut guard = self.target_emb.lock().unwrap();
        if let Some((p, k, e)) = guard.as_ref() {
            if p == path && *k == v.key {
                return Ok(e.clone());
            }
        }
        let e = Arc::new(self.embed_path(path)?);
        *guard = Some((path.to_path_buf(), v.key, e.clone()));
        Ok(e)
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
        let v = variant(model).ok_or_else(|| Error::Msg(format!("unknown PLONK model '{model}'")))?;
        if self.active.lock().unwrap().variant == v {
            return Ok(());
        }
        if !v.files().iter().all(|f| self.dir.join(f).is_file()) {
            return Err(Error::Msg(format!("the {} model is not installed", v.label)));
        }
        let loaded = Active::load(&self.dir, v, self.accel)?;
        *self.active.lock().unwrap() = loaded;
        Ok(())
    }

    fn sample(&self, path: &Path, batch: usize, seed: Option<u64>) -> Result<Vec<[f32; 2]>> {
        self.sample_progress(path, batch, seed, &mut |_, _| true)
    }

    fn sample_progress(
        &self,
        path: &Path,
        batch: usize,
        seed: Option<u64>,
        progress: &mut dyn FnMut(usize, usize) -> bool,
    ) -> Result<Vec<[f32; 2]>> {
        let emb = self.target_embedding(path)?;
        let (_, step, _) = self.current();
        let seed = seed.unwrap_or_else(|| {
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64)
        });
        step.sample(&emb, batch, seed, progress)
    }

    fn embed_path(&self, path: &Path) -> Result<Vec<f32>> {
        let (_, _, encoder) = self.current();
        self.embed_one(&encoder, &imgio::open_rgb(path)?)
    }

    fn embed(&self, images: &[&RgbImage]) -> Result<Vec<Vec<f32>>> {
        // Preprocessing (Pillow-exact resize) is the CPU-heavy part: do it on several threads.
        let (_, _, encoder) = self.current();
        let dino = encoder.is_some();
        let cancel = Cancel::new();
        let mut prepared: Vec<Option<Vec<f32>>> = (0..images.len()).map(|_| None).collect();
        par_for_each_completed(
            images.iter().enumerate().collect(),
            8,
            &cancel,
            |(_, img)| if dino { prep::dinov2_pixel_values(img).ok() } else { prep::clip_pixel_values(img).ok() },
            |_n, i, px| prepared[i] = px,
        )?;
        let mut out = Vec::with_capacity(images.len());
        for px in prepared {
            let px = px.ok_or_else(|| Error::Msg("could not preprocess an image".into()))?;
            let e = match &encoder {
                Some(enc) => enc.embed_pixels(px, 1)?,
                None => self.onnx.embed_pixels(px, 1)?,
            };
            out.push(e.into_iter().next().ok_or_else(|| Error::Msg("embedder returned nothing".into()))?);
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
