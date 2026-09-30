//! PLONK sampling in-process: the Riemannian flow sampler from
//! diff-plonk (`riemannian_flow_sampler`) driving an exported ONNX graph of
//! one Euler step (tools/export/export_plonk.py):
//!
//!   x_{k+1} = projx(x_k + (g_{k+1} - g_k) * net(x_k, g_k, emb)),   projx(v) = v / |v|
//!
//! with g = SigmoidScheduler(-7, 3, tau=1) over 250 steps, starting from
//! standard Gaussian noise, and the result mapped to (lat, lon) degrees.
//! The conditioning embedding comes from the variant's image encoder
//! (StreetCLIP for OSV-5M, DINOv2 for YFCC and iNaturalist); the retrieval
//! stages rank candidates with the same encoder, as the Python pipeline does.

use std::path::Path;
use std::sync::Mutex;

use ort::session::Session;
use ort::value::Tensor;

use crate::onnx::{build_session, Accel};
use crate::util::{Error, Result};

/// Image encoder a PLONK variant is conditioned on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoder {
    /// StreetCLIP's vision tower, CLS token (see `prep::clip_pixel_values`).
    StreetClip,
    /// DINOv2 ViT-L/14 with registers (see `prep::dinov2_pixel_values`).
    DinoV2,
}

impl Encoder {
    pub fn file(self) -> &'static str {
        match self {
            Encoder::StreetClip => "streetclip_vision.onnx",
            Encoder::DinoV2 => "dinov2_vitl14_reg.onnx",
        }
    }
}

/// One of the pretrained PLONK models (plonk.pipe.MODELS, Riemannian flow).
#[derive(Debug, PartialEq, Eq)]
pub struct Variant {
    /// Short name used in settings and on the command line.
    pub key: &'static str,
    pub hf_id: &'static str,
    pub label: &'static str,
    pub step_file: &'static str,
    pub encoder: Encoder,
}

pub const VARIANTS: [Variant; 3] = [
    Variant { key: "osv5m", hf_id: "nicolas-dufour/PLONK_OSV_5M", label: "OSV-5M", step_file: "plonk_osv5m_step.onnx", encoder: Encoder::StreetClip },
    Variant { key: "yfcc", hf_id: "nicolas-dufour/PLONK_YFCC", label: "YFCC", step_file: "plonk_yfcc_step.onnx", encoder: Encoder::DinoV2 },
    Variant { key: "inat", hf_id: "nicolas-dufour/PLONK_iNaturalist", label: "iNaturalist", step_file: "plonk_inat_step.onnx", encoder: Encoder::DinoV2 },
];

/// The variant for a settings key or a Hugging Face id.
pub fn variant(model: &str) -> Option<&'static Variant> {
    VARIANTS.iter().find(|v| v.key == model || v.hf_id == model)
}

impl Variant {
    /// Files this variant needs on top of the shared retrieval models.
    pub fn files(&self) -> Vec<&'static str> {
        let mut f = vec![self.step_file];
        if self.encoder != Encoder::StreetClip {
            f.push(self.encoder.file());
        }
        f
    }
}

pub const NUM_STEPS: usize = 250;
const SCHED_START: f32 = -7.0;
const SCHED_END: f32 = 3.0;
const SCHED_TAU: f32 = 1.0;
const SCHED_CLIP_MIN: f32 = 1e-9;
/// Rows per step call. Bounds activation memory (rows x 2048 floats per
/// block) while keeping the GPU busy.
const CHUNK: usize = 8192;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The sampler's noise schedule, `num_steps + 1` values from ~0 to 1, computed
/// in float32 as PyTorch does.
pub fn schedule(num_steps: usize) -> Vec<f32> {
    let v_start = sigmoid(SCHED_START / SCHED_TAU);
    let v_end = sigmoid(SCHED_END / SCHED_TAU);
    (0..=num_steps)
        .map(|i| {
            let t = 1.0f32 - i as f32 / num_steps as f32;
            let g = (-sigmoid((t * (SCHED_END - SCHED_START) + SCHED_START) / SCHED_TAU) + v_end) / (v_end - v_start);
            g.clamp(SCHED_CLIP_MIN, 1.0)
        })
        .collect()
}

/// Deterministic standard-normal noise (SplitMix64 + Box-Muller). PyTorch's
/// generator can't be reproduced anyway; this only has to be Gaussian and
/// repeatable for a given seed.
pub fn gaussian_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        // (0, 1] so ln() is finite
        let u1 = ((next() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
        let u2 = (next() >> 11) as f64 / (1u64 << 53) as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        let th = 2.0 * std::f64::consts::PI * u2;
        out.push((r * th.cos()) as f32);
        out.push((r * th.sin()) as f32);
    }
    out.truncate(n);
    out
}

/// Unit vectors on the sphere to (lat, lon) degrees (CartesiantoGPS + np.degrees).
pub fn to_lat_lon(x: &[f32]) -> Vec<[f32; 2]> {
    x.chunks_exact(3)
        .map(|v| {
            let lat = v[2].clamp(-1.0, 1.0).asin();
            let lon = v[1].atan2(v[0]);
            [lat.to_degrees(), lon.to_degrees()]
        })
        .collect()
}

pub struct PlonkStep {
    session: Mutex<Session>,
}

impl PlonkStep {
    pub fn load(dir: &Path, variant: &Variant, accel: Accel) -> Result<Self> {
        Ok(PlonkStep { session: Mutex::new(build_session(&dir.join(variant.step_file), accel)?) })
    }

    fn step(&self, x: Vec<f32>, rows: usize, gamma: f32, dt: f32, emb: &[f32]) -> Result<Vec<f32>> {
        let err = |e: ort::Error| Error::Msg(format!("onnxruntime (PLONK): {e}"));
        let tx = Tensor::from_array(([rows, 3usize], x)).map_err(err)?;
        let tg = Tensor::from_array(([1usize], vec![gamma])).map_err(err)?;
        let td = Tensor::from_array(([1usize], vec![dt])).map_err(err)?;
        let te = Tensor::from_array(([1usize, emb.len()], emb.to_vec())).map_err(err)?;
        let mut s = self.session.lock().unwrap();
        let out = s.run(ort::inputs!["x" => tx, "gamma" => tg, "dt" => td, "emb" => te]).map_err(err)?;
        let (_, data) = out["x_next"].try_extract_tensor::<f32>().map_err(err)?;
        Ok(data.to_vec())
    }

    /// Integrate `x0` (rows x 3, row-major) through the full schedule.
    /// `progress(done_steps, total_steps)` is called after every step.
    pub fn integrate(&self, mut x: Vec<f32>, emb: &[f32], progress: &mut dyn FnMut(usize, usize) -> bool) -> Result<Vec<f32>> {
        let g = schedule(NUM_STEPS);
        let rows = x.len() / 3;
        for k in 0..NUM_STEPS {
            let (gamma, dt) = (g[k], g[k + 1] - g[k]);
            let mut next = Vec::with_capacity(x.len());
            for start in (0..rows).step_by(CHUNK) {
                let end = (start + CHUNK).min(rows);
                next.extend(self.step(x[start * 3..end * 3].to_vec(), end - start, gamma, dt, emb)?);
            }
            x = next;
            if !progress(k + 1, NUM_STEPS) {
                return Err(Error::Cancelled);
            }
        }
        Ok(x)
    }

    /// Measured sampling speed in samples/s, used to size the default Samples.
    /// Throughput keeps rising with batch size on a GPU, so like calibrate.py
    /// this escalates through batch sizes while a batch finishes quickly and
    /// reports the largest one measured; a slow (CPU) machine stops early.
    pub fn throughput(&self) -> Result<f64> {
        let emb = vec![0.0f32; 1024];
        self.sample(&emb, 256, 0, &mut |_, _| true)?; // warm-up: first calls compile kernels
        let mut best = 0.0;
        for (i, batch) in [1024usize, 2048, 4096, 8192].into_iter().enumerate() {
            let t = std::time::Instant::now();
            self.sample(&emb, batch, i as u64 + 1, &mut |_, _| true)?;
            let secs = t.elapsed().as_secs_f64();
            best = batch as f64 / secs;
            if secs > 4.0 {
                break;
            }
        }
        Ok(best)
    }

    /// `batch` (lat, lon) samples for one conditioning embedding.
    pub fn sample(&self, emb: &[f32], batch: usize, seed: u64, progress: &mut dyn FnMut(usize, usize) -> bool) -> Result<Vec<[f32; 2]>> {
        let x = self.integrate(gaussian_noise(batch * 3, seed), emb, progress)?;
        Ok(to_lat_lon(&x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_matches_pytorch() {
        // SigmoidScheduler(-7, 3, 1) over 250 steps, as printed by export_plonk.py
        let g = schedule(NUM_STEPS);
        assert_eq!(g.len(), NUM_STEPS + 1);
        assert_eq!(g[0], 1e-9);
        assert!((g[1] - 0.001_933_640_5).abs() < 1e-8, "{}", g[1]);
        assert_eq!(g[NUM_STEPS], 1.0);
        assert!(g.windows(2).all(|w| w[1] >= w[0]));
    }

    #[test]
    fn noise_is_standard_normal_and_repeatable() {
        let a = gaussian_noise(200_001, 7);
        assert_eq!(a, gaussian_noise(200_001, 7));
        assert_ne!(a[..10], gaussian_noise(10, 8)[..]);
        let n = a.len() as f64;
        let mean = a.iter().map(|v| *v as f64).sum::<f64>() / n;
        let var = a.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / n;
        assert!(mean.abs() < 0.01 && (var - 1.0).abs() < 0.01, "mean {mean} var {var}");
    }

    #[test]
    fn variants_resolve_by_key_and_id() {
        assert_eq!(variant("osv5m").unwrap().step_file, "plonk_osv5m_step.onnx");
        assert_eq!(variant("nicolas-dufour/PLONK_iNaturalist").unwrap().key, "inat");
        assert!(variant("nope").is_none());
        assert_eq!(variant("osv5m").unwrap().files(), ["plonk_osv5m_step.onnx"]);
        assert_eq!(variant("yfcc").unwrap().files(), ["plonk_yfcc_step.onnx", "dinov2_vitl14_reg.onnx"]);
    }

    #[test]
    fn lat_lon_of_axes() {
        let ll = to_lat_lon(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(ll[0], [0.0, 0.0]);
        assert_eq!(ll[1], [0.0, 90.0]);
        assert_eq!(ll[2][0], 90.0);
    }
}
