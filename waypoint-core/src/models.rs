//! The neural pieces behind a small trait, so the orchestrator does not care
//! whether they run in the Python model server or in-process (ONNX).

use std::path::Path;

use image::RgbImage;

use crate::util::Result;

/// Matched keypoints of one image pair, after DISK + LightGlue.
pub struct PairMatches {
    pub pts1: Vec<[f32; 2]>,
    pub pts2: Vec<[f32; 2]>,
    pub n_kp1: usize,
    pub n_kp2: usize,
}

pub trait Models: Send + Sync {
    /// Load the PLONK pipeline (`osv5m`, `yfcc`, ...). Slow the first time.
    fn load(&self, model: &str) -> Result<()>;
    /// Draw `batch` (lat, lon) degree samples for the image at `path`.
    /// `seed` makes the run reproducible where the backend supports it.
    fn sample(&self, path: &Path, batch: usize, seed: Option<u64>) -> Result<Vec<[f32; 2]>>;
    /// PLONK's own image embedding (StreetCLIP CLS token) for the target file.
    fn embed_path(&self, path: &Path) -> Result<Vec<f32>>;
    /// Same embedding for in-memory images.
    fn embed(&self, images: &[&RgbImage]) -> Result<Vec<Vec<f32>>>;
    /// DISK + LightGlue matches between the target file and a candidate.
    fn match_pair(&self, target: &Path, candidate: &RgbImage) -> Result<PairMatches>;
}
