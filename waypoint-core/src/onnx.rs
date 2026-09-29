//! In-process inference through ONNX Runtime (the `ort` crate, loading the
//! runtime DLL from disk at start-up):
//!
//! * StreetCLIP vision tower: the retrieval embedding.
//! * DISK: the UNet runs in ONNX; keypoint selection (5x5 NMS, top-n) and
//!   descriptor sampling are done here, mirroring kornia.
//! * LightGlue: exported as separate graphs; the adaptive loop (early stop and
//!   point pruning) is reimplemented here exactly as kornia's `LightGlue._forward`.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, Once};

use ort::ep;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use crate::models::PairMatches;
use crate::util::{Error, Result};

pub const MAX_KEYPOINTS: usize = 2048;
const N_LAYERS: usize = 9;
const PRUNE_MIN_KPTS: usize = 1536; // kornia: flash-attention threshold on CUDA
const DEPTH_CONFIDENCE: f32 = 0.95;
const WIDTH_CONFIDENCE: f64 = 0.99;

fn e<E: std::fmt::Display>(err: E) -> Error {
    Error::Msg(format!("onnxruntime: {err}"))
}

/// Which execution provider a model should prefer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Accel {
    Cpu,
    DirectMl,
    Cuda,
}

impl Accel {
    pub fn parse(s: &str) -> Option<Accel> {
        match s.to_ascii_lowercase().as_str() {
            "cpu" => Some(Accel::Cpu),
            "dml" | "directml" => Some(Accel::DirectMl),
            "cuda" => Some(Accel::Cuda),
            _ => None,
        }
    }
}

static INIT: Once = Once::new();
static INIT_ERR: Mutex<Option<String>> = Mutex::new(None);

/// Load the ONNX Runtime library. Must happen before any session is created.
pub fn init_runtime(dll: &Path) -> Result<()> {
    INIT.call_once(|| match ort::init_from(dll) {
        Ok(b) => {
            b.commit();
        }
        Err(err) => *INIT_ERR.lock().unwrap() = Some(format!("could not load {}: {err}", dll.display())),
    });
    match INIT_ERR.lock().unwrap().clone() {
        Some(m) => Err(Error::Msg(m)),
        None => Ok(()),
    }
}

pub fn build_session(path: &Path, accel: Accel) -> Result<Session> {
    let level = match std::env::var("WAYPOINT_ORT_OPT").as_deref() {
        Ok("0") => GraphOptimizationLevel::Disable,
        Ok("1") => GraphOptimizationLevel::Level1,
        Ok("2") => GraphOptimizationLevel::Level2,
        _ => GraphOptimizationLevel::Level3,
    };
    let mut b = Session::builder().map_err(e)?.with_optimization_level(level).map_err(e)?;
    match accel {
        Accel::Cpu => {}
        Accel::DirectMl => {
            // DirectML requires sequential execution and no memory-pattern reuse.
            b = b
                .with_memory_pattern(false)
                .map_err(e)?
                .with_parallel_execution(false)
                .map_err(e)?
                .with_execution_providers([ep::DirectML::default().build()])
                .map_err(e)?;
        }
        Accel::Cuda => {
            b = b.with_execution_providers([ep::CUDA::default().build()]).map_err(e)?;
        }
    }
    b.commit_from_file(path).map_err(|err| Error::Msg(format!("loading {}: {err}", path.display())))
}

/// (name of first output, data) helper for f32 outputs.
fn run_f32(session: &Mutex<Session>, inputs: Vec<(&str, Tensor<f32>)>, outputs: &[&str]) -> Result<Vec<(Vec<i64>, Vec<f32>)>> {
    let mut s = session.lock().unwrap();
    let mut feed: Vec<(std::borrow::Cow<'static, str>, ort::session::SessionInputValue<'static>)> = vec![];
    for (name, t) in inputs {
        feed.push((name.to_string().into(), t.into()));
    }
    let out = s.run(feed).map_err(e)?;
    outputs
        .iter()
        .map(|n| {
            let (shape, data) = out[*n].try_extract_tensor::<f32>().map_err(e)?;
            Ok((shape.iter().copied().collect(), data.to_vec()))
        })
        .collect()
}

pub struct Keypoints {
    /// (x, y) pixel coordinates in the DISK input frame
    pub xy: Vec<[f32; 2]>,
    /// N x 128, L2-normalised
    pub desc: Vec<f32>,
}

pub struct OnnxModels {
    clip: Mutex<Session>,
    unet: Mutex<Session>,
    lg_front: Mutex<Session>,
    lg_layers: Vec<Mutex<Session>>,
    lg_post: Mutex<Session>,
    lg_assign: Mutex<Session>,
    pub accel_desc: String,
}

pub struct ModelFiles {
    pub dir: PathBuf,
}

impl ModelFiles {
    pub fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }
}

impl OnnxModels {
    /// `heavy` (StreetCLIP, DISK UNet) uses `accel`; LightGlue's small,
    /// shape-varying graphs use `lg_accel`.
    pub fn load(dir: &Path, accel: Accel, lg_accel: Accel) -> Result<Self> {
        let f = ModelFiles { dir: dir.to_path_buf() };
        let m = |n: &str, a: Accel| build_session(&f.path(n), a).map(Mutex::new);
        Ok(OnnxModels {
            clip: m("streetclip_vision.onnx", accel)?,
            unet: m("disk_unet.onnx", accel)?,
            lg_front: m("lg_front.onnx", lg_accel)?,
            lg_layers: (0..N_LAYERS).map(|i| m(&format!("lg_layer{i}.onnx"), lg_accel)).collect::<Result<_>>()?,
            lg_post: m("lg_post.onnx", lg_accel)?,
            lg_assign: m("lg_assign.onnx", lg_accel)?,
            accel_desc: format!("{accel:?} (StreetCLIP, DISK), {lg_accel:?} (LightGlue)"),
        })
    }

    /// StreetCLIP CLS embeddings for CLIP-preprocessed pixel batches (n x 3 x 336 x 336).
    pub fn embed_pixels(&self, pixels: Vec<f32>, n: usize) -> Result<Vec<Vec<f32>>> {
        let t = Tensor::from_array(([n, 3usize, 336, 336], pixels)).map_err(e)?;
        let out = run_f32(&self.clip, vec![("pixel_values", t)], &["emb"])?;
        let (shape, data) = &out[0];
        let dim = *shape.last().unwrap_or(&0) as usize;
        Ok(data.chunks_exact(dim).map(|c| c.to_vec()).collect())
    }

    /// DISK keypoints + descriptors (`disk(img, n=2048, pad_if_not_divisible=True)`).
    pub fn disk_features(&self, chw: Vec<f32>, w: usize, h: usize) -> Result<Keypoints> {
        let t = Tensor::from_array(([1usize, 3, h, w], chw)).map_err(e)?;
        let out = run_f32(&self.unet, vec![("image", t)], &["out"])?;
        let data = &out[0].1;
        let plane = h * w;
        let heat = &data[128 * plane..129 * plane];
        Ok(select_keypoints(heat, &data[..128 * plane], w, h, MAX_KEYPOINTS))
    }

    /// Full pair matching: LightGlue on two DISK feature sets, kornia-equivalent.
    /// Returns matched keypoint coordinates.
    pub fn lightglue(&self, a: &Keypoints, b: &Keypoints) -> Result<PairMatches> {
        let (m, n) = (a.xy.len(), b.xy.len());
        if std::env::var_os("WAYPOINT_DEBUG").is_some() {
            eprintln!("input: kp0[0]={:?} desc0 len {} sum {:.4} first {:?}", a.xy[0], a.desc.len(), a.desc.iter().map(|x| *x as f64).sum::<f64>(), &a.desc[..3]);
        }
        let norm = |k: &Keypoints| -> Vec<f32> {
            let (mut mx, mut my) = (f32::MIN, f32::MIN);
            for p in &k.xy {
                mx = mx.max(p[0]);
                my = my.max(p[1]);
            }
            let (sx, sy) = (mx / 2.0, my / 2.0);
            let scale = mx.max(my) / 2.0;
            k.xy.iter().flat_map(|p| [(p[0] - sx) / scale, (p[1] - sy) / scale]).collect()
        };
        let tk0 = Tensor::from_array(([1usize, m, 2], norm(a))).map_err(e)?;
        let td0 = Tensor::from_array(([1usize, m, 128], a.desc.clone())).map_err(e)?;
        let tk1 = Tensor::from_array(([1usize, n, 2], norm(b))).map_err(e)?;
        let td1 = Tensor::from_array(([1usize, n, 128], b.desc.clone())).map_err(e)?;
        let mut r = run_f32(&self.lg_front, vec![("kpts0", tk0), ("desc0", td0), ("kpts1", tk1), ("desc1", td1)], &["desc0o", "desc1o", "enc0", "enc1"])?;
        let (mut e1, mut e0) = (r.pop().unwrap().1, r.pop().unwrap().1);
        let (mut d1, mut d0) = (r.pop().unwrap().1, r.pop().unwrap().1);
        if std::env::var_os("WAYPOINT_DEBUG").is_some() {
            let s = |v: &Vec<f32>| v.iter().map(|x| *x as f64).sum::<f64>();
            eprintln!("front: d0 sum {:.4} d1 sum {:.4} e0 sum {:.4} e1 sum {:.4} (len {} {} {} {})", s(&d0), s(&d1), s(&e0), s(&e1), d0.len(), d1.len(), e0.len(), e1.len());
        }
        // enc layout [2, 1, 1, len, 64]; d layout [1, len, 256]
        let mut ind0: Vec<usize> = (0..m).collect();
        let mut ind1: Vec<usize> = (0..n).collect();

        let mut last_layer = 0usize;
        for i in 0..N_LAYERS {
            last_layer = i;
            let (m0, n0) = (ind0.len(), ind1.len());
            let mk = |name: &'static str, v: &Vec<f32>, shape: Vec<usize>| -> Result<(&'static str, Tensor<f32>)> {
                Ok((name, Tensor::from_array((shape, v.clone())).map_err(e)?))
            };
            let mut r = run_f32(
                &self.lg_layers[i],
                vec![mk("desc0", &d0, vec![1, m0, 256])?, mk("desc1", &d1, vec![1, n0, 256])?, mk("enc0", &e0, vec![2, 1, 1, m0, 64])?, mk("enc1", &e1, vec![2, 1, 1, n0, 64])?],
                &["desc0o", "desc1o"],
            )?;
            d1 = r.pop().unwrap().1;
            d0 = r.pop().unwrap().1;
            if i == N_LAYERS - 1 {
                continue;
            }
            // token confidences and matchability of the current (possibly pruned) sets
            let (tok0, tok1, mat0, mat1) = self.post(&d0, &d1, i)?;
            let thr = confidence_threshold(i);
            // check_if_stop: pruned points count as confident, denominator is the original m + n
            let below = tok0.iter().chain(tok1.iter()).filter(|c| **c < thr).count();
            let ratio = 1.0f32 - below as f32 / (m + n) as f32;
            if std::env::var_os("WAYPOINT_DEBUG").is_some() {
                eprintln!("  layer {i}: ratio {ratio:.4}, tok0 mean {:.4}, thr {thr}", tok0.iter().sum::<f32>() / tok0.len() as f32);
            }
            if ratio > DEPTH_CONFIDENCE {
                break;
            }
            let keep_thr = (1.0 - WIDTH_CONFIDENCE) as f32;
            if ind0.len() > PRUNE_MIN_KPTS {
                let keep: Vec<usize> = (0..ind0.len()).filter(|&j| mat0[j] > keep_thr || tok0[j] <= thr).collect();
                (ind0, d0, e0) = gather(&ind0, &d0, &e0, &keep, 256);
            }
            if ind1.len() > PRUNE_MIN_KPTS {
                let keep: Vec<usize> = (0..ind1.len()).filter(|&j| mat1[j] > keep_thr || tok1[j] <= thr).collect();
                (ind1, d1, e1) = gather(&ind1, &d1, &e1, &keep, 256);
            }
        }

        let (m0, n0) = (ind0.len(), ind1.len());
        if m0 == 0 || n0 == 0 {
            return Ok(PairMatches { pts1: vec![], pts2: vec![], n_kp1: m, n_kp2: n });
        }
        let matches0 = self.assign(&d0, &d1, m0, n0, last_layer)?;
        if std::env::var_os("WAYPOINT_DEBUG").is_some() {
            eprintln!("lightglue: m={m} n={n} pruned to {m0}/{n0}, last layer {last_layer}, valid {}", matches0.iter().filter(|x| **x >= 0).count());
        }
        let (mut pts1, mut pts2) = (vec![], vec![]);
        for (j, &t) in matches0.iter().enumerate() {
            if t >= 0 {
                pts1.push(a.xy[ind0[j]]);
                pts2.push(b.xy[ind1[t as usize]]);
            }
        }
        Ok(PairMatches { pts1, pts2, n_kp1: m, n_kp2: n })
    }

    fn post(&self, d0: &[f32], d1: &[f32], layer: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let (m, n) = (d0.len() / 256, d1.len() / 256);
        let mut s = self.lg_post.lock().unwrap();
        let t0 = Tensor::from_array(([1usize, m, 256], d0.to_vec())).map_err(e)?;
        let t1 = Tensor::from_array(([1usize, n, 256], d1.to_vec())).map_err(e)?;
        let li = Tensor::from_array((Vec::<usize>::new(), vec![layer as i64])).map_err(e)?;
        let out = s.run(ort::inputs!["desc0" => t0, "desc1" => t1, "layer" => li]).map_err(e)?;
        let g = |k: &str| -> Result<Vec<f32>> { Ok(out[k].try_extract_tensor::<f32>().map_err(e)?.1.to_vec()) };
        Ok((g("tok0")?, g("tok1")?, g("mat0")?, g("mat1")?))
    }

    fn assign(&self, d0: &[f32], d1: &[f32], m: usize, n: usize, layer: usize) -> Result<Vec<i64>> {
        let mut s = self.lg_assign.lock().unwrap();
        let t0 = Tensor::from_array(([1usize, m, 256], d0.to_vec())).map_err(e)?;
        let t1 = Tensor::from_array(([1usize, n, 256], d1.to_vec())).map_err(e)?;
        let li = Tensor::from_array((Vec::<usize>::new(), vec![layer as i64])).map_err(e)?;
        let out = s.run(ort::inputs!["desc0" => t0, "desc1" => t1, "layer" => li]).map_err(e)?;
        Ok(out["matches0"].try_extract_tensor::<i64>().map_err(e)?.1.to_vec())
    }
}

/// `LightGlue.confidence_threshold`, computed in float32 like the module buffer.
fn confidence_threshold(layer: usize) -> f32 {
    let t = 0.8 + 0.1 * (-4.0 * layer as f64 / N_LAYERS as f64).exp();
    (t.clamp(0.0, 1.0)) as f32
}

fn gather(ind: &[usize], d: &[f32], enc: &[f32], keep: &[usize], dim: usize) -> (Vec<usize>, Vec<f32>, Vec<f32>) {
    let len = ind.len();
    let ed = 64;
    let ind2: Vec<usize> = keep.iter().map(|&k| ind[k]).collect();
    let mut d2 = Vec::with_capacity(keep.len() * dim);
    for &k in keep {
        d2.extend_from_slice(&d[k * dim..(k + 1) * dim]);
    }
    let mut e2 = Vec::with_capacity(2 * keep.len() * ed);
    for plane in 0..2 {
        for &k in keep {
            let o = (plane * len + k) * ed;
            e2.extend_from_slice(&enc[o..o + ed]);
        }
    }
    (ind2, d2, e2)
}

/// kornia `heatmap_to_keypoints` + `Keypoints.merge_with_descriptors`.
pub fn select_keypoints(heat: &[f32], desc: &[f32], w: usize, h: usize, n: usize) -> Keypoints {
    let at = |y: usize, x: usize| heat[y * w + x];
    // 5x5 non-maxima suppression: strict maximum among its 24 neighbours; a 2px border is never a keypoint.
    let mut cand: Vec<(usize, usize, f32)> = vec![];
    if h > 4 && w > 4 {
        for y in 2..h - 2 {
            for x in 2..w - 2 {
                let c = at(y, x);
                if c <= 0.0 {
                    continue;
                }
                let mut is_max = true;
                'nb: for dy in 0..5 {
                    for dx in 0..5 {
                        if dy == 2 && dx == 2 {
                            continue;
                        }
                        if !(c > at(y + dy - 2, x + dx - 2)) {
                            is_max = false;
                            break 'nb;
                        }
                    }
                }
                if is_max {
                    cand.push((y, x, c));
                }
            }
        }
    }
    let mut keep: Vec<(usize, usize, f32)> = cand.clone();
    if !cand.is_empty() {
        // threshold = the min(n+1, count)-th largest score; keep strictly greater (kornia's kthvalue trick)
        let k = (n + 1).min(cand.len());
        let mut scores: Vec<f32> = cand.iter().map(|c| c.2).collect();
        scores.sort_by(|a, b| b.total_cmp(a));
        let thr = scores[k - 1];
        keep = cand.into_iter().filter(|c| c.2 > thr).collect();
        keep.truncate(n);
    }
    let plane = h * w;
    let mut xy = Vec::with_capacity(keep.len());
    let mut dd = Vec::with_capacity(keep.len() * 128);
    for (y, x, _) in keep {
        xy.push([x as f32, y as f32]);
        let mut v = [0f32; 128];
        let mut norm = 0f32;
        for (c, slot) in v.iter_mut().enumerate() {
            *slot = desc[c * plane + y * w + x];
            norm += *slot * *slot;
        }
        let norm = norm.sqrt().max(1e-12);
        dd.extend(v.iter().map(|q| q / norm));
    }
    Keypoints { xy, desc: dd }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(w: usize, h: usize, peaks: &[(usize, usize, f32)]) -> (Vec<f32>, Vec<f32>) {
        let mut heat = vec![-1.0f32; w * h];
        for &(x, y, v) in peaks {
            heat[y * w + x] = v;
        }
        // channel 0 of the descriptor at pixel i holds i + 1, all other channels are zero
        let mut desc = vec![0f32; 128 * w * h];
        for i in 0..w * h {
            desc[i] = i as f32 + 1.0;
        }
        (heat, desc)
    }

    #[test]
    fn nms_keeps_strict_maxima_and_skips_the_border() {
        let (heat, desc) = synthetic(20, 20, &[(10, 10, 3.0), (12, 10, 2.0), (5, 5, 1.0), (1, 1, 9.0)]);
        // (12,10) sits inside the 5x5 window of (10,10) and is suppressed; (1,1) is in
        // the 2px border and never a keypoint. With fewer candidates than n, kornia
        // drops the lowest-scoring one (strictly-greater-than-kth trick): (5,5).
        let k = select_keypoints(&heat, &desc, 20, 20, 2048);
        assert_eq!(k.xy, vec![[10.0, 10.0]]);
        assert_eq!(k.desc.len(), 128);
        assert!((k.desc[0] - 1.0).abs() < 1e-6, "descriptor is L2-normalised");
    }

    #[test]
    fn top_n_keeps_strictly_above_the_n_plus_1th_score() {
        let peaks: Vec<(usize, usize, f32)> = (0..6).map(|i| (3 + i * 6, 5, 1.0 + i as f32)).collect();
        let (heat, desc) = synthetic(50, 12, &peaks);
        let k = select_keypoints(&heat, &desc, 50, 12, 3);
        // scores 6,5,4,3,2,1 with n = 3: threshold is the 4th largest (3.0); keep > 3.0, in raster order
        assert_eq!(k.xy.iter().map(|p| p[0] as usize).collect::<Vec<_>>(), vec![21, 27, 33]);
    }

    #[test]
    fn confidence_thresholds_match_the_kornia_buffer() {
        assert!((confidence_threshold(0) - 0.9).abs() < 1e-6);
        assert!((confidence_threshold(1) - 0.864_118_04).abs() < 1e-6);
    }

    #[test]
    fn gather_keeps_descriptor_and_encoding_rows_together() {
        let len = 4;
        let ind: Vec<usize> = vec![10, 11, 12, 13];
        let d: Vec<f32> = (0..len * 256).map(|i| (i / 256) as f32).collect();
        let e: Vec<f32> = (0..2 * len * 64).map(|i| (i / 64) as f32).collect();
        let (i2, d2, e2) = gather(&ind, &d, &e, &[1, 3], 256);
        assert_eq!(i2, vec![11, 13]);
        assert!(d2[..256].iter().all(|v| *v == 1.0) && d2[256..].iter().all(|v| *v == 3.0));
        // plane 0 rows 1 and 3, then plane 1 rows 1 and 3 (value = plane * len + row)
        assert_eq!((e2[0], e2[64], e2[128], e2[192]), (1.0, 3.0, 5.0, 7.0));
    }
}
