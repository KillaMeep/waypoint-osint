//! Phase 2 parity gates for the ONNX embedder / DISK / LightGlue against the
//! recorded PyTorch outputs. Ignored by default (needs the runtime DLL, the
//! exported models and tools/parity/ref):
//!
//!   $env:WAYPOINT_ORT_DLL="...\onnxruntime.dll"; $env:WAYPOINT_ONNX_DIR="...\onnx"
//!   cargo test --release --test onnx_parity -- --ignored --nocapture

use std::path::PathBuf;
use std::time::Instant;

use serde_json::Value;
use waypoint_core::onnx::{init_runtime, Accel, Keypoints, OnnxModels};
use waypoint_core::{prep, ransac};

/// The test photos live outside the repo: set WAYPOINT_TEST_IMAGES to the folder holding test_pano.jpg.
fn test_image(name: &str) -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var_os("WAYPOINT_TEST_IMAGES")?).join(format!("test_{name}.jpg"));
    p.is_file().then_some(p)
}

fn need_image(name: &str) -> PathBuf {
    test_image(name).expect("set WAYPOINT_TEST_IMAGES to the folder holding the test photos")
}

fn refdir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/parity/ref")
}

fn models() -> OnnxModels {
    let dll = std::env::var("WAYPOINT_ORT_DLL").expect("WAYPOINT_ORT_DLL");
    let dir = std::env::var("WAYPOINT_ONNX_DIR").expect("WAYPOINT_ONNX_DIR");
    let accel = Accel::parse(&std::env::var("WAYPOINT_ACCEL").unwrap_or("dml".into())).unwrap();
    let lg = Accel::parse(&std::env::var("WAYPOINT_LG_ACCEL").unwrap_or("cpu".into())).unwrap();
    init_runtime(std::path::Path::new(&dll)).unwrap();
    let t = Instant::now();
    let m = OnnxModels::load(std::path::Path::new(&dir), accel, lg).unwrap();
    eprintln!("models loaded in {:?} on {}", t.elapsed(), m.accel_desc);
    m
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    dot / (a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt() * b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt())
}

fn npy_f32(p: &PathBuf) -> Vec<f32> {
    npyz::NpyFile::new(std::io::BufReader::new(std::fs::File::open(p).unwrap())).unwrap().into_vec::<f32>().unwrap()
}

fn embed(m: &OnnxModels, img: &image::RgbImage) -> Vec<f32> {
    m.embed_pixels(prep::clip_pixel_values(img).unwrap(), 1).unwrap().remove(0)
}

fn candidates() -> Vec<(String, PathBuf)> {
    let man: Value = serde_json::from_str(&std::fs::read_to_string(refdir().join("cands/manifest.json")).unwrap()).unwrap();
    man.as_array().unwrap().iter().map(|c| (c["file"].as_str().unwrap().to_string(), refdir().join("cands").join(c["file"].as_str().unwrap()))).collect()
}

/// Where does embedding error come from: model/EP, resize, or JPEG decode?
#[test]
#[ignore]
fn embedder_error_budget() {
    let m = models();
    for name in ["pano"] {
        let path = need_image(name);
        let want = npy_f32(&refdir().join(format!("{name}_emb.npy")).exists().then(|| refdir().join(format!("{name}_emb.npy"))).unwrap_or_else(|| refdir().join("nonexistent")));
        let _ = want;
        let pv: Vec<f32> = npy_f32(&refdir().join(format!("pv_{name}.npy")));
        // reference embedding: recompute through the CPU EP path is unavailable here, so use the recorded one when present
        let ref_emb = {
            let p = refdir().join(format!("{name}_emb.npy"));
            if p.exists() { Some(npy_f32(&p)) } else { None }
        };
        let e_pv = m.embed_pixels(pv.clone(), 1).unwrap().remove(0);
        let rgb: Vec<u8> = npyz::NpyFile::new(std::io::BufReader::new(std::fs::File::open(refdir().join(format!("rgb_{name}.npy"))).unwrap())).unwrap().into_vec().unwrap();
        let shape = npyz::NpyFile::new(std::io::BufReader::new(std::fs::File::open(refdir().join(format!("rgb_{name}.npy"))).unwrap())).unwrap().shape().to_vec();
        let pil_img = image::RgbImage::from_raw(shape[1] as u32, shape[0] as u32, rgb).unwrap();
        let e_pil = embed(&m, &pil_img);
        eprint!("{name:>22}: cos(ONNX[ref pixels], ONNX[PIL pixels + our resize]) = {:.6}", cos(&e_pv, &e_pil));
        {
            let e_dec = embed(&m, &waypoint_core::imgio::open_rgb(&path).unwrap());
            eprint!("  cos(ONNX[ref pixels], ONNX[our decode + our resize]) = {:.6}", cos(&e_pv, &e_dec));
        }
        if let Some(r) = ref_emb {
            eprint!("  | vs PyTorch: ref-pixels {:.6}, PIL+resize {:.6}", cos(&r, &e_pv), cos(&r, &e_pil));
        }
        eprintln!();
    }
}

#[test]
#[ignore]
fn embedder_matches_pytorch() {
    let m = models();
    let mut z = npyz::npz::NpzArchive::open(refdir().join("cand_embs.npz")).unwrap();
    let target_ref: Vec<f32> = z.by_name("target").unwrap().unwrap().into_vec().unwrap();
    let target = waypoint_core::imgio::open_rgb(&need_image("pano")).unwrap();
    let t0 = Instant::now();
    let te = embed(&m, &target);
    eprintln!("target embed {:?}, cosine vs PyTorch {:.6}", t0.elapsed(), cos(&te, &target_ref));
    let mut coss = vec![];
    let mut sims_rust = vec![];
    let mut sims_ref = vec![];
    let start = Instant::now();
    for (name, path) in candidates() {
        let img = waypoint_core::imgio::open_rgb(std::path::Path::new(&path)).unwrap();
        let e = embed(&m, &img);
        let key = name.replace('.', "_");
        let r: Vec<f32> = z.by_name(&key).unwrap().unwrap().into_vec().unwrap();
        coss.push(cos(&e, &r));
        sims_rust.push((name.clone(), cos(&te, &e)));
        sims_ref.push((name, cos(&target_ref, &r)));
    }
    eprintln!("{} candidate embeddings in {:?}", coss.len(), start.elapsed());
    let min = coss.iter().cloned().fold(1.0, f64::min);
    let mean = coss.iter().sum::<f64>() / coss.len() as f64;
    eprintln!("embedding cosine vs PyTorch: min {min:.6} mean {mean:.6}");
    let rank = |v: &Vec<(String, f64)>| -> Vec<String> {
        let mut v = v.clone();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v.into_iter().map(|x| x.0).collect()
    };
    let (rr, rf) = (rank(&sims_rust), rank(&sims_ref));
    let top5_common = rr[..5].iter().filter(|n| rf[..5].contains(n)).count();
    let max_sim_diff = sims_rust.iter().zip(&sims_ref).map(|(a, b)| (a.1 - b.1).abs()).fold(0.0, f64::max);
    eprintln!("retrieval: top-5 overlap {top5_common}/5, top-1 same: {}, max |similarity diff| {max_sim_diff:.5}", rr[0] == rf[0]);
    eprintln!("ranking rust: {:?}", &rr[..5]);
    eprintln!("ranking ref : {:?}", &rf[..5]);
    assert!(min >= 0.999, "embedding cosine {min}");
}

fn load_kp(z: &mut npyz::npz::NpzArchive<std::io::BufReader<std::fs::File>>, kp: &str, desc: &str) -> Keypoints {
    // numpy may have stored the array in Fortran order (transposed views): normalise to row-major.
    let mut rd = |name: &str| -> Vec<f32> {
        let f = z.by_name(name).unwrap().unwrap();
        let shape: Vec<usize> = f.shape().iter().map(|d| *d as usize).collect();
        let fortran = matches!(f.order(), npyz::Order::Fortran);
        let raw: Vec<f32> = f.into_vec().unwrap();
        if !fortran {
            return raw;
        }
        let (r, c) = (shape[0], shape[1]);
        let mut out = vec![0f32; raw.len()];
        for i in 0..r {
            for j in 0..c {
                out[i * c + j] = raw[j * r + i];
            }
        }
        out
    };
    let k = rd(kp);
    let d = rd(desc);
    Keypoints { xy: k.chunks_exact(2).map(|c| [c[0], c[1]]).collect(), desc: d }
}

#[test]
#[ignore]
fn lightglue_matches_kornia() {
    let m = models();
    for name in ["pano__self_crop"] {
        let mut z = npyz::npz::NpzArchive::open(refdir().join(format!("match_{name}.npz"))).unwrap();
        let (a, b) = (load_kp(&mut z, "kp1", "desc1"), load_kp(&mut z, "kp2", "desc2"));
        let idxs: Vec<i64> = z.by_name("idxs").unwrap().unwrap().into_vec().unwrap();
        let want: std::collections::HashSet<(i64, i64)> = idxs.chunks_exact(2).map(|c| (c[0], c[1])).collect();
        let t = Instant::now();
        let r = m.lightglue(&a, &b).unwrap();
        // recover index pairs from the matched coordinates
        let find = |k: &Keypoints, p: [f32; 2]| k.xy.iter().position(|q| *q == p).unwrap() as i64;
        let got: std::collections::HashSet<(i64, i64)> = r.pts1.iter().zip(&r.pts2).map(|(p, q)| (find(&a, *p), find(&b, *q))).collect();
        let common = got.intersection(&want).count();
        eprintln!("{name}: kornia {} matches, onnx {} matches, common {common} ({:?})", want.len(), got.len(), t.elapsed());
        assert!(common as f64 >= 0.95 * want.len() as f64, "{name}: only {common}/{} matches agree", want.len());
    }
}

#[test]
#[ignore]
fn disk_matches_kornia() {
    let m = models();
    let idx: Value = serde_json::from_str(&std::fs::read_to_string(refdir().join("prep_index.json")).unwrap()).unwrap();
    for (name, path) in idx.as_object().unwrap() {
        let img = waypoint_core::imgio::open_rgb(std::path::Path::new(path.as_str().unwrap())).unwrap();
        let (chw, w, h) = prep::disk_input(&img).unwrap();
        let t = Instant::now();
        let f = m.disk_features(chw, w, h).unwrap();
        let dt = t.elapsed();
        let mut z = npyz::npz::NpzArchive::open(refdir().join(format!("disk_out_{name}.npz"))).unwrap();
        let kp: Vec<f32> = z.by_name("kp").unwrap().unwrap().into_vec().unwrap();
        let want: std::collections::HashSet<(i32, i32)> = kp.chunks_exact(2).map(|c| (c[0] as i32, c[1] as i32)).collect();
        let got: std::collections::HashSet<(i32, i32)> = f.xy.iter().map(|p| (p[0] as i32, p[1] as i32)).collect();
        let common = got.intersection(&want).count();
        eprintln!("{name:>42}: {} keypoints (kornia {}), common {common} ({:.1}%), {dt:?}", got.len(), want.len(), 100.0 * common as f64 / want.len() as f64);
        assert!(common as f64 >= 0.9 * want.len() as f64);
    }
}

#[test]
#[ignore]
fn end_to_end_pairs_vs_prod() {
    let m = models();
    let report: Value = serde_json::from_str(&std::fs::read_to_string(refdir().join("match_report.json")).unwrap()).unwrap();
    let target = waypoint_core::imgio::open_rgb(&need_image("pano")).unwrap();
    let feats = |img: &image::RgbImage| {
        let (chw, w, h) = prep::disk_input(img).unwrap();
        m.disk_features(chw, w, h).unwrap()
    };
    let ft = feats(&target);
    let man = candidates();
    let mut rows = vec![];
    let start = Instant::now();
    for r in report.as_array().unwrap() {
        let (a, b) = (r["a"].as_str().unwrap(), r["b"].as_str().unwrap());
        if a != "pano" {
            continue;
        }
        let img = if b == "self_crop" {
            let (w, h) = (target.width(), target.height());
            let crop = image::imageops::crop_imm(&target, (w as f64 * 0.1) as u32, (h as f64 * 0.1) as u32, (w as f64 * 0.7) as u32, (h as f64 * 0.8) as u32).to_image();
            image::imageops::resize(&crop, 900, 700, image::imageops::FilterType::CatmullRom)
        } else {
            let p = &man.iter().find(|c| c.0 == b).unwrap().1;
            waypoint_core::imgio::open_rgb(std::path::Path::new(p)).unwrap()
        };
        let fc = feats(&img);
        let pm = m.lightglue(&ft, &fc).unwrap();
        let total = pm.pts1.len();
        let inl = if pm.n_kp1 < 8 || pm.n_kp2 < 8 { 0 } else if total < 8 { 0 } else { ransac::fundamental_ransac_inliers(&pm.pts1, &pm.pts2, 3.0, 0.99, 1000, 0x5EED) };
        let (pi, pt) = (r["prod"]["inliers"].as_u64().unwrap(), r["prod"]["total"].as_u64().unwrap());
        rows.push((b.to_string(), inl as u64, total as u64, pi, pt));
    }
    eprintln!("pairs done in {:?}", start.elapsed());
    eprintln!("{:<48} {:>10} {:>10}", "pair", "onnx inl/tot", "prod inl/tot");
    for r in &rows {
        eprintln!("{:<48} {:>4}/{:<5} {:>4}/{:<5}", r.0, r.1, r.2, r.3, r.4);
    }
    let control: Vec<_> = rows.iter().filter(|r| r.0 == "self_crop").collect();
    for c in control {
        let rel = (c.1 as f64 - c.3 as f64).abs() / c.3 as f64;
        eprintln!("control pair inlier diff: {:.2}%", rel * 100.0);
        assert!(rel < 0.10);
    }
}
