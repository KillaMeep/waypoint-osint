//! Phase 3 parity gates: PLONK sampling through the exported ONNX step graph
//! against the recorded PyTorch outputs (tools/parity/ref_plonk.py for OSV-5M,
//! ref_plonk_variants.py for YFCC and iNaturalist). Ignored by default (needs
//! the runtime DLL, the exported models and tools/parity/ref):
//!
//!   $env:WAYPOINT_ORT_DLL="...\onnxruntime.dll"; $env:WAYPOINT_ONNX_DIR="...\onnx"
//!   $env:WAYPOINT_PLONK="yfcc"   # osv5m (default) | yfcc | inat
//!   cargo test --release --test plonk_parity -- --ignored --nocapture --test-threads 1

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::Value;
use waypoint_core::geo;
use waypoint_core::onnx::{init_runtime, Accel, Encoder as EncoderSession, OnnxModels};
use waypoint_core::plonk::{to_lat_lon, variant, Encoder, PlonkStep, Variant};
use waypoint_core::{imgio, prep};

const IMAGES: [(&str, &str); 2] = [("pano", "test_pano.jpg"), ("photo2", "test_photo2.jpg")];

fn refdir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/parity/ref")
}

fn accel() -> Accel {
    Accel::parse(&std::env::var("WAYPOINT_ACCEL").unwrap_or("dml".into())).unwrap()
}

fn plonk_variant() -> &'static Variant {
    variant(&std::env::var("WAYPOINT_PLONK").unwrap_or("osv5m".into())).expect("WAYPOINT_PLONK: osv5m | yfcc | inat")
}

fn onnx_dir() -> PathBuf {
    PathBuf::from(std::env::var("WAYPOINT_ONNX_DIR").expect("WAYPOINT_ONNX_DIR"))
}

fn plonk() -> PlonkStep {
    init_runtime(Path::new(&std::env::var("WAYPOINT_ORT_DLL").expect("WAYPOINT_ORT_DLL"))).unwrap();
    eprintln!("PLONK {} on {:?}", plonk_variant().label, accel());
    PlonkStep::load(&onnx_dir(), plonk_variant(), accel()).unwrap()
}

/// Reference file for the selected variant (OSV-5M's have no prefix).
fn ref_name(name: &str) -> String {
    match plonk_variant().key {
        "osv5m" => name.to_string(),
        k => format!("{k}_{name}"),
    }
}

fn npy_f32(name: &str) -> Vec<f32> {
    let f = npyz::NpyFile::new(std::io::BufReader::new(std::fs::File::open(refdir().join(ref_name(name))).unwrap())).unwrap();
    f.into_vec::<f32>().unwrap()
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (n(a) * n(b))
}

fn gc_km(a: [f32; 2], b: [f32; 2]) -> f64 {
    let (la1, lo1, la2, lo2) = (
        (a[0] as f64).to_radians(),
        (a[1] as f64).to_radians(),
        (b[0] as f64).to_radians(),
        (b[1] as f64).to_radians(),
    );
    let s = ((la2 - la1) / 2.0).sin().powi(2) + la1.cos() * la2.cos() * ((lo2 - lo1) / 2.0).sin().powi(2);
    6371.0 * 2.0 * s.clamp(0.0, 1.0).sqrt().asin()
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn ref_clusters() -> Value {
    serde_json::from_str(&std::fs::read_to_string(refdir().join(ref_name("plonk_clusters.json"))).unwrap()).unwrap()
}

/// The reference's top clusters must each have a counterpart (the nearest of
/// ours, so near-ties that swap rank still pair up): same count within
/// `count_tol` samples, and centres within `km_tol`, or, for the minor clusters
/// (index > 0), within two standard errors of the reference cluster's mean
/// (spread / sqrt(n)), which is how far the centre moves on its own between
/// PyTorch seeds.
fn assert_clusters(label: &str, got: &[geo::Cluster], noise: f64, want: &Value, count_tol: usize, km_tol: f64) {
    let wc = want["clusters"].as_array().unwrap();
    eprintln!(
        "  {label}: rust {} clusters noise {noise:.3} | python {} clusters noise {:.3}",
        got.len(),
        wc.len(),
        want["noise_frac"].as_f64().unwrap()
    );
    for (i, w) in wc.iter().enumerate().take(3) {
        let wpos = [w["lat"].as_f64().unwrap() as f32, w["lon"].as_f64().unwrap() as f32];
        let (j, g) = got
            .iter()
            .enumerate()
            .min_by(|a, b| gc_km([a.1.lat as f32, a.1.lon as f32], wpos).total_cmp(&gc_km([b.1.lat as f32, b.1.lon as f32], wpos)))
            .expect("no clusters");
        let d = gc_km([g.lat as f32, g.lon as f32], wpos);
        let wn = w["count"].as_u64().unwrap() as usize;
        let lat = w["lat"].as_f64().unwrap();
        let spread_km = ((w["lat_std"].as_f64().unwrap() * 111.2).powi(2)
            + (w["lon_std"].as_f64().unwrap() * 111.2 * lat.to_radians().cos()).powi(2))
        .sqrt();
        let se2 = 2.0 * spread_km / (wn as f64).sqrt();
        let tol = if i == 0 { km_tol } else { km_tol.max(se2) };
        eprintln!(
            "    python #{i} ~ rust #{j}: rust ({:.4}, {:.4}) n={} w={} | python n={} w={} | centre gap {d:.3} km (limit {tol:.2}, 2 SE {se2:.2})",
            g.lat, g.lon, g.count, g.weight, wn, w["weight"]
        );
        assert!(g.count.abs_diff(wn) <= count_tol, "{label} cluster {i}: count {} vs {wn}", g.count);
        assert!(d <= tol, "{label} cluster {i}: centre {d:.3} km apart (limit {tol:.2})");
    }
}

/// DBSCAN labels as in `geo::cluster_samples`, renumbered by cluster size (0 = largest; -1 noise).
fn ranked_labels(coords: &[[f32; 2]]) -> Vec<i32> {
    let labels = geo::dbscan_haversine(coords, 100.0 / 6371.0, 3usize.max((coords.len() as f64 * 0.03) as usize));
    let k = labels.iter().copied().max().map_or(0, |m| (m + 1).max(0)) as usize;
    let mut sizes: Vec<(usize, i32)> = (0..k as i32).map(|l| (labels.iter().filter(|x| **x == l).count(), l)).collect();
    sizes.sort_by(|a, b| b.0.cmp(&a.0));
    let mut rank = vec![0i32; k];
    for (r, (_, l)) in sizes.iter().enumerate() {
        rank[*l as usize] = r as i32;
    }
    labels.iter().map(|l| if *l < 0 { -1 } else { rank[*l as usize] }).collect()
}

/// With identical noise, sample i is the same draw on both sides, so clusters
/// can be compared by membership: where do the samples of each of `a`'s top
/// three clusters end up in `b`? DBSCAN on a broad mode can merge or split two
/// clusters over a handful of bridging samples (PyTorch does so between its
/// own seeds), so up to two destination clusters are accepted, and a few edge
/// samples may cross DBSCAN's density threshold (up to 3% on OSV-5M, 5.5% on
/// a small YFCC cluster). What must not happen is a cluster scattering.
fn assert_same_membership(label: &str, a_name: &str, a: &[i32], b_name: &str, b: &[i32]) {
    for c in 0..3 {
        let members: Vec<usize> = (0..a.len()).filter(|i| a[*i] == c).collect();
        if members.is_empty() {
            break;
        }
        let mut dest: std::collections::BTreeMap<i32, usize> = Default::default();
        for i in &members {
            *dest.entry(b[*i]).or_default() += 1;
        }
        let mut top: Vec<(i32, usize)> = dest.iter().filter(|(l, _)| **l >= 0).map(|(l, n)| (*l, *n)).collect();
        top.sort_by(|x, y| y.1.cmp(&x.1));
        let n = members.len() as f64;
        let clustered = top.iter().map(|t| t.1).sum::<usize>() as f64 / n;
        let in_two = top.iter().take(2).map(|t| t.1).sum::<usize>() as f64 / n;
        let shown: Vec<String> = top.iter().take(3).map(|(l, k)| format!("#{l}: {k}")).collect();
        eprintln!(
            "    {a_name} #{c} (n={}) -> {b_name} {} | noise {} | clustered {:.1}%, in <= 2 clusters {:.1}%",
            members.len(),
            shown.join(", "),
            dest.get(&-1).copied().unwrap_or(0),
            clustered * 100.0,
            in_two * 100.0
        );
        assert!(clustered >= 0.90 && in_two >= 0.90, "{label}: {a_name} cluster {c} does not survive in {b_name}");
    }
}

#[test]
#[ignore]
fn fixed_noise_matches_pytorch() {
    let p = plonk();
    let refs = ref_clusters();
    for (name, _) in IMAGES {
        let emb = npy_f32(&format!("{name}_emb.npy"));
        for seed in 1..=3 {
            let xn = npy_f32(&format!("{name}_xN_s{seed}.npy"));
            let want = npy_f32(&format!("{name}_samples_s{seed}.npy"));
            let t = Instant::now();
            let x = p.integrate(xn, &emb, &mut |_, _| true).unwrap();
            let el = t.elapsed();
            let got = to_lat_lon(&x);
            let mut d: Vec<f64> = got.iter().zip(want.chunks_exact(2)).map(|(g, w)| gc_km(*g, [w[0], w[1]])).collect();
            let (med, p99, max) = (pct(&mut d, 0.5), pct(&mut d, 0.99), pct(&mut d, 1.0));
            eprintln!("{name} s{seed}: {} samples in {el:?}, vs PyTorch: median {med:.4} km, p99 {p99:.3} km, max {max:.3} km", got.len());
            assert!(med < 0.05 && p99 < 1.0, "{name} s{seed}: samples drifted from PyTorch");
            let (cl, noise) = geo::cluster_samples(&got, 100.0, 0.03, 5);
            assert_clusters(&format!("{name} s{seed}"), &cl, noise, &refs[format!("{name}_s{seed}")], 2, 0.5);
        }
    }
}

#[test]
#[ignore]
fn rust_embedding_end_to_end() {
    // Rust decode + resize + the variant's ONNX encoder feeding the ONNX sampler, same noise.
    let p = plonk();
    let v = plonk_variant();
    let embed: Box<dyn Fn(&image::RgbImage) -> Vec<f32>> = match v.encoder {
        Encoder::StreetClip => {
            let m = OnnxModels::load(&onnx_dir(), accel(), Accel::Cpu).unwrap();
            Box::new(move |img| m.embed_pixels(prep::clip_pixel_values(img).unwrap(), 1).unwrap().remove(0))
        }
        Encoder::DinoV2 => {
            let m = EncoderSession::load(&onnx_dir().join(v.encoder.file()), accel()).unwrap();
            Box::new(move |img| m.embed_pixels(prep::dinov2_pixel_values(img).unwrap(), 1).unwrap().remove(0))
        }
    };
    let refs = ref_clusters();
    for (name, path) in IMAGES {
        let img = imgio::open_rgb(Path::new(path)).unwrap();
        let emb = embed(&img);
        let c = cos(&emb, &npy_f32(&format!("{name}_emb.npy")));
        eprintln!("{name}: embedding cosine vs PyTorch {c:.6}");
        assert!(c >= 0.999, "{name}: embedding differs from PyTorch");
        for seed in 1..=3 {
            let xn = npy_f32(&format!("{name}_xN_s{seed}.npy"));
            let got = to_lat_lon(&p.integrate(xn, &emb, &mut |_, _| true).unwrap());
            let want = npy_f32(&format!("{name}_samples_s{seed}.npy"));
            let want: Vec<[f32; 2]> = want.chunks_exact(2).map(|w| [w[0], w[1]]).collect();
            let mut d: Vec<f64> = got.iter().zip(&want).map(|(g, w)| gc_km(*g, *w)).collect();
            let (med, p90) = (pct(&mut d, 0.5), pct(&mut d, 0.9));
            eprintln!("{name} s{seed}: all-Rust vs PyTorch per-sample median {med:.3} km, p90 {p90:.2} km");
            // The embedding differs only by JPEG decoding, so each sample stays
            // near PyTorch's draw from the same noise. How near depends on the
            // model: diffuse outputs (iNaturalist on a street scene, 60% noise)
            // move a few km, so this is a sanity bound; clusters are checked below.
            assert!(med < 5.0, "{name} s{seed}: samples drifted from PyTorch");
            let (cl, noise) = geo::cluster_samples(&got, 100.0, 0.03, 5);
            let want_cl = &refs[format!("{name}_s{seed}")];
            let wc = want_cl["clusters"].as_array().unwrap();
            let fmt = |c: &geo::Cluster| format!("({:.2}, {:.2}) n={}", c.lat, c.lon, c.count);
            eprintln!("    rust:   {} | noise {noise:.3}", cl.iter().take(4).map(fmt).collect::<Vec<_>>().join(", "));
            eprintln!(
                "    python: {} | noise {:.3}",
                wc.iter().take(4).map(|c| format!("({:.2}, {:.2}) n={}", c["lat"].as_f64().unwrap(), c["lon"].as_f64().unwrap(), c["count"])).collect::<Vec<_>>().join(", "),
                want_cl["noise_frac"].as_f64().unwrap()
            );
            let (lr, lp) = (ranked_labels(&got), ranked_labels(&want));
            let label = format!("{name} s{seed}");
            assert_same_membership(&label, "python", &lp, "rust", &lr);
            assert_same_membership(&label, "rust", &lr, "python", &lp);
        }
    }
}

#[test]
#[ignore]
fn own_noise_distribution_and_speed() {
    // Our own Gaussian noise: compare against PyTorch's 3 x 8192 samples for pano.
    let p = plonk();
    let emb = npy_f32("pano_emb.npy");
    let mut all = Vec::new();
    let t = Instant::now();
    for seed in 0..3u64 {
        all.extend(p.sample(&emb, 8192, 1000 + seed, &mut |_, _| true).unwrap());
    }
    let sps = all.len() as f64 / t.elapsed().as_secs_f64();
    let refs = ref_clusters();
    eprintln!(
        "throughput {sps:.0} samples/s at batch 8192 ({:?}) | PyTorch CUDA {:.0} samples/s (same CUDA machine)",
        accel(),
        refs["throughput_8192"].as_f64().unwrap()
    );
    let (cl, noise) = geo::cluster_samples(&all, 100.0, 0.03, 5);
    assert_clusters("pano 3x8192 own noise", &cl, noise, &refs["pano_big"], 600, 10.0);
    // Small batches still work (tail chunk + tiny batch).
    assert_eq!(p.sample(&emb, 5, 1, &mut |_, _| true).unwrap().len(), 5);
}

#[test]
#[ignore]
fn sampling_speed_by_batch() {
    // Throughput per batch size on the selected provider (WAYPOINT_ACCEL=cpu|dml),
    // plus what calibration (plonk::throughput) would measure.
    let p = plonk();
    let emb = npy_f32("pano_emb.npy");
    p.sample(&emb, 64, 0, &mut |_, _| true).unwrap(); // warm-up
    for batch in [256usize, 512, 1024, 2048, 4096] {
        let t = Instant::now();
        p.sample(&emb, batch, 1, &mut |_, _| true).unwrap();
        let s = t.elapsed().as_secs_f64();
        eprintln!("{:?} batch {batch:5}: {s:6.2} s, {:6.0} samples/s", accel(), batch as f64 / s);
    }
    let t = Instant::now();
    let sps = p.throughput().unwrap();
    eprintln!("calibration: {sps:.0} samples/s (took {:.1} s)", t.elapsed().as_secs_f64());
}

#[test]
#[ignore]
fn dinov2_matches_pytorch() {
    // DINOv2 (YFCC / iNaturalist conditioning and retrieval embedding): Rust
    // preprocessing against torchvision's tensors, then embeddings of the two
    // test images and the 21 recorded retrieval candidates against PyTorch.
    init_runtime(Path::new(&std::env::var("WAYPOINT_ORT_DLL").expect("WAYPOINT_ORT_DLL"))).unwrap();
    let m = EncoderSession::load(&onnx_dir().join(Encoder::DinoV2.file()), accel()).unwrap();
    let npy = |f: &str| -> Vec<f32> {
        npyz::NpyFile::new(std::io::BufReader::new(std::fs::File::open(refdir().join(f)).unwrap())).unwrap().into_vec().unwrap()
    };
    for (name, path) in IMAGES {
        let img = imgio::open_rgb(Path::new(path)).unwrap();
        let px = prep::dinov2_pixel_values(&img).unwrap();
        let want_px = npy(&format!("yfcc_{name}_pixel_values.npy"));
        let dmax = px.iter().zip(&want_px).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let dmean = px.iter().zip(&want_px).map(|(a, b)| (a - b).abs() as f64).sum::<f64>() / px.len() as f64;
        // Same preprocessing on Pillow's decode (tools/parity/ref_prep.py): isolates crop + resize from JPEG decoding.
        let rgb_file = || npyz::NpyFile::new(std::io::BufReader::new(std::fs::File::open(refdir().join(format!("rgb_{name}.npy"))).unwrap())).unwrap();
        let shape = rgb_file().shape().to_vec();
        let pil = image::RgbImage::from_raw(shape[1] as u32, shape[0] as u32, rgb_file().into_vec::<u8>().unwrap()).unwrap();
        let pil_px = prep::dinov2_pixel_values(&pil).unwrap();
        let pil_max = pil_px.iter().zip(&want_px).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        eprintln!("{name}: preprocessing on Pillow's decode: max |diff| {pil_max:.6}");
        assert!(pil_max < 1e-4, "DINOv2 crop/resize differs from torchvision");
        let e_ref_px = m.embed_pixels(want_px, 1).unwrap().remove(0);
        let e = m.embed_pixels(px, 1).unwrap().remove(0);
        let want = npy(&format!("yfcc_{name}_emb.npy"));
        eprintln!(
            "{name}: pixels max |diff| {dmax:.4} mean {dmean:.6} | cosine vs PyTorch: ref pixels {:.6}, Rust pixels {:.6}",
            cos(&e_ref_px, &want),
            cos(&e, &want)
        );
        assert!(cos(&e, &want) >= 0.999);
    }
    let mut z = npyz::npz::NpzArchive::open(refdir().join("dinov2_cand_embs.npz")).unwrap();
    let target_ref: Vec<f32> = z.by_name("target").unwrap().unwrap().into_vec().unwrap();
    let target = m.embed_pixels(prep::dinov2_pixel_values(&imgio::open_rgb(Path::new(IMAGES[0].1)).unwrap()).unwrap(), 1).unwrap().remove(0);
    let man: Value = serde_json::from_str(&std::fs::read_to_string(refdir().join("cands/manifest.json")).unwrap()).unwrap();
    let (mut coss, mut ours, mut theirs) = (vec![], vec![], vec![]);
    let t = Instant::now();
    for c in man.as_array().unwrap() {
        let file = c["file"].as_str().unwrap();
        let img = imgio::open_rgb(&refdir().join("cands").join(file)).unwrap();
        let e = m.embed_pixels(prep::dinov2_pixel_values(&img).unwrap(), 1).unwrap().remove(0);
        let r: Vec<f32> = z.by_name(&file.replace('.', "_")).unwrap().unwrap().into_vec().unwrap();
        coss.push(cos(&e, &r));
        ours.push((file.to_string(), cos(&target, &e)));
        theirs.push((file.to_string(), cos(&target_ref, &r)));
    }
    let n = coss.len();
    let min = coss.iter().cloned().fold(1.0, f64::min);
    eprintln!("{n} candidates in {:?}: cosine vs PyTorch min {min:.6} mean {:.6}", t.elapsed(), coss.iter().sum::<f64>() / n as f64);
    let rank = |v: &mut Vec<(String, f64)>| {
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v.iter().map(|x| x.0.clone()).collect::<Vec<_>>()
    };
    let (ro, rt) = (rank(&mut ours), rank(&mut theirs));
    eprintln!("retrieval: top-1 same {}, top-5 overlap {}/5", ro[0] == rt[0], ro[..5].iter().filter(|f| rt[..5].contains(f)).count());
    assert!(min >= 0.999, "candidate embedding cosine {min}");
}
