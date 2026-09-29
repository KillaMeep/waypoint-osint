//! Preprocessing parity against the tensors PyTorch actually consumed
//! (tools/parity/ref_prep.py). Skipped when the reference data is absent.

use std::path::PathBuf;

use image::RgbImage;
use waypoint_core::prep;

fn refdir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/parity/ref")
}

fn npy<T: npyz::Deserialize>(name: &str) -> Option<(Vec<u64>, Vec<T>)> {
    let p = refdir().join(name);
    if !p.exists() {
        return None;
    }
    let f = npyz::NpyFile::new(std::io::BufReader::new(std::fs::File::open(p).unwrap())).unwrap();
    Some((f.shape().to_vec(), f.into_vec::<T>().unwrap()))
}

fn stats(a: &[f32], b: &[f32]) -> (f64, f64) {
    let mut max = 0f64;
    let mut sum = 0f64;
    for (x, y) in a.iter().zip(b) {
        let d = (*x as f64 - *y as f64).abs();
        max = max.max(d);
        sum += d;
    }
    (max, sum / a.len() as f64)
}

fn names() -> Vec<String> {
    let idx = refdir().join("prep_index.json");
    if !idx.exists() {
        return vec![];
    }
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(idx).unwrap()).unwrap();
    v.as_object().unwrap().keys().cloned().collect()
}

#[test]
fn preprocessing_matches_pytorch_inputs() {
    let idx: serde_json::Value = match std::fs::read_to_string(refdir().join("prep_index.json")) {
        Ok(s) => serde_json::from_str(&s).unwrap(),
        Err(_) => {
            eprintln!("skipped: no reference data");
            return;
        }
    };
    let _ = names();
    for (name, path) in idx.as_object().unwrap() {
        let (shape, rgb) = npy::<u8>(&format!("rgb_{name}.npy")).unwrap();
        let pil_img = RgbImage::from_raw(shape[1] as u32, shape[0] as u32, rgb).unwrap();
        let rust_img = waypoint_core::imgio::open_rgb(std::path::Path::new(path.as_str().unwrap())).unwrap();
        let (_, want_pv) = npy::<f32>(&format!("pv_{name}.npy")).unwrap();
        let (_, want_disk) = npy::<f32>(&format!("disk_in_{name}.npy")).unwrap();
        for (label, img) in [("PIL pixels", &pil_img), ("Rust decode", &rust_img)] {
            let pv = prep::clip_pixel_values(img).unwrap();
            let (dmax, dmean) = stats(&pv, &want_pv);
            let (d, w, h) = prep::disk_input(img).unwrap();
            assert_eq!(d.len(), want_disk.len(), "{name}: disk tensor size ({w}x{h})");
            let (kmax, kmean) = stats(&d, &want_disk);
            eprintln!("{name:>40} [{label:>11}] clip pixel_values: max {dmax:.4} mean {dmean:.5} | disk input: max {kmax:.4} mean {kmean:.6}");
            if label == "PIL pixels" {
                assert!(dmean < 0.02, "{name}: CLIP preprocessing mean diff {dmean}");
                assert!(kmean < 0.002, "{name}: DISK preprocessing mean diff {kmean}");
            }
        }
    }
}
