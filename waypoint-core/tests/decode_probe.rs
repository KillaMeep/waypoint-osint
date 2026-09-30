//! How closely do Rust JPEG decoders track PIL/libjpeg-turbo? (needs tools/parity/ref)

use std::path::PathBuf;

fn refdir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/parity/ref")
}

fn npy_u8(name: &str) -> Option<(Vec<u64>, Vec<u8>)> {
    let p = refdir().join(name);
    if !p.exists() {
        return None;
    }
    let f = npyz::NpyFile::new(std::io::BufReader::new(std::fs::File::open(p).unwrap())).unwrap();
    let shape = f.shape().to_vec();
    Some((shape, f.into_vec::<u8>().unwrap()))
}

fn diff(a: &[u8], b: &[u8]) -> (f64, u8, f64) {
    let mut sum = 0u64;
    let mut max = 0u8;
    let mut over1 = 0u64;
    for (x, y) in a.iter().zip(b) {
        let d = x.abs_diff(*y);
        sum += d as u64;
        max = max.max(d);
        if d > 1 {
            over1 += 1;
        }
    }
    (sum as f64 / a.len() as f64, max, over1 as f64 / a.len() as f64)
}

/// The test photos live outside the repo: set WAYPOINT_TEST_IMAGES to the folder holding test_pano.jpg.
fn test_image(name: &str) -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var_os("WAYPOINT_TEST_IMAGES")?).join(format!("test_{name}.jpg"));
    p.is_file().then_some(p)
}

#[test]
fn decode_vs_pil() {
    for name in ["pano"] {
        let Some(path) = test_image(name) else { eprintln!("skipped: set WAYPOINT_TEST_IMAGES"); return };
        let Some((shape, want)) = npy_u8(&format!("rgb_{name}.npy")) else { eprintln!("skipped (no ref)"); return };
        let img = image::open(path).unwrap().to_rgb8();
        assert_eq!(shape, vec![img.height() as u64, img.width() as u64, 3]);
        let (mean, max, frac) = diff(img.as_raw(), &want);
        eprintln!("{name}: image-crate(zune-jpeg) vs PIL: mean abs diff {mean:.4}, max {max}, frac >1: {frac:.5}");
    }
}

#[test]
fn decode_jpeg_decoder_vs_pil() {
    for name in ["pano"] {
        let Some(path) = test_image(name) else { return };
        let Some((_shape, want)) = npy_u8(&format!("rgb_{name}.npy")) else { return };
        let f = std::fs::File::open(path).unwrap();
        let mut d = jpeg_decoder::Decoder::new(std::io::BufReader::new(f));
        let px = d.decode().unwrap();
        let (mean, max, frac) = diff(&px, &want);
        eprintln!("{name}: jpeg-decoder vs PIL: mean abs diff {mean:.4}, max {max}, frac >1: {frac:.5}");
    }
}
