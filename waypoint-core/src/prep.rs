//! Image preprocessing for the ONNX models. Resampling is a port of Pillow's
//! convolution resize, which is what the PyTorch pipelines this replaces use
//! (verified bit-exact on the DISK input, ~1e-5 mean on CLIP pixel values).

use image::RgbImage;

use crate::util::Result;

pub const CLIP_SIZE: u32 = 336;
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];
pub const DISK_MAX_SIDE: u32 = 1024;
pub const DINOV2_SIZE: u32 = 336;
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Pillow's `Image.resize` for RGB8 (Resample.c): separable convolution with
/// a scaled filter support, 22-bit fixed-point coefficients, horizontal pass
/// first, each pass rounded back to 8 bits.
pub fn pil_resize(img: &RgbImage, out_w: u32, out_h: u32, bicubic: bool) -> RgbImage {
    const PRECISION_BITS: i32 = 32 - 8 - 2;
    fn filter(x: f64, bicubic: bool) -> f64 {
        let x = x.abs();
        if bicubic {
            let a = -0.5;
            if x < 1.0 {
                ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
            } else if x < 2.0 {
                (((x - 5.0) * x + 8.0) * x - 4.0) * a
            } else {
                0.0
            }
        } else if x < 1.0 {
            1.0 - x
        } else {
            0.0
        }
    }
    // (bounds start, coefficient list) per output index
    fn coeffs(in_size: u32, out_size: u32, bicubic: bool) -> Vec<(usize, Vec<i32>)> {
        let scale = in_size as f64 / out_size as f64;
        let filterscale = scale.max(1.0);
        let support = (if bicubic { 2.0 } else { 1.0 }) * filterscale;
        let mut all = Vec::with_capacity(out_size as usize);
        for xx in 0..out_size {
            let center = (xx as f64 + 0.5) * scale;
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = (((center + support + 0.5) as i64).min(in_size as i64)) as usize - xmin;
            let mut k: Vec<f64> = (0..xmax).map(|x| filter((x as f64 + xmin as f64 - center + 0.5) / filterscale, bicubic)).collect();
            let ww: f64 = k.iter().sum();
            if ww != 0.0 {
                k.iter_mut().for_each(|v| *v /= ww);
            }
            let ki: Vec<i32> = k.iter().map(|&c| if c < 0.0 { (c * (1 << PRECISION_BITS) as f64 - 0.5) as i32 } else { (c * (1 << PRECISION_BITS) as f64 + 0.5) as i32 }).collect();
            all.push((xmin, ki));
        }
        all
    }
    let clip8 = |v: i64| -> u8 { (v >> PRECISION_BITS).clamp(0, 255) as u8 };
    let (iw, ih) = (img.width() as usize, img.height() as usize);
    let mut cur: Vec<u8> = img.as_raw().clone();
    let (mut cw, mut ch) = (iw, ih);
    if out_w as usize != iw {
        let cf = coeffs(iw as u32, out_w, bicubic);
        let mut out = vec![0u8; out_w as usize * ch * 3];
        for y in 0..ch {
            for (x, (xmin, k)) in cf.iter().enumerate() {
                for c in 0..3 {
                    let mut ss: i64 = 1 << (PRECISION_BITS - 1);
                    for (j, kv) in k.iter().enumerate() {
                        ss += cur[(y * cw + xmin + j) * 3 + c] as i64 * *kv as i64;
                    }
                    out[(y * out_w as usize + x) * 3 + c] = clip8(ss);
                }
            }
        }
        cur = out;
        cw = out_w as usize;
    }
    if out_h as usize != ih {
        let cf = coeffs(ih as u32, out_h, bicubic);
        let mut out = vec![0u8; cw * out_h as usize * 3];
        for (y, (ymin, k)) in cf.iter().enumerate() {
            for x in 0..cw {
                for c in 0..3 {
                    let mut ss: i64 = 1 << (PRECISION_BITS - 1);
                    for (j, kv) in k.iter().enumerate() {
                        ss += cur[((ymin + j) * cw + x) * 3 + c] as i64 * *kv as i64;
                    }
                    out[(y * cw + x) * 3 + c] = clip8(ss);
                }
            }
        }
        cur = out;
        ch = out_h as usize;
    }
    RgbImage::from_raw(cw as u32, ch as u32, cur).unwrap()
}

fn resize_rgb(img: &RgbImage, w: u32, h: u32, bicubic: bool) -> Result<RgbImage> {
    if img.width() == w && img.height() == h {
        return Ok(img.clone());
    }
    Ok(pil_resize(img, w, h, bicubic))
}
/// CLIPImageProcessor for StreetCLIP: shortest side to 336 (bicubic), centre
/// crop 336x336, scale to [0,1], normalise. Returns CHW f32.
pub fn clip_pixel_values(img: &RgbImage) -> Result<Vec<f32>> {
    let (w, h) = (img.width(), img.height());
    let (short, long) = if w <= h { (w, h) } else { (h, w) };
    let new_long = (CLIP_SIZE as u64 * long as u64 / short as u64) as u32; // int() truncation
    let (nw, nh) = if w <= h { (CLIP_SIZE, new_long) } else { (new_long, CLIP_SIZE) };
    let resized = resize_rgb(img, nw, nh, true)?; // PIL BICUBIC (a = -0.5)
    // torchvision center_crop: int(round((dim - crop) / 2.0)), Python's half-even round
    let top = (((nh - CLIP_SIZE) as f64) / 2.0).round_ties_even() as u32;
    let left = (((nw - CLIP_SIZE) as f64) / 2.0).round_ties_even() as u32;
    let s = CLIP_SIZE as usize;
    let mut out = vec![0f32; 3 * s * s];
    let raw = resized.as_raw();
    for y in 0..s {
        for x in 0..s {
            let i = (((top as usize + y) * nw as usize) + left as usize + x) * 3;
            for c in 0..3 {
                out[c * s * s + y * s + x] = (raw[i + c] as f32 / 255.0 - CLIP_MEAN[c]) / CLIP_STD[c];
            }
        }
    }
    Ok(out)
}

/// plonk.pipe.DinoV2FeatureExtractor's augmentation: the largest centred
/// square (plonk CenterCrop(ratio="1:1"), torchvision's centre-crop offsets),
/// Pillow bicubic resize to 336x336, scale to [0,1], ImageNet mean/std. CHW f32.
pub fn dinov2_pixel_values(img: &RgbImage) -> Result<Vec<f32>> {
    let (w, h) = (img.width(), img.height());
    let side = w.min(h);
    // torchvision center_crop: int(round((dim - crop) / 2.0)), Python's half-even round
    let top = (((h - side) as f64) / 2.0).round_ties_even() as u32;
    let left = (((w - side) as f64) / 2.0).round_ties_even() as u32;
    let square = image::imageops::crop_imm(img, left, top, side, side).to_image();
    let resized = resize_rgb(&square, DINOV2_SIZE, DINOV2_SIZE, true)?;
    let s = DINOV2_SIZE as usize;
    let mut out = vec![0f32; 3 * s * s];
    for (i, px) in resized.as_raw().chunks_exact(3).enumerate() {
        for c in 0..3 {
            out[c * s * s + i] = (px[c] as f32 / 255.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
        }
    }
    Ok(out)
}

/// `verify_utils._prep`: cap the long side at 1024 (bilinear), round both
/// sides down to a multiple of 16 (bilinear again), scale to [0,1]. CHW f32.
pub fn disk_input(img: &RgbImage) -> Result<(Vec<f32>, usize, usize)> {
    let (w, h) = (img.width(), img.height());
    let scale = DISK_MAX_SIDE as f64 / w.max(h) as f64;
    let mut cur = img.clone();
    if scale < 1.0 {
        cur = resize_rgb(&cur, (w as f64 * scale) as u32, (h as f64 * scale) as u32, false)?;
    }
    let (w2, h2) = ((cur.width() / 16) * 16, (cur.height() / 16) * 16);
    let cur = resize_rgb(&cur, w2.max(16), h2.max(16), false)?;
    let (w, h) = (cur.width() as usize, cur.height() as usize);
    let raw = cur.as_raw();
    let mut out = vec![0f32; 3 * w * h];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            for c in 0..3 {
                out[c * w * h + y * w + x] = raw[i + c] as f32 / 255.0;
            }
        }
    }
    Ok((out, w, h))
}
