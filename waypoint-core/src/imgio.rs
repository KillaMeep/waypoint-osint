//! Image decoding. JPEGs go through `jpeg-decoder`, whose chroma upsampling and
//! IDCT track libjpeg-turbo (what Pillow uses) more closely than the `image`
//! crate's default decoder (mean abs error 0.07 vs 0.13 grey levels on the test
//! photos); everything else uses the `image` crate. EXIF orientation is
//! ignored on purpose, like Pillow.

use std::io::Cursor;
use std::path::Path;

use image::RgbImage;

use crate::util::{Error, Result};

pub fn decode_rgb(bytes: &[u8]) -> std::result::Result<RgbImage, String> {
    if bytes.len() > 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 {
        if let Ok(img) = decode_jpeg(bytes) {
            return Ok(img);
        }
    }
    let img = image::ImageReader::new(Cursor::new(bytes)).with_guessed_format().map_err(|e| e.to_string())?.decode().map_err(|e| e.to_string())?;
    Ok(img.to_rgb8())
}

fn decode_jpeg(bytes: &[u8]) -> std::result::Result<RgbImage, String> {
    let mut d = jpeg_decoder::Decoder::new(Cursor::new(bytes));
    let px = d.decode().map_err(|e| e.to_string())?;
    let info = d.info().ok_or("no jpeg info")?;
    let (w, h) = (info.width as u32, info.height as u32);
    match info.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 => RgbImage::from_raw(w, h, px).ok_or_else(|| "bad jpeg buffer".to_string()),
        jpeg_decoder::PixelFormat::L8 => {
            let mut out = Vec::with_capacity(px.len() * 3);
            for v in px {
                out.extend_from_slice(&[v, v, v]);
            }
            RgbImage::from_raw(w, h, out).ok_or_else(|| "bad jpeg buffer".to_string())
        }
        _ => Err("unsupported jpeg pixel format".into()),
    }
}

pub fn open_rgb(path: &Path) -> Result<RgbImage> {
    let bytes = std::fs::read(path).map_err(|e| Error::Msg(format!("could not read {}: {e}", path.display())))?;
    decode_rgb(&bytes).map_err(|e| Error::Msg(format!("could not decode {}: {e}", path.display())))
}
