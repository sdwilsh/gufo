//! PNG and WebP decoding for `gufo::core::DecodeImage`.
//!
//! Output matches Pillow's `convert("RGB")`. Decoding drops alpha without
//! compositing and ignores gamma and ICC metadata. 16-bit samples keep their
//! high byte, except in 16-bit grayscale PNG, where values above 255 saturate.
//! The caller applies EXIF orientation.

use std::io::Cursor;
use std::panic::{AssertUnwindSafe, catch_unwind};

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GufoImageStatus {
    Ok = 0,
    Invalid = 1,
    TooLarge = 2,
    Internal = 3,
}

/// Packed RGB8 owned by Rust; release it with `gufo_image_free`.
#[repr(C)]
pub struct GufoDecodedImage {
    pub width: u32,
    pub height: u32,
    pub pixels: *mut u8,
    pub len: usize,
    pub capacity: usize,
}

struct Rgb {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

type Result<T> = std::result::Result<T, GufoImageStatus>;

/// # Safety
/// `bytes` must point to `len` readable bytes and `out` to writable storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_image_decode_png(
    bytes: *const u8,
    len: usize,
    max_pixels: u64,
    out: *mut GufoDecodedImage,
) -> GufoImageStatus {
    unsafe { export(bytes, len, max_pixels, out, decode_png) }
}

/// # Safety
/// `bytes` must point to `len` readable bytes and `out` to writable storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_image_decode_webp(
    bytes: *const u8,
    len: usize,
    max_pixels: u64,
    out: *mut GufoDecodedImage,
) -> GufoImageStatus {
    unsafe { export(bytes, len, max_pixels, out, decode_webp) }
}

/// # Safety
/// `image` must come from a successful decode and not be freed before.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gufo_image_free(image: *mut GufoDecodedImage) {
    let Some(image) = (unsafe { image.as_mut() }) else {
        return;
    };
    if !image.pixels.is_null() {
        drop(unsafe { Vec::from_raw_parts(image.pixels, image.len, image.capacity) });
        image.pixels = std::ptr::null_mut();
    }
}

unsafe fn export(
    bytes: *const u8,
    len: usize,
    max_pixels: u64,
    out: *mut GufoDecodedImage,
    decode: fn(&[u8], u64) -> Result<Rgb>,
) -> GufoImageStatus {
    if bytes.is_null() || out.is_null() {
        return GufoImageStatus::Internal;
    }
    let input = unsafe { std::slice::from_raw_parts(bytes, len) };
    // A decoder panic comes from the input, so it rejects the image like any
    // other malformed file instead of unwinding into C++.
    let decoded = catch_unwind(AssertUnwindSafe(|| decode(input, max_pixels)))
        .unwrap_or(Err(GufoImageStatus::Invalid));
    match decoded {
        Ok(image) => {
            let mut pixels = std::mem::ManuallyDrop::new(image.pixels);
            unsafe {
                out.write(GufoDecodedImage {
                    width: image.width,
                    height: image.height,
                    pixels: pixels.as_mut_ptr(),
                    len: pixels.len(),
                    capacity: pixels.capacity(),
                });
            }
            GufoImageStatus::Ok
        }
        Err(status) => status,
    }
}

/// Fails instead of aborting the process when memory is exhausted.
fn zeroed(len: usize) -> Result<Vec<u8>> {
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(len)
        .map_err(|_| GufoImageStatus::Internal)?;
    buffer.resize(len, 0);
    Ok(buffer)
}

fn validate_dimensions(width: u32, height: u32, max_pixels: u64) -> Result<()> {
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > max_pixels {
        return Err(GufoImageStatus::TooLarge);
    }
    Ok(())
}

fn decode_png(bytes: &[u8], max_pixels: u64) -> Result<Rgb> {
    use png::{BitDepth, ColorType, Transformations};
    let invalid = |_| GufoImageStatus::Invalid;
    // The dimension check runs before any pixel allocation. The byte limit
    // covers the 16-bit RGBA worst case plus metadata chunks.
    let pixel_bytes = usize::try_from(max_pixels.saturating_mul(8)).unwrap_or(usize::MAX);
    let mut decoder = png::Decoder::new_with_limits(
        Cursor::new(bytes),
        png::Limits {
            bytes: pixel_bytes.saturating_add(64 << 20),
        },
    );
    decoder.set_transformations(Transformations::EXPAND);
    let mut reader = decoder.read_info().map_err(invalid)?;
    let info = reader.info();
    let (width, height) = (info.width, info.height);
    validate_dimensions(width, height, max_pixels)?;
    // EXPAND turns a tRNS chunk into alpha, so check the stored color type.
    let gray16 = info.color_type == ColorType::Grayscale && info.bit_depth == BitDepth::Sixteen;
    let (color, depth) = reader.output_color_type();
    let size = reader
        .output_buffer_size()
        .ok_or(GufoImageStatus::TooLarge)?;
    let mut raw = zeroed(size)?;
    reader.next_frame(&mut raw).map_err(invalid)?;
    reader.finish().map_err(invalid)?;

    let sample_bytes = if depth == BitDepth::Sixteen { 2 } else { 1 };
    let stride = color.samples() * sample_bytes;
    let mut pixels = zeroed(width as usize * height as usize * 3)?;
    for (src, dst) in raw.chunks_exact(stride).zip(pixels.as_chunks_mut::<3>().0) {
        // The first byte of a big-endian 16-bit sample is its high byte.
        let sample = |channel: usize| src[channel * sample_bytes];
        match color {
            ColorType::Grayscale | ColorType::GrayscaleAlpha if gray16 => {
                dst.fill(u16::from_be_bytes([src[0], src[1]]).min(255) as u8);
            }
            ColorType::Grayscale | ColorType::GrayscaleAlpha => dst.fill(sample(0)),
            ColorType::Rgb | ColorType::Rgba => {
                *dst = [sample(0), sample(1), sample(2)];
            }
            ColorType::Indexed => return Err(GufoImageStatus::Internal),
        }
    }
    Ok(Rgb {
        width,
        height,
        pixels,
    })
}

fn decode_webp(bytes: &[u8], max_pixels: u64) -> Result<Rgb> {
    let invalid = |_| GufoImageStatus::Invalid;
    let mut decoder = image_webp::WebPDecoder::new(Cursor::new(bytes)).map_err(invalid)?;
    let (width, height) = decoder.dimensions();
    validate_dimensions(width, height, max_pixels)?;
    // Gufo accepts still WebP images only.
    if decoder.is_animated() {
        return Err(GufoImageStatus::Invalid);
    }
    let size = decoder
        .output_buffer_size()
        .ok_or(GufoImageStatus::TooLarge)?;
    let mut raw = zeroed(size)?;
    decoder.read_image(&mut raw).map_err(invalid)?;
    if !decoder.has_alpha() {
        return Ok(Rgb {
            width,
            height,
            pixels: raw,
        });
    }
    let mut pixels = zeroed(width as usize * height as usize * 3)?;
    for (rgba, rgb) in raw
        .as_chunks::<4>()
        .0
        .iter()
        .zip(pixels.as_chunks_mut::<3>().0)
    {
        *rgb = [rgba[0], rgba[1], rgba[2]];
    }
    Ok(Rgb {
        width,
        height,
        pixels,
    })
}
