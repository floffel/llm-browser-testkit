//! Vision support — screenshot capture, tiling, downscaling, and JPEG
//! encoding for LLM visual assertions.
//!
//! When an `assert` step sets `screenshot = true`, the runner captures the
//! **full scrollable page** (via the CDP `captureBeyondViewport` flag —
//! nothing below the fold is skipped) and splits it into viewport-tall
//! bands ("tiles") from the top, covering at most
//! `scenario::ScenarioConfig::screenshot_max_height` (default `"20x"` =
//! four viewports). Each tile is downscaled independently so its longest
//! edge is at most [`default_max_dimension`] (the page's `[config]
//! screenshot_max_dimension` wins when set), JPEG-encoded, and sent as its
//! own OpenAI-compatible `image_url` content part next to the text prompt —
//! detail is preserved at every depth, and the tile count (hence token
//! cost) is bounded by the height cap and [`max_tiles`].
//!
//! Downscaling and cropping happen in Rust (no page JS, no fragile
//! `canvas` evaluation): the PNG bytes are decoded with the `image` crate,
//! banded, resized with Lanczos filtering, and re-encoded as quality-85
//! JPEG in a single capture call.

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use headless_chrome::protocol::cdp::Page::CaptureScreenshot;
use headless_chrome::protocol::cdp::Page::CaptureScreenshotFormatOption;
use headless_chrome::Tab;
use image::DynamicImage;

/// Default longest edge (px) of screenshots sent to vision endpoints.
pub const DEFAULT_MAX_DIMENSION: u32 = 1400;
/// Hard ceiling on the number of tiles attached to a single vision assert,
/// so a huge `screenshot_max_height` can never produce an unbounded
/// request (or token bill).
pub const MAX_TILES: u32 = 30;
/// JPEG quality used for the encoded screenshot.
const JPEG_QUALITY: u8 = 85;

/// Captures the full page and returns one JPEG data URL per viewport-tall
/// band, ordered from the top of the page down — suitable for the
/// OpenAI-compatible `image_url` content-part array.
///
/// The capture is covered from the top to `height_cap` pixels (pages
/// shorter than the cap or the viewport are returned as a single tile).
/// Each band is downscaled independently so its longest edge is at most
/// `max_dimension` (no upscaling; `0` disables resizing). The number of
/// tiles is bounded by [`max_tiles`].
///
/// # Errors
///
/// Returns a description when the CDP screenshot capture, PNG decode,
/// crop, resize, or JPEG encode fails.
pub fn capture_screenshot_data_urls(
    tab: &Tab,
    max_dimension: u32,
    height_cap: u32,
    band_height: u32,
) -> Result<Vec<String>, String> {
    let png = tab
        .call_method(CaptureScreenshot {
            format: Some(CaptureScreenshotFormatOption::Png),
            quality: None,
            clip: None,
            from_surface: Some(true),
            capture_beyond_viewport: Some(true),
            optimize_for_speed: None,
        })
        .map_err(|e| format!("screenshot capture failed: {e}"))?
        .data;
    let png_bytes = STANDARD
        .decode(png)
        .map_err(|e| format!("screenshot base64 decode failed: {e}"))?;

    let img = image::load_from_memory(&png_bytes)
        .map_err(|e| format!("screenshot decode failed: {e}"))?;

    // Coverage from the top: the height cap bounds token cost, tile bands
    // keep 1:1 detail for everything that is covered.
    let coverage = img.height().min(height_cap);
    let count = tile_count(coverage, band_height);
    let mut tiles: Vec<String> = Vec::new();
    for i in 0..count {
        let top = i * band_height;
        let band_h = (coverage - top).min(band_height);
        let band = img.crop_imm(0, top, img.width(), band_h);
        tiles.push(encode_jpeg(band, max_dimension)?);
    }
    Ok(tiles)
}

/// Number of viewport-tall bands covering `coverage` pixels, clamped to
/// [`max_tiles`] so a single assert can never send an unbounded number of
/// image parts.
#[must_use]
pub fn tile_count(coverage: u32, band_height: u32) -> u32 {
    if coverage == 0 {
        return 0;
    }
    if band_height == 0 {
        return 1;
    }
    coverage.div_ceil(band_height).min(MAX_TILES)
}

/// Downscales (optional) and JPEG-encodes a single band into a data URL.
fn encode_jpeg(img: DynamicImage, max_dimension: u32) -> Result<String, String> {
    let (width, height) = (img.width(), img.height());
    let longest = width.max(height);
    let resized = if max_dimension > 0 && longest > max_dimension {
        let scale = f64::from(max_dimension) / f64::from(longest);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let new_width = (f64::from(width) * scale).round().max(1.0) as u32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let new_height = (f64::from(height) * scale).round().max(1.0) as u32;
        img.resize(new_width, new_height, image::imageops::FilterType::Lanczos3)
    } else {
        img
    };

    let mut jpeg = Vec::new();
    {
        let mut encoder =
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, JPEG_QUALITY);
        encoder
            .encode_image(&resized)
            .map_err(|e| format!("screenshot encode failed: {e}"))?;
    }

    Ok(format!("data:image/jpeg;base64,{}", STANDARD.encode(jpeg)))
}

#[cfg(test)]
mod tests {
    use super::{tile_count, DEFAULT_MAX_DIMENSION, MAX_TILES};

    #[test]
    fn default_max_dimension_sane() {
        // 1280x720 viewports stay untouched; 1920-wide screens are scaled
        // down to keep vision tokens reasonable.
        const {
            assert!(DEFAULT_MAX_DIMENSION >= 1280 && DEFAULT_MAX_DIMENSION <= 1600);
        }
    }

    #[test]
    fn tile_count_bands_and_clamps() {
        assert_eq!(tile_count(0, 720), 0);
        assert_eq!(tile_count(720, 720), 1);
        assert_eq!(tile_count(1440, 720), 2);
        assert_eq!(tile_count(1500, 720), 3);
        assert_eq!(tile_count(2880, 720), 4);
        assert_eq!(tile_count(720, 0), 1);
        assert_eq!(tile_count(MAX_TILES * 720 + 1, 720), MAX_TILES);
    }
}
