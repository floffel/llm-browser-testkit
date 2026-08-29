//! Vision support — screenshot capture, downscaling, and JPEG encoding for
//! LLM visual assertions.
//!
//! When an `assert` step sets `screenshot = true`, the runner captures the
//! current viewport as PNG, downscales it so its longest edge is at most
//! [`default_max_dimension`] (the page's `[config]
//! screenshot_max_dimension` wins when set), and encodes it as a JPEG data
//! URL. The image is sent to the vision endpoint as an OpenAI-compatible
//! `image_url` content part alongside the text prompt.
//!
//! Downscaling happens in Rust (no page JS, no fragile `canvas` evaluation):
//! the PNG bytes are decoded with the `image` crate, resized with Lanczos
//! filtering, and re-encoded as quality-85 JPEG.

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use headless_chrome::Tab;

/// Default longest edge (px) of screenshots sent to vision endpoints.
pub const DEFAULT_MAX_DIMENSION: u32 = 1400;
/// JPEG quality used for the encoded screenshot.
const JPEG_QUALITY: u8 = 85;

/// Captures the current viewport and returns a JPEG data URL suitable for
/// the OpenAI-compatible `image_url` content part.
///
/// The image is downscaled so its longest edge is at most
/// `max_dimension` (no upscaling; `0` disables resizing).
///
/// # Errors
///
/// Returns a description when the CDP screenshot capture, PNG decode,
/// resize, or JPEG encode fails.
pub fn capture_screenshot_data_url(tab: &Tab, max_dimension: u32) -> Result<String, String> {
    let png = tab
        .capture_screenshot(
            headless_chrome::protocol::cdp::Page::CaptureScreenshotFormatOption::Png,
            None,
            None,
            true,
        )
        .map_err(|e| format!("screenshot capture failed: {e}"))?;

    let img =
        image::load_from_memory(&png).map_err(|e| format!("screenshot decode failed: {e}"))?;

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
    use super::DEFAULT_MAX_DIMENSION;

    #[test]
    fn default_max_dimension_sane() {
        // 1280x720 viewports stay untouched; 1920-wide screens are scaled
        // down to keep vision tokens reasonable.
        const {
            assert!(DEFAULT_MAX_DIMENSION >= 1280 && DEFAULT_MAX_DIMENSION <= 1600);
        }
    }
}
