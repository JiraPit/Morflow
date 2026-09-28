use crate::tensor::{Tensor, TensorDType};
use abi_stable::std_types::{RString, RVec};
use abi_stable::StableAbi;

/// Supported color space representations for image payloads.
#[repr(u8)]
#[derive(StableAbi, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    Rgb = 0,
    Rgba = 1,
    Bgr = 2,
    Bgra = 3,
    Grayscale = 4,
    GrayscaleAlpha = 5,
    Ycbcr = 6,
    Hsv = 7,
    Lab = 8,
    Cmyk = 9,
}

impl ColorSpace {
    /// Number of color channels expected for this color space format.
    #[inline]
    pub const fn channels(&self) -> usize {
        match self {
            ColorSpace::Grayscale => 1,
            ColorSpace::GrayscaleAlpha => 2,
            ColorSpace::Rgb
            | ColorSpace::Bgr
            | ColorSpace::Ycbcr
            | ColorSpace::Hsv
            | ColorSpace::Lab => 3,
            ColorSpace::Rgba | ColorSpace::Bgra | ColorSpace::Cmyk => 4,
        }
    }
}

/// Memory layout of image dimensions.
#[repr(u8)]
#[derive(StableAbi, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageLayout {
    /// Height x Width x Channels (Interleaved / Packed, standard in OpenCV, PIL, and image codecs)
    Hwc = 0,
    /// Channels x Height x Width (Planar, standard in ML inference / PyTorch / ONNX)
    Chw = 1,
}

/// A high-performance, zero-copy, multi-channel image representation backed by `Tensor`.
///
/// Supports arbitrary color spaces, bit depths (`U8`, `U16`, `F32`, etc.), memory layouts
/// (HWC / CHW), zero-copy ROI cropping, strided views, and Rayon parallelization across FFI.
#[repr(C)]
#[derive(StableAbi, Debug, Clone)]
pub struct Image {
    /// Underlying multidimensional tensor containing pixel sample buffers.
    pub tensor: Tensor,
    /// Color space format of the image (RGB, RGBA, BGR, Grayscale, etc.).
    pub color_space: ColorSpace,
    /// Dimension memory layout (HWC or CHW).
    pub layout: ImageLayout,
}

unsafe impl Send for Image {}
unsafe impl Sync for Image {}

impl Image {
    /// Creates a new `Image` wrapping a `Tensor` with validation.
    pub fn new(
        tensor: Tensor,
        color_space: ColorSpace,
        layout: ImageLayout,
    ) -> Result<Self, RString> {
        let shape = tensor.shape.as_slice();
        let expected_channels = color_space.channels();

        match (shape.len(), layout) {
            (2, _) => {
                if expected_channels != 1 {
                    return Err(RString::from(format!(
                        "2D tensor shape {:?} requires single-channel color space (Grayscale), got {:?}",
                        shape, color_space
                    )));
                }
            }
            (3, ImageLayout::Hwc) => {
                let c = shape[2];
                if c != expected_channels {
                    return Err(RString::from(format!(
                        "HWC shape {:?} has {} channels, but color space {:?} requires {}",
                        shape, c, color_space, expected_channels
                    )));
                }
            }
            (3, ImageLayout::Chw) => {
                let c = shape[0];
                if c != expected_channels {
                    return Err(RString::from(format!(
                        "CHW shape {:?} has {} channels, but color space {:?} requires {}",
                        shape, c, color_space, expected_channels
                    )));
                }
            }
            _ => {
                return Err(RString::from(format!(
                    "Invalid tensor rank {} for Image (expected 2 or 3 dimensions, got shape {:?})",
                    shape.len(),
                    shape
                )));
            }
        }

        Ok(Self {
            tensor,
            color_space,
            layout,
        })
    }

    /// Creates an 8-bit unsigned integer (U8) image in standard HWC (Height x Width x Channels) layout.
    pub fn from_u8_hwc(
        data: &[u8],
        width: usize,
        height: usize,
        color_space: ColorSpace,
    ) -> Result<Self, RString> {
        let channels = color_space.channels();
        let shape = if channels == 1 {
            vec![height, width]
        } else {
            vec![height, width, channels]
        };

        let tensor = Tensor::from_rvec_u8(RVec::from(data.to_vec()), shape, TensorDType::U8)?;
        Ok(Self {
            tensor,
            color_space,
            layout: ImageLayout::Hwc,
        })
    }

    /// Creates a 32-bit floating point (F32) image in standard HWC layout.
    pub fn from_f32_hwc(
        data: &[f32],
        width: usize,
        height: usize,
        color_space: ColorSpace,
    ) -> Result<Self, RString> {
        let channels = color_space.channels();
        let shape = if channels == 1 {
            vec![height, width]
        } else {
            vec![height, width, channels]
        };

        let tensor = Tensor::from_f32_shape(data, shape)?;
        Ok(Self {
            tensor,
            color_space,
            layout: ImageLayout::Hwc,
        })
    }

    /// Creates an 8-bit unsigned integer (U8) image in planar CHW (Channels x Height x Width) layout.
    pub fn from_u8_chw(
        data: &[u8],
        width: usize,
        height: usize,
        color_space: ColorSpace,
    ) -> Result<Self, RString> {
        let channels = color_space.channels();
        let shape = if channels == 1 {
            vec![height, width]
        } else {
            vec![channels, height, width]
        };

        let tensor = Tensor::from_rvec_u8(RVec::from(data.to_vec()), shape, TensorDType::U8)?;
        Ok(Self {
            tensor,
            color_space,
            layout: ImageLayout::Chw,
        })
    }

    /// Creates a 32-bit floating point (F32) image in planar CHW layout.
    pub fn from_f32_chw(
        data: &[f32],
        width: usize,
        height: usize,
        color_space: ColorSpace,
    ) -> Result<Self, RString> {
        let channels = color_space.channels();
        let shape = if channels == 1 {
            vec![height, width]
        } else {
            vec![channels, height, width]
        };

        let tensor = Tensor::from_f32_shape(data, shape)?;
        Ok(Self {
            tensor,
            color_space,
            layout: ImageLayout::Chw,
        })
    }

    /// Returns the pixel width of the image.
    #[inline]
    pub fn width(&self) -> usize {
        let shape = self.tensor.shape.as_slice();
        match (shape.len(), self.layout) {
            (2, _) => shape[1],
            (3, ImageLayout::Hwc) => shape[1],
            (3, ImageLayout::Chw) => shape[2],
            _ => 0,
        }
    }

    /// Returns the pixel height of the image.
    #[inline]
    pub fn height(&self) -> usize {
        let shape = self.tensor.shape.as_slice();
        match (shape.len(), self.layout) {
            (2, _) => shape[0],
            (3, ImageLayout::Hwc) => shape[0],
            (3, ImageLayout::Chw) => shape[1],
            _ => 0,
        }
    }

    /// Returns the number of color channels.
    #[inline]
    pub fn channels(&self) -> usize {
        let shape = self.tensor.shape.as_slice();
        match (shape.len(), self.layout) {
            (2, _) => 1,
            (3, ImageLayout::Hwc) => shape[2],
            (3, ImageLayout::Chw) => shape[0],
            _ => 0,
        }
    }

    /// Returns the primitive data type of the pixel samples.
    #[inline]
    pub fn dtype(&self) -> TensorDType {
        self.tensor.dtype
    }

    /// Checks if the image memory layout is contiguous in memory.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.tensor.is_contiguous()
    }

    /// Access contiguous byte slice if this image view is contiguous.
    #[inline]
    pub fn as_bytes(&self) -> Option<&[u8]> {
        self.tensor.as_bytes()
    }

    /// Converts non-contiguous or cropped image view into an owned contiguous byte buffer.
    #[inline]
    pub fn to_contiguous_bytes(&self) -> RVec<u8> {
        self.tensor.to_contiguous_bytes()
    }

    /// Access contiguous U8 byte slice if contiguous and dtype is U8.
    #[inline]
    pub fn as_u8_slice(&self) -> Option<&[u8]> {
        if self.tensor.dtype == TensorDType::U8 {
            self.tensor.as_bytes()
        } else {
            None
        }
    }

    /// Access contiguous F32 slice if contiguous and dtype is F32.
    #[inline]
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        self.tensor.as_f32_slice()
    }

    /// Performs an **O(1) zero-copy crop** Region of Interest (ROI) slicing along Y and X dimensions.
    pub fn crop(
        &self,
        y_start: usize,
        y_end: usize,
        x_start: usize,
        x_end: usize,
    ) -> Result<Self, RString> {
        let shape = self.tensor.shape.as_slice();
        let (y_axis, x_axis) = match (shape.len(), self.layout) {
            (2, _) => (0, 1),
            (3, ImageLayout::Hwc) => (0, 1),
            (3, ImageLayout::Chw) => (1, 2),
            _ => {
                return Err(RString::from(format!(
                    "Cannot crop image with unsupported shape {:?}",
                    shape
                )));
            }
        };

        let y_sliced = self.tensor.slice_range(y_axis, y_start, y_end, 1)?;
        let xy_sliced = y_sliced.slice_range(x_axis, x_start, x_end, 1)?;

        Ok(Self {
            tensor: xy_sliced,
            color_space: self.color_space,
            layout: self.layout,
        })
    }

    /// Re-interprets image color space without modifying raw pixels if channel counts match.
    pub fn with_color_space(mut self, new_color_space: ColorSpace) -> Result<Self, RString> {
        if new_color_space.channels() != self.channels() {
            return Err(RString::from(format!(
                "Cannot change color space to {:?} (requires {} channels, image has {})",
                new_color_space,
                new_color_space.channels(),
                self.channels()
            )));
        }
        self.color_space = new_color_space;
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_u8_hwc_creation_and_properties() {
        let width = 4;
        let height = 3;
        let data = vec![128u8; width * height * 3];

        let img = Image::from_u8_hwc(&data, width, height, ColorSpace::Rgb).unwrap();
        assert_eq!(img.width(), 4);
        assert_eq!(img.height(), 3);
        assert_eq!(img.channels(), 3);
        assert_eq!(img.dtype(), TensorDType::U8);
        assert_eq!(img.layout, ImageLayout::Hwc);
        assert!(img.is_contiguous());
        assert_eq!(img.as_u8_slice().unwrap().len(), 36);
    }

    #[test]
    fn test_image_f32_chw_creation() {
        let width = 16;
        let height = 8;
        let data = vec![0.5f32; 4 * width * height];

        let img = Image::from_f32_chw(&data, width, height, ColorSpace::Rgba).unwrap();
        assert_eq!(img.width(), 16);
        assert_eq!(img.height(), 8);
        assert_eq!(img.channels(), 4);
        assert_eq!(img.dtype(), TensorDType::F32);
        assert_eq!(img.layout, ImageLayout::Chw);
        assert_eq!(img.as_f32_slice().unwrap().len(), 4 * 16 * 8);
    }

    #[test]
    fn test_zero_copy_roi_crop() {
        let width = 100;
        let height = 100;
        let data = vec![255u8; width * height * 3];

        let img = Image::from_u8_hwc(&data, width, height, ColorSpace::Rgb).unwrap();
        let cropped = img.crop(10, 30, 20, 70).unwrap();

        assert_eq!(cropped.height(), 20);
        assert_eq!(cropped.width(), 50);
        assert_eq!(cropped.channels(), 3);
        assert_eq!(cropped.color_space, ColorSpace::Rgb);
        // Zero-copy cropped view has 20 * 50 * 3 = 3000 elements
        assert_eq!(cropped.tensor.num_elements(), 3000);
    }
}
