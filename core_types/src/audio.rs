use crate::tensor::{Tensor, TensorDType};
use abi_stable::std_types::{RString, RVec};
use abi_stable::StableAbi;

/// Standard speaker and channel configurations for multi-channel audio.
#[repr(u8)]
#[derive(StableAbi, Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioChannelLayout {
    Mono = 0,
    Stereo = 1,
    Stereo2_1 = 2,
    Quad = 3,
    Surround5_1 = 4,
    Surround7_1 = 5,
    AmbisonicOrder1 = 6,
    AmbisonicOrder2 = 7,
    AmbisonicOrder3 = 8,
    Custom = 9,
}

impl AudioChannelLayout {
    /// Number of audio channels defined by this layout.
    #[inline]
    pub const fn channels(&self) -> usize {
        match self {
            AudioChannelLayout::Mono => 1,
            AudioChannelLayout::Stereo => 2,
            AudioChannelLayout::Stereo2_1 => 3,
            AudioChannelLayout::Quad => 4,
            AudioChannelLayout::Surround5_1 => 6,
            AudioChannelLayout::Surround7_1 => 8,
            AudioChannelLayout::AmbisonicOrder1 => 4,
            AudioChannelLayout::AmbisonicOrder2 => 9,
            AudioChannelLayout::AmbisonicOrder3 => 16,
            AudioChannelLayout::Custom => 0,
        }
    }

    /// Automatically resolves standard channel layout from channel count.
    pub fn from_channel_count(count: usize) -> Self {
        match count {
            1 => AudioChannelLayout::Mono,
            2 => AudioChannelLayout::Stereo,
            3 => AudioChannelLayout::Stereo2_1,
            4 => AudioChannelLayout::Quad,
            6 => AudioChannelLayout::Surround5_1,
            8 => AudioChannelLayout::Surround7_1,
            9 => AudioChannelLayout::AmbisonicOrder2,
            16 => AudioChannelLayout::AmbisonicOrder3,
            _ => AudioChannelLayout::Custom,
        }
    }
}

/// Memory layout of multi-channel audio samples.
#[repr(u8)]
#[derive(StableAbi, Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioLayout {
    /// Planar / Non-interleaved: [Channels, Samples] (Industry standard for DSP pipelines, DAWs, and Rayon SIMD)
    Planar = 0,
    /// Interleaved / Packed: [Samples, Channels] (Standard for hardware audio streams: ALSA, CoreAudio, WASAPI)
    Interleaved = 1,
}

/// A high-performance, zero-copy multi-channel audio stream backed by `Tensor`.
///
/// Stores audio samples with sample rate, channel layout, memory layout (Planar or Interleaved),
/// and supports zero-copy channel extraction, time-domain slicing, and Rayon parallelization across FFI.
#[repr(C)]
#[derive(StableAbi, Debug, Clone)]
pub struct Audio {
    /// Underlying multidimensional tensor containing sample buffers.
    pub tensor: Tensor,
    /// Sampling rate in Hertz (e.g. 44100, 48000, 96000).
    pub sample_rate: u32,
    /// Speaker channel configuration layout.
    pub channel_layout: AudioChannelLayout,
    /// Memory buffer layout (Planar or Interleaved).
    pub layout: AudioLayout,
}

unsafe impl Send for Audio {}
unsafe impl Sync for Audio {}

impl Audio {
    /// Creates a new `Audio` payload wrapping a `Tensor` with validation.
    pub fn new(
        tensor: Tensor,
        sample_rate: u32,
        channel_layout: AudioChannelLayout,
        layout: AudioLayout,
    ) -> Result<Self, RString> {
        if sample_rate == 0 {
            return Err(RString::from("Audio sample rate cannot be 0"));
        }

        let shape = tensor.shape.as_slice();
        let expected_channels = channel_layout.channels();

        match (shape.len(), layout) {
            (1, _) => {
                // 1D tensor is always Mono (1 channel)
                if expected_channels != 0 && expected_channels != 1 {
                    return Err(RString::from(format!(
                        "1D tensor shape {:?} is mono, but channel layout {:?} expects {} channels",
                        shape, channel_layout, expected_channels
                    )));
                }
            }
            (2, AudioLayout::Planar) => {
                let ch = shape[0];
                if expected_channels != 0 && ch != expected_channels {
                    return Err(RString::from(format!(
                        "Planar shape {:?} has {} channels, but channel layout {:?} expects {}",
                        shape, ch, channel_layout, expected_channels
                    )));
                }
            }
            (2, AudioLayout::Interleaved) => {
                let ch = shape[1];
                if expected_channels != 0 && ch != expected_channels {
                    return Err(RString::from(format!(
                        "Interleaved shape {:?} has {} channels, but channel layout {:?} expects {}",
                        shape, ch, channel_layout, expected_channels
                    )));
                }
            }
            _ => {
                return Err(RString::from(format!(
                    "Invalid tensor rank {} for Audio (expected 1 or 2 dimensions, got shape {:?})",
                    shape.len(),
                    shape
                )));
            }
        }

        Ok(Self {
            tensor,
            sample_rate,
            channel_layout,
            layout,
        })
    }

    /// Creates a 32-bit floating point (F32) audio stream in standard Planar layout [channels, samples].
    pub fn from_f32_planar(
        data: &[f32],
        channels: usize,
        sample_rate: u32,
    ) -> Result<Self, RString> {
        if channels == 0 {
            return Err(RString::from("Channel count cannot be 0"));
        }
        if data.len() % channels != 0 {
            return Err(RString::from(format!(
                "Data length {} is not divisible by channel count {}",
                data.len(),
                channels
            )));
        }

        let samples_per_channel = data.len() / channels;
        let shape = if channels == 1 {
            vec![samples_per_channel]
        } else {
            vec![channels, samples_per_channel]
        };

        let tensor = Tensor::from_f32_shape(data, shape)?;
        let channel_layout = AudioChannelLayout::from_channel_count(channels);

        Ok(Self {
            tensor,
            sample_rate,
            channel_layout,
            layout: AudioLayout::Planar,
        })
    }

    /// Creates a 32-bit floating point (F32) audio stream in Interleaved layout [samples, channels].
    pub fn from_f32_interleaved(
        data: &[f32],
        channels: usize,
        sample_rate: u32,
    ) -> Result<Self, RString> {
        if channels == 0 {
            return Err(RString::from("Channel count cannot be 0"));
        }
        if data.len() % channels != 0 {
            return Err(RString::from(format!(
                "Data length {} is not divisible by channel count {}",
                data.len(),
                channels
            )));
        }

        let samples_per_channel = data.len() / channels;
        let shape = vec![samples_per_channel, channels];

        let tensor = Tensor::from_f32_shape(data, shape)?;
        let channel_layout = AudioChannelLayout::from_channel_count(channels);

        Ok(Self {
            tensor,
            sample_rate,
            channel_layout,
            layout: AudioLayout::Interleaved,
        })
    }

    /// Returns the number of audio channels.
    #[inline]
    pub fn channels(&self) -> usize {
        let shape = self.tensor.shape.as_slice();
        match (shape.len(), self.layout) {
            (1, _) => 1,
            (2, AudioLayout::Planar) => shape[0],
            (2, AudioLayout::Interleaved) => shape[1],
            _ => 0,
        }
    }

    /// Returns the number of audio samples per channel.
    #[inline]
    pub fn num_samples(&self) -> usize {
        let shape = self.tensor.shape.as_slice();
        match (shape.len(), self.layout) {
            (1, _) => shape[0],
            (2, AudioLayout::Planar) => shape[1],
            (2, AudioLayout::Interleaved) => shape[0],
            _ => 0,
        }
    }

    /// Returns the total duration of the audio in seconds.
    #[inline]
    pub fn duration_seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.num_samples() as f64 / self.sample_rate as f64
        }
    }

    /// Returns the primitive data type of the audio samples.
    #[inline]
    pub fn dtype(&self) -> TensorDType {
        self.tensor.dtype
    }

    /// Checks if the audio memory layout is contiguous in memory.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.tensor.is_contiguous()
    }

    /// Access contiguous byte slice if this audio view is contiguous.
    #[inline]
    pub fn as_bytes(&self) -> Option<&[u8]> {
        self.tensor.as_bytes()
    }

    /// Converts non-contiguous or cropped audio view into an owned contiguous byte buffer.
    #[inline]
    pub fn to_contiguous_bytes(&self) -> RVec<u8> {
        self.tensor.to_contiguous_bytes()
    }

    /// Access contiguous F32 slice if contiguous and dtype is F32.
    #[inline]
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        self.tensor.as_f32_slice()
    }

    /// Converts non-contiguous or contiguous audio view into an owned `Vec<f32>`.
    #[inline]
    pub fn to_vec_f32(&self) -> Vec<f32> {
        self.tensor.to_vec_f32()
    }

    /// Performs an **O(1) zero-copy sample range slice** along the time dimension [start_sample, end_sample).
    pub fn slice_samples(&self, start_sample: usize, end_sample: usize) -> Result<Self, RString> {
        let shape = self.tensor.shape.as_slice();
        let time_axis = match (shape.len(), self.layout) {
            (1, _) => 0,
            (2, AudioLayout::Planar) => 1,
            (2, AudioLayout::Interleaved) => 0,
            _ => {
                return Err(RString::from(format!(
                    "Cannot slice audio with unsupported shape {:?}",
                    shape
                )));
            }
        };

        let sliced_tensor = self
            .tensor
            .slice_range(time_axis, start_sample, end_sample, 1)?;
        Ok(Self {
            tensor: sliced_tensor,
            sample_rate: self.sample_rate,
            channel_layout: self.channel_layout,
            layout: self.layout,
        })
    }

    /// Performs an **O(1) zero-copy duration slice** in seconds [start_sec, end_sec).
    pub fn slice_time(&self, start_sec: f64, end_sec: f64) -> Result<Self, RString> {
        let sr = self.sample_rate as f64;
        let start_sample = (start_sec.max(0.0) * sr).round() as usize;
        let end_sample = (end_sec.max(start_sec) * sr).round() as usize;
        self.slice_samples(start_sample, end_sample)
    }

    /// Extracts an individual channel as a mono `Audio` stream with **O(1) zero-copy**.
    pub fn channel(&self, ch_idx: usize) -> Result<Self, RString> {
        if self.layout != AudioLayout::Planar {
            return Err(RString::from(
                "Direct channel extraction is only supported on Planar audio layout",
            ));
        }

        let shape = self.tensor.shape.as_slice();
        if shape.len() == 1 {
            if ch_idx == 0 {
                return Ok(self.clone());
            } else {
                return Err(RString::from(format!(
                    "Channel index {} out of bounds for mono audio",
                    ch_idx
                )));
            }
        }

        let ch_tensor = self.tensor.slice_axis_index(0, ch_idx)?;
        Ok(Self {
            tensor: ch_tensor,
            sample_rate: self.sample_rate,
            channel_layout: AudioChannelLayout::Mono,
            layout: AudioLayout::Planar,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_f32_planar_creation_and_properties() {
        let sample_rate = 48000;
        let channels = 2;
        let num_samples = 48000; // 1 second of stereo audio
        let data = vec![0.0f32; channels * num_samples];

        let audio = Audio::from_f32_planar(&data, channels, sample_rate).unwrap();
        assert_eq!(audio.channels(), 2);
        assert_eq!(audio.num_samples(), 48000);
        assert_eq!(audio.sample_rate, 48000);
        assert_eq!(audio.duration_seconds(), 1.0);
        assert_eq!(audio.channel_layout, AudioChannelLayout::Stereo);
        assert_eq!(audio.layout, AudioLayout::Planar);
        assert!(audio.is_contiguous());
    }

    #[test]
    fn test_audio_zero_copy_time_slicing() {
        let sample_rate = 44100;
        let channels = 1;
        let num_samples = 44100 * 5; // 5 seconds
        let data: Vec<f32> = (0..num_samples).map(|x| x as f32).collect();

        let audio = Audio::from_f32_planar(&data, channels, sample_rate).unwrap();
        // Slice from second 1.0 to 3.0 (2 seconds = 88200 samples)
        let sliced = audio.slice_time(1.0, 3.0).unwrap();

        assert_eq!(sliced.duration_seconds(), 2.0);
        assert_eq!(sliced.num_samples(), 88200);
        assert_eq!(sliced.tensor.num_elements(), 88200);
    }

    #[test]
    fn test_audio_zero_copy_channel_extraction() {
        let sample_rate = 48000;
        let channels = 2;
        let num_samples = 1000;
        let mut data = vec![1.0f32; num_samples]; // Left channel: 1.0
        data.extend(vec![2.0f32; num_samples]); // Right channel: 2.0

        let audio = Audio::from_f32_planar(&data, channels, sample_rate).unwrap();
        let right_ch = audio.channel(1).unwrap();

        assert_eq!(right_ch.channels(), 1);
        assert_eq!(right_ch.channel_layout, AudioChannelLayout::Mono);
        assert_eq!(right_ch.num_samples(), 1000);
        assert_eq!(right_ch.as_f32_slice().unwrap()[0], 2.0);
    }
}
