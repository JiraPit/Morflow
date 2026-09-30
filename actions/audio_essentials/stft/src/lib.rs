#![allow(clippy::needless_range_loop)]

use core_types::{
    ActionArgs, DataType, GetShapeFn, Payload, RString, Shape, ShapeResult, Tensor, TensorDType,
};
use rayon::prelude::*;
use std::f32::consts::PI;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Audio
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
}

fn shape_impl(input: Shape, args: ActionArgs) -> ShapeResult {
    use core_types::contract::{self, arg, Error};
    contract::finish((|| {
        if !matches!(input.rank(), 1 | 2) {
            return Err("STFT requires mono or planar audio".into());
        }
        let n = arg::<usize>(&args, &["n_fft"], Some(0), Some(1024))?.unwrap();
        let hop = arg::<usize>(&args, &["hop_size", "hop_length"], None, Some(256))?
            .unwrap()
            .max(1);
        if n < 2 {
            return Err("STFT n_fft must be at least 2".into());
        }
        let fft = n
            .max(16)
            .checked_next_power_of_two()
            .ok_or(Error::from("STFT FFT size overflows"))?;
        let (channels, samples) = if input.rank() == 1 {
            (1, input.dims()[0])
        } else {
            (input.dims()[0], input.dims()[1])
        };
        if samples == 0 {
            return Err(Error::Unknown);
        }
        if samples < n {
            return Ok(input);
        }
        contract::shape([channels, fft / 2 + 1, (samples - n) / hop + 1])
    })())
}

// Compile-time check that get_output_shape matches the core_types ABI.
const _: GetShapeFn = get_output_shape;

#[no_mangle]
pub extern "C" fn get_output_shape(input: Shape, args: ActionArgs) -> ShapeResult {
    shape_impl(input, args)
}

#[derive(Clone, Copy, Default)]
struct Complex {
    re: f32,
    im: f32,
}

impl Complex {
    #[inline]
    fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    #[inline]
    fn mag(self) -> f32 {
        (self.re * self.re + self.im * self.im).sqrt()
    }
}

/// In-place Radix-2 Cooley-Tukey FFT.
fn fft_radix2(buffer: &mut [Complex]) {
    let n = buffer.len();
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation
    let mut j = 0;
    for i in 0..n {
        if i < j {
            buffer.swap(i, j);
        }
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
    }

    // Cooley-Tukey butterfly stages
    let mut len = 2;
    while len <= n {
        let half_len = len / 2;
        let angle = -2.0 * PI / (len as f32);
        let w_step = Complex::new(angle.cos(), angle.sin());

        let mut i = 0;
        while i < n {
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..half_len {
                let u = buffer[i + k];
                let v = buffer[i + k + half_len].mul(w);
                buffer[i + k] = u.add(v);
                buffer[i + k + half_len] = u.sub(v);
                w = w.mul(w_step);
            }
            i += len;
        }
        len <<= 1;
    }
}

fn compute_hann_window(size: usize) -> Vec<f32> {
    if size <= 1 {
        return vec![1.0; size];
    }
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * (i as f32) / ((size - 1) as f32)).cos()))
        .collect()
}

fn next_power_of_two(mut x: usize) -> usize {
    if x <= 1 {
        return 1;
    }
    x -= 1;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x + 1
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    core_types::contract::run(payload, get_output_shape, process_impl)
}

fn process_impl(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();
    let audio = match inner_payload {
        Payload::Audio(a) => a,
        _ => {
            return Payload::Error(RString::from("Action 'stft' requires Payload::Audio"));
        }
    };

    if audio.tensor.rank() == 2 && audio.layout != core_types::AudioLayout::Planar {
        return Payload::Error("STFT requires planar multi-channel audio".into());
    }

    if audio.dtype() != TensorDType::F32 {
        return Payload::Error(RString::from(format!(
            "Action 'stft' requires F32 audio samples, found {:?}",
            audio.dtype()
        )));
    }

    let mut n_fft = 1024usize;
    let mut hop_size = 256usize;

    if let Some(args) = &args_opt {
        if let Some(n) = args
            .get_named("n_fft")
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(v) = n.parse::<usize>() {
                n_fft = v;
            }
        }
        if let Some(h) = args
            .get_named("hop_size")
            .or_else(|| args.get_named("hop_length"))
        {
            if let Ok(v) = h.parse::<usize>() {
                hop_size = v.max(1);
            }
        }
    }

    let fft_size = next_power_of_two(n_fft.max(16));
    let num_bins = fft_size / 2 + 1;
    let window = compute_hann_window(n_fft);

    let num_channels = audio.channels();
    let channel_len = audio.num_samples();
    let tensor = audio.tensor;

    let samples = tensor.to_vec_f32();
    if channel_len < n_fft {
        return Payload::Tensor(tensor);
    }

    let num_frames = (channel_len - n_fft) / hop_size + 1;
    let mut all_spectrograms = Vec::with_capacity(num_channels * num_bins * num_frames);

    for ch in 0..num_channels {
        let ch_samples = &samples[ch * channel_len..(ch + 1) * channel_len];

        // Compute STFT frames in parallel across Rayon workers
        let frames: Vec<Vec<f32>> = (0..num_frames)
            .into_par_iter()
            .map(|f_idx| {
                let start = f_idx * hop_size;
                let mut frame_buf = vec![Complex::default(); fft_size];

                for i in 0..n_fft {
                    if start + i < ch_samples.len() {
                        frame_buf[i] = Complex::new(ch_samples[start + i] * window[i], 0.0);
                    }
                }

                fft_radix2(&mut frame_buf);

                // Take first (N/2 + 1) magnitude bins
                frame_buf[0..num_bins].iter().map(|c| c.mag()).collect()
            })
            .collect();

        // Format into [bins, frames] order for this channel
        for bin_idx in 0..num_bins {
            for frame_idx in 0..num_frames {
                all_spectrograms.push(frames[frame_idx][bin_idx]);
            }
        }
    }

    // Output tensor shape: [channels, freq_bins, time_frames]
    let out_tensor =
        Tensor::from_f32_vec(all_spectrograms, vec![num_channels, num_bins, num_frames])
            .unwrap_or_else(|_| tensor.clone());

    Payload::Tensor(out_tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ActionArgs, Audio, RBox, RString, Tuple2};

    #[test]
    fn test_stft_dimensions() {
        // 1024 samples with n_fft=256, hop_size=128
        // num_frames = (1024 - 256) / 128 + 1 = 7 frames
        // num_bins = 256 / 2 + 1 = 129 bins
        let input_samples: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();
        let audio = Audio::from_f32_planar(&input_samples, 1, 44100).unwrap();

        let mut named = core_types::RVec::new();
        named.push(Tuple2(RString::from("n_fft"), RString::from("256")));
        named.push(Tuple2(RString::from("hop_size"), RString::from("128")));
        let args = ActionArgs {
            positional: core_types::RVec::new(),
            named,
        };

        let payload = Payload::WithArgs {
            payload: RBox::new(Payload::Audio(audio)),
            args,
        };

        let result = process(payload);
        if let Payload::Tensor(out_t) = result {
            assert_eq!(out_t.shape.as_slice(), &[1, 129, 7]);
        } else {
            panic!("Expected Payload::Tensor");
        }
    }
}
