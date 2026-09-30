pub mod check;
pub mod cli;
pub mod engine;
pub mod outputs;
pub mod registry;
pub mod resolver;
pub mod scheduler;
pub mod types;
pub mod validator;

pub use core_types;
pub use core_types::{Audio, ColorSpace, DataType, Image, Payload, RVec, Tensor};
pub use engine::{Morflow, MorflowError, MorflowPipeline};
pub use outputs::PipelineOutputs;
pub use registry::{ActionRegistry, LoadedAction};
pub use resolver::ActionResolver;
pub use scheduler::AutoParallelScheduler;
pub use types::{coerce_host_payload, default_payload, ptype_of, shape_spec_of};

#[cfg(test)]
mod tests {
    use super::*;
    use abi_stable::std_types::RVec;
    use core_types::{
        Audio, AudioChannelLayout, AudioLayout, ColorSpace, Image, ImageLayout, Payload, Tensor,
        TensorDType,
    };

    #[test]
    fn test_morflow_pipeline_execution_with_emit() {
        let morf_src = r#"
            accept RawBytes input_data

            $input_data >> identity >> emit
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to parse pipeline");
        assert_eq!(pipeline.params().len(), 1);
        assert_eq!(pipeline.params()[0].name, "input_data");

        let payload = Payload::Data {
            buffer: RVec::from(vec![1, 2, 3, 4, 5]),
        };

        let outputs = pipeline.run(payload).expect("Pipeline run failed");
        let result = outputs.into_single().expect("Expected single output");
        if let Payload::Data { buffer } = result {
            assert_eq!(buffer.as_slice(), &[1, 2, 3, 4, 5]);
        } else {
            panic!("Expected Data payload");
        }
    }

    #[test]
    fn test_morflow_multiple_emit_outputs() {
        let morf_src = r#"
            accept Tensor audio

            $audio[0:4] >> identity >> emit("low_band")
            $audio[4:8] >> identity >> emit("high_band")
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to parse pipeline");
        let tensor = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let outputs = pipeline
            .run(Payload::Tensor(tensor))
            .expect("Pipeline run failed");

        assert_eq!(outputs.len(), 2);
        assert!(outputs.contains_key("low_band"));
        assert!(outputs.contains_key("high_band"));

        if let Payload::Tensor(low) = &outputs["low_band"] {
            assert_eq!(low.shape.as_slice(), &[4]);
            assert_eq!(low.num_elements(), 4);
        } else {
            panic!("Expected Tensor for low_band");
        }

        if let Payload::Tensor(high) = outputs.get("high_band").unwrap() {
            assert_eq!(high.shape.as_slice(), &[4]);
            assert_eq!(high.num_elements(), 4);
        } else {
            panic!("Expected Tensor for high_band");
        }
    }

    #[test]
    fn test_morflow_tensor_slice_pipeline() {
        let morf_src = r#"
            accept Tensor tensor_in

            $tensor_in[1:3] >> identity >> emit
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to parse pipeline");

        let data: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let tensor = Tensor::from_f32_shape(&data, vec![4, 8]).unwrap();

        let outputs = pipeline
            .run(Payload::Tensor(tensor))
            .expect("Pipeline execution failed");
        let result = outputs.into_single().unwrap();
        if let Payload::Tensor(out_t) = result {
            assert_eq!(out_t.shape.as_slice(), &[2, 8]);
            assert_eq!(out_t.num_elements(), 16);
        } else {
            panic!("Expected Tensor payload");
        }
    }

    #[test]
    fn test_morflow_auto_parallel_each_loop() {
        let morf_src = r#"
            accept Tensor tensor_in

            $tensor_in >> each ($ch) {
                $ch >> identity
            } >> emit
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to parse pipeline");

        // 8 independent channels x 128 samples (parallelized across CPU cores)
        let data: Vec<f32> = (0..1024).map(|x| x as f32).collect();
        let tensor = Tensor::from_f32_shape(&data, vec![8, 128]).unwrap();

        let outputs = pipeline
            .run(Payload::Tensor(tensor))
            .expect("Pipeline execution failed");
        let result = outputs.into_single().unwrap();
        if let Payload::Tensor(out_t) = result {
            assert_eq!(out_t.shape.as_slice(), &[8, 128]);
            assert_eq!(out_t.num_elements(), 1024);
        } else {
            panic!("Expected Tensor payload");
        }
    }

    #[test]
    fn test_morflow_branching_with_tensor_metric() {
        let morf_src = r#"
            accept Tensor audio

            $audio >> if ($audio.peak > 5.0) {
                identity
            } else {
                identity
            } >> emit
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to parse pipeline");

        let tensor = Tensor::from_f32_slice(&[1.0, 10.0, -3.0]); // peak = 10.0 > 5.0
        let outputs = pipeline
            .run(Payload::Tensor(tensor))
            .expect("Pipeline execution failed");
        let result = outputs.into_single().unwrap();

        if let Payload::Tensor(out_t) = result {
            assert_eq!(out_t.peak_abs(), 10.0);
        } else {
            panic!("Expected Tensor payload");
        }
    }

    #[test]
    fn test_morflow_multi_statement_with_taps() {
        let morf_src = r#"
            accept RawBytes source

            $source >> identity >> $saved1
            $saved1 >> identity >> $saved2
            $saved2 >> identity >> emit
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to parse pipeline");
        let payload = Payload::Data {
            buffer: RVec::from(vec![42]),
        };
        let outputs = pipeline.run(payload).expect("Pipeline run failed");
        let result = outputs.into_single().unwrap();

        if let Payload::Data { buffer } = result {
            assert_eq!(buffer.as_slice(), &[42]);
        } else {
            panic!("Expected Data payload");
        }
    }

    #[test]
    fn test_morflow_load_from_file() {
        let pipeline = Morflow::load("../examples/rust/audio_split/audio_split_pipeline.morf")
            .or_else(|_| Morflow::load("examples/rust/audio_split/audio_split_pipeline.morf"))
            .expect("Failed to load audio_split_pipeline.morf");

        assert_eq!(pipeline.params().len(), 1);
    }

    #[test]
    fn test_morflow_independent_flows_parallelism() {
        let morf_src = r#"
            accept Tensor audio

            # Two independent branches computed from $audio
            $audio[0:4] >> identity >> $low_freq
            $audio[4:8] >> identity >> $high_freq

            # Final recombination
            $low_freq >> identity >> emit
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to parse pipeline");
        let tensor = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let outputs = pipeline
            .run(Payload::Tensor(tensor))
            .expect("Pipeline run failed");
        let res = outputs.into_single().unwrap();

        if let Payload::Tensor(out_t) = res {
            assert_eq!(out_t.num_elements(), 4);
            assert_eq!(out_t.shape.as_slice(), &[4]);
        } else {
            panic!("Expected Tensor payload");
        }
    }

    #[test]
    fn test_morflow_each_with_external_variable() {
        let morf_src = r#"
            accept Tensor tensor_in

            # 1. Define an external variable outside the loop
            $tensor_in[0:2] >> identity >> $external_filter

            # 2. Inside each ($ch), use both $ch and $external_filter
            $tensor_in >> each ($ch) {
                $ch >> identity(filter=$external_filter)
            } >> emit
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to parse pipeline");
        let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let tensor = Tensor::from_f32_shape(&data, vec![4, 4]).unwrap();

        let outputs = pipeline
            .run(Payload::Tensor(tensor))
            .expect("Pipeline run failed");
        let res = outputs.into_single().unwrap();
        if let Payload::Tensor(out_t) = res {
            assert_eq!(out_t.shape.as_slice(), &[4, 4]);
            assert_eq!(out_t.num_elements(), 16);
        } else {
            panic!("Expected Tensor payload");
        }
    }

    #[test]
    fn test_compile_error_on_multiple_unnamed_emits() {
        let invalid_morf = r#"
            accept Tensor audio

            $audio[0:4] >> identity >> emit
            $audio[4:8] >> identity >> emit
        "#;

        let res = Morflow::from_str(invalid_morf);
        assert!(res.is_err());
        if let Err(MorflowError::Compile(msg)) = res {
            assert!(
                msg.contains("Multiple flows emit outputs, but one or more emit calls are unnamed"),
                "Unexpected message: {}",
                msg
            );
        } else {
            panic!("Expected MorflowError::Compile error");
        }
    }

    #[test]
    fn test_compile_error_on_duplicate_emit_names() {
        let invalid_morf = r#"
            accept Tensor audio

            $audio[0:4] >> identity >> emit("track")
            $audio[4:8] >> identity >> emit("track")
        "#;

        let res = Morflow::from_str(invalid_morf);
        assert!(res.is_err());
        if let Err(MorflowError::Compile(msg)) = res {
            assert!(
                msg.contains("Duplicate emit name 'track'"),
                "Unexpected message: {}",
                msg
            );
        } else {
            panic!("Expected MorflowError::Compile error");
        }
    }

    #[test]
    fn test_mid_stream_emit_allowed() {
        let morf_src = r#"
            accept Tensor audio

            $audio >> emit("intermediate") >> identity >> emit("final")
        "#;

        let res = Morflow::from_str(morf_src);
        assert!(
            res.is_ok(),
            "Mid-stream emit should be valid and compile successfully: {:?}",
            res.err()
        );
    }

    #[test]
    fn test_compile_error_on_emit_inside_loop() {
        let invalid_morf = r#"
            accept Tensor tensor_in

            $tensor_in >> each ($ch) {
                $ch >> emit
            } >> emit
        "#;

        let res = Morflow::from_str(invalid_morf);
        assert!(res.is_err());
        if let Err(MorflowError::Compile(msg)) = res {
            assert!(
                msg.contains("'emit' cannot be called inside a nested sub-flow"),
                "Unexpected message: {}",
                msg
            );
        } else {
            panic!("Expected MorflowError::Compile error");
        }
    }

    #[test]
    fn test_compile_error_on_named_emit_inside_each_loop() {
        let invalid_morf = r#"
            accept Tensor tensor_in

            $tensor_in >> each ($ch) {
                $ch >> identity >> emit("channel_out")
            } >> emit
        "#;

        let res = Morflow::from_str(invalid_morf);
        assert!(res.is_err(), "Must reject named emit inside each block");
        if let Err(MorflowError::Compile(msg)) = res {
            assert!(
                msg.contains("'emit' cannot be called inside a nested sub-flow"),
                "Unexpected message: {}",
                msg
            );
        } else {
            panic!("Expected MorflowError::Compile error");
        }
    }

    #[test]
    fn test_compile_error_on_emit_inside_if_branch() {
        let invalid_morf = r#"
            accept Tensor audio

            $audio >> if ($audio.peak > 1.0) {
                identity >> emit("branch_out")
            } >> emit
        "#;

        let res = Morflow::from_str(invalid_morf);
        assert!(res.is_err(), "Must reject emit inside if branch");
        if let Err(MorflowError::Compile(msg)) = res {
            assert!(
                msg.contains("'emit' cannot be called inside a nested sub-flow"),
                "Unexpected message: {}",
                msg
            );
        } else {
            panic!("Expected MorflowError::Compile error");
        }
    }

    #[test]
    fn test_each_loop_tapped_and_emitted_in_subsequent_flow() {
        let morf_src = r#"
            accept Tensor tensor_in

            $tensor_in >> each ($ch) {
                $ch >> identity
            } >> $processed_tensor

            $processed_tensor >> identity >> emit
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to compile pipeline");
        let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let tensor = Tensor::from_f32_shape(&data, vec![4, 4]).unwrap();

        let outputs = pipeline
            .run(Payload::Tensor(tensor))
            .expect("Pipeline execution failed");
        let result = outputs.into_single().unwrap();
        if let Payload::Tensor(out_t) = result {
            assert_eq!(out_t.shape.as_slice(), &[4, 4]);
            assert_eq!(out_t.num_elements(), 16);
        } else {
            panic!("Expected Tensor payload");
        }
    }

    #[test]
    fn test_compile_error_on_top_level_variable_reassignment() {
        let invalid_morf = r#"
            accept RawBytes source

            $source >> action_a >> $duplicate_var
            $source >> action_b >> $duplicate_var
        "#;

        let res = Morflow::from_str(invalid_morf);
        assert!(
            res.is_err(),
            "Should have failed static compile-time validation"
        );
        if let Err(MorflowError::Compile(msg)) = res {
            assert!(
                msg.contains("$duplicate_var is written to by multiple flows: action_a, action_b"),
                "Unexpected message: {}",
                msg
            );
        } else {
            panic!("Expected MorflowError::Compile error");
        }
    }

    #[test]
    fn test_compile_error_on_inner_loop_writing_to_external_variable() {
        let invalid_morf = r#"
            accept Tensor tensor_in

            $tensor_in[0:2] >> calibrate_noise >> $outer_var

            $tensor_in >> each ($ch) {
                $ch >> denoise >> $outer_var
            } >> emit
        "#;

        let res = Morflow::from_str(invalid_morf);
        assert!(
            res.is_err(),
            "Should have failed static compile-time validation"
        );
        if let Err(MorflowError::Compile(msg)) = res {
            assert!(
                msg.contains("$outer_var is written to by multiple flows: calibrate_noise, each ($ch) sub-flow (denoise)"),
                "Unexpected message: {}",
                msg
            );
        } else {
            panic!("Expected MorflowError::Compile error");
        }
    }

    #[test]
    fn test_error_on_legacy_pipeline_wrapper() {
        let legacy_pipeline_morf = r#"
            pipeline "FirstPipeline" ($a) {
                $a >> identity >> emit
            }
        "#;

        let res = Morflow::from_str(legacy_pipeline_morf);
        assert!(
            res.is_err(),
            "Must reject legacy explicit pipeline wrapper syntax"
        );
    }

    #[test]
    fn test_nested_each_loop_with_branching() {
        let morf_src = r#"
            accept Tensor tensor_in

            $tensor_in >> each ($ch) {
                $ch >> if ($ch.peak > 2.0) {
                    identity
                } else {
                    identity
                }
            } >> emit
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to parse pipeline");
        let data: Vec<f32> = vec![
            1.0, 1.0, // peak = 1.0 <= 2.0 (else branch)
            5.0, 3.0, // peak = 5.0 > 2.0 (then branch)
        ];
        let tensor = Tensor::from_f32_shape(&data, vec![2, 2]).unwrap();

        let outputs = pipeline
            .run(Payload::Tensor(tensor))
            .expect("Pipeline run failed");
        let out = outputs.into_single().unwrap();
        if let Payload::Tensor(t) = out {
            assert_eq!(t.shape.as_slice(), &[2, 2]);
            assert_eq!(t.num_elements(), 4);
        } else {
            panic!("Expected Tensor output");
        }
    }

    #[test]
    fn test_pipeline_no_emit_error() {
        let morf_src = r#"
            accept RawBytes source

            $source >> identity >> $tapped
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to parse pipeline");
        let payload = Payload::Data {
            buffer: RVec::from(vec![1, 2, 3]),
        };
        let outputs = pipeline.run(payload).expect("Pipeline run should succeed");
        assert!(outputs.is_empty());
        let res = outputs.into_single();
        assert!(res.is_err());
    }

    #[test]
    fn test_dsp_pipeline_end_to_end() {
        let morf_src = r#"
            accept Audio audio_in

            $audio_in >> gain(db=+6.0) >> biquad_filter(type="lowpass", freq=5000.0) >> limiter(ceiling_db=-1.0) >> normalize(target_peak=0.9) >> emit
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to compile DSP pipeline");
        let input_samples: Vec<f32> = (0..200).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        let audio = Audio::from_f32_planar(&input_samples, 1, 44100).unwrap();

        let outputs = pipeline
            .run(Payload::Audio(audio))
            .expect("Failed to execute DSP pipeline");
        let result = outputs.into_single().expect("Expected single output");
        if let Payload::Audio(out_aud) = result {
            assert_eq!(out_aud.num_samples(), 200);
            let peak = out_aud.tensor.peak_abs();
            assert!(
                (peak - 0.9).abs() < 1e-3,
                "Expected normalized peak near 0.9, got {}",
                peak
            );
        } else {
            panic!("Expected Audio output");
        }
    }

    #[test]
    fn test_multichannel_spatial_dsp_pipeline() {
        let morf_src = r#"
            accept Audio stereo_in

            $stereo_in >> stereo_widen(width=1.5) >> delay(time_ms=10.0, feedback=0.2, mix=0.3) >> compressor(threshold_db=-10.0, ratio=3.0) >> emit
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to compile stereo pipeline");
        let left: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let right: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).cos()).collect();
        let mut combined = left;
        combined.extend(right);
        let audio = Audio::from_f32_planar(&combined, 2, 44100).unwrap();

        let outputs = pipeline
            .run(Payload::Audio(audio))
            .expect("Failed to run stereo pipeline");
        let result = outputs.into_single().expect("Expected single output");
        if let Payload::Audio(out_aud) = result {
            assert_eq!(out_aud.tensor.shape.as_slice(), &[2, 100]);
            assert!(out_aud.tensor.peak_abs() > 0.0);
        } else {
            panic!("Expected Audio output");
        }
    }

    #[test]
    fn test_spectral_stft_and_resample_pipeline() {
        let morf_src = r#"
            accept Audio audio_in

            $audio_in >> resample(from_rate=48000.0, to_rate=44100.0) >> stft(n_fft=256, hop_size=128) >> emit
        "#;

        let mut pipeline =
            Morflow::from_str(morf_src).expect("Failed to compile spectral pipeline");
        let input_samples: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.05).sin()).collect();
        let audio = Audio::from_f32_planar(&input_samples, 1, 48000).unwrap();

        let outputs = pipeline
            .run(Payload::Audio(audio))
            .expect("Failed to run spectral pipeline");
        let result = outputs.into_single().expect("Expected single output");
        if let Payload::Tensor(out_t) = result {
            // Rank 3: [channels, freq_bins, time_frames]
            assert_eq!(out_t.rank(), 3);
            assert_eq!(out_t.shape[0], 1);
            assert_eq!(out_t.shape[1], 129); // 256/2 + 1
            assert!(out_t.shape[2] > 0);
        } else {
            panic!("Expected 3D Spectrogram Tensor output");
        }
    }

    #[test]
    fn test_image_payload_pipeline_execution() {
        let morf_src = r#"
            accept Image img_in

            $img_in >> if ($img_in.width > 30) {
                identity
            } else {
                identity
            } >> emit
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to compile image pipeline");
        let width = 64;
        let height = 48;
        let data = vec![200u8; width * height * 3];
        let img = Image::from_u8_hwc(&data, width, height, ColorSpace::Rgb).unwrap();

        let outputs = pipeline
            .run(Payload::Image(img))
            .expect("Failed to execute image pipeline");
        let result = outputs.into_single().expect("Expected single output");
        if let Payload::Image(out_img) = result {
            assert_eq!(out_img.width(), 64);
            assert_eq!(out_img.height(), 48);
            assert_eq!(out_img.channels(), 3);
            assert_eq!(out_img.color_space, ColorSpace::Rgb);
            assert_eq!(out_img.layout, ImageLayout::Hwc);
        } else {
            panic!("Expected Image payload output");
        }
    }

    #[test]
    fn test_audio_payload_pipeline_execution() {
        let morf_src = r#"
            accept Audio audio_in

            $audio_in >> if ($audio_in.sample_rate >= 44100) {
                gain(linear=2.0)
            } else {
                gain(linear=1.0)
            } >> emit
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to compile audio pipeline");
        let sample_rate = 48000;
        let channels = 2;
        let num_samples = 500;
        let data = vec![0.25f32; channels * num_samples];
        let audio = Audio::from_f32_planar(&data, channels, sample_rate).unwrap();

        let outputs = pipeline
            .run(Payload::Audio(audio))
            .expect("Failed to execute audio pipeline");
        let result = outputs.into_single().expect("Expected single output");
        if let Payload::Audio(out_aud) = result {
            assert_eq!(out_aud.sample_rate, 48000);
            assert_eq!(out_aud.channels(), 2);
            assert_eq!(out_aud.channel_layout, AudioChannelLayout::Stereo);
            assert_eq!(out_aud.layout, AudioLayout::Planar);
            let samples = out_aud.as_f32_slice().unwrap();
            assert_eq!(samples[0], 0.5);
        } else {
            panic!("Expected Audio payload output");
        }
    }

    #[test]
    fn test_image_pipeline_end_to_end() {
        let morf_src = r#"
            accept Image img_in

            $img_in >> to_tensor(color="rgb", dtype="f32", layout="hwc", normalize=true)
                    >> resize(width=32, height=24, filter="bilinear")
                    >> color_adjust(contrast=1.1, brightness=0.05)
                    >> gaussian_blur(sigma=1.0)
                    >> to_image(color="rgba", dtype="u8")
                    >> emit
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to compile image pipeline");
        let width = 64;
        let height = 48;
        let data = vec![128u8; width * height * 3];
        let img = Image::from_u8_hwc(&data, width, height, ColorSpace::Rgb).unwrap();

        let outputs = pipeline
            .run(Payload::Image(img))
            .expect("Failed to execute image pipeline");
        let result = outputs.into_single().expect("Expected single output");
        if let Payload::Image(out_img) = result {
            assert_eq!(out_img.width(), 32);
            assert_eq!(out_img.height(), 24);
            assert_eq!(out_img.channels(), 4);
            assert_eq!(out_img.color_space, ColorSpace::Rgba);
            assert_eq!(out_img.dtype(), TensorDType::U8);
            assert_eq!(out_img.layout, ImageLayout::Hwc);
        } else {
            panic!("Expected Image payload output");
        }
    }

    #[test]
    fn test_mid_stream_emit_pass_through() {
        let morf_src = r#"
            accept Audio audio_in

            $audio_in
                >> gain(linear=2.0)
                >> emit("boosted")
                >> gain(linear=3.0)
                >> emit("amplified_final")
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to compile pipeline");
        let audio = Audio::from_f32_planar(&[1.0, 2.0, 3.0], 1, 44100).unwrap();
        let outputs = pipeline
            .run(Payload::Audio(audio))
            .expect("Execution failed");

        assert_eq!(outputs.len(), 2);
        assert!(outputs.contains_key("boosted"));
        assert!(outputs.contains_key("amplified_final"));

        if let Payload::Audio(boosted) = &outputs["boosted"] {
            let slice = boosted.as_f32_slice().unwrap();
            assert_eq!(slice, &[2.0, 4.0, 6.0]);
        } else {
            panic!("Expected Audio for boosted");
        }

        if let Payload::Audio(final_a) = &outputs["amplified_final"] {
            let slice = final_a.as_f32_slice().unwrap();
            assert_eq!(slice, &[6.0, 12.0, 18.0]);
        } else {
            panic!("Expected Audio for amplified_final");
        }
    }

    #[test]
    fn test_action_pack_package_import() {
        let morf_src = r#"
            import audio_essentials.latest

            accept Audio audio_in

            $audio_in >> gain(linear=2.5) >> emit
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to compile pipeline");
        let audio = Audio::from_f32_planar(&[2.0, 4.0], 1, 44100).unwrap();
        let outputs = pipeline
            .run(Payload::Audio(audio))
            .expect("Execution failed");
        let result = outputs.into_single().expect("Expected single output");
        if let Payload::Audio(a) = result {
            assert_eq!(a.as_f32_slice().unwrap(), &[5.0, 10.0]);
        } else {
            panic!("Expected Audio output");
        }
    }

    #[test]
    fn test_action_pack_from_import_with_alias() {
        let morf_src = r#"
            import base.latest
            from audio_essentials.latest import gain as amp

            accept Audio audio_in

            $audio_in >> identity >> amp(linear=3.0) >> emit
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to compile pipeline");
        let audio = Audio::from_f32_planar(&[1.0, 3.0], 1, 44100).unwrap();
        let outputs = pipeline
            .run(Payload::Audio(audio))
            .expect("Execution failed");
        let result = outputs.into_single().expect("Expected single output");
        if let Payload::Audio(a) = result {
            assert_eq!(a.as_f32_slice().unwrap(), &[3.0, 9.0]);
        } else {
            panic!("Expected Audio output");
        }
    }

    #[test]
    fn test_action_pack_qualified_invocation() {
        let morf_src = r#"
            accept Audio audio_in

            $audio_in >> audio_essentials.gain(linear=4.0) >> base.identity >> emit
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to compile pipeline");
        let audio = Audio::from_f32_planar(&[2.0, 5.0], 1, 44100).unwrap();
        let outputs = pipeline
            .run(Payload::Audio(audio))
            .expect("Execution failed");
        let result = outputs.into_single().expect("Expected single output");
        if let Payload::Audio(a) = result {
            assert_eq!(a.as_f32_slice().unwrap(), &[8.0, 20.0]);
        } else {
            panic!("Expected Audio output");
        }
    }

    #[test]
    fn test_audio_to_audio_and_to_wav_end_to_end() {
        let morf_src = r#"
            import audio_essentials/latest

            accept RawBytes audio_in

            $audio_in >> to_audio(channels=1, sample_rate=48000, dtype="f32") >> gain(linear=2.0) >> to_wav >> emit
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to compile pipeline");
        let f32_samples = [0.25f32, -0.25f32];
        let byte_slice: &[u8] = unsafe {
            std::slice::from_raw_parts(
                f32_samples.as_ptr() as *const u8,
                f32_samples.len() * std::mem::size_of::<f32>(),
            )
        };

        let outputs = pipeline
            .run(Payload::Data {
                buffer: abi_stable::std_types::RVec::from(byte_slice.to_vec()),
            })
            .expect("Execution failed");

        let result = outputs.into_single().expect("Expected single output");
        if let Payload::Data { buffer } = result {
            assert_eq!(buffer.len(), 44 + 4); // 44 byte header + 2 samples * 2 bytes i16
            assert_eq!(&buffer[0..4], b"RIFF");
            assert_eq!(&buffer[8..12], b"WAVE");
            let s0 = i16::from_le_bytes([buffer[44], buffer[45]]);
            let s1 = i16::from_le_bytes([buffer[46], buffer[47]]);
            assert_eq!(s0, 16384); // 0.25 * 2.0 = 0.5 * 32767 = 16384
            assert_eq!(s1, -16384);
        } else {
            panic!("Expected Payload::Data output");
        }
    }

    #[test]
    fn test_tensor_and_math_pipeline() {
        let morf_src = r#"
            import tensor_essentials/latest
            import math_essentials/latest
            import tensor_stats/latest
            import nn_essentials/latest

            accept Tensor x

            $x >> reshape(shape="2,2") >> relu >> mul(scalar=2.0) >> sum(axis=1) >> emit
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to compile pipeline");
        let tensor = Tensor::from_f32_slice(&[-1.0, 2.0, 3.0, -4.0]);
        let outputs = pipeline
            .run(Payload::Tensor(tensor))
            .expect("Execution failed");
        let result = outputs.into_single().expect("Expected single output");
        if let Payload::Tensor(t) = result {
            // [-1, 2] -> relu -> [0, 2] -> mul 2 -> [0, 4] -> sum -> 4
            // [3, -4] -> relu -> [3, 0] -> mul 2 -> [6, 0] -> sum -> 6
            assert_eq!(t.shape.as_slice(), &[2]);
            assert_eq!(t.as_f32_slice().unwrap(), &[4.0, 6.0]);
        } else {
            panic!("Expected Tensor output");
        }
    }

    #[test]
    fn test_linalg_and_nn_pipeline() {
        let morf_src = r#"
            import nn_essentials/latest
            import linalg_essentials/latest

            accept Tensor x

            $x >> softmax(axis=-1) >> trace >> emit
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to compile pipeline");
        let tensor = Tensor::from_f32_shape(&[0.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let outputs = pipeline
            .run(Payload::Tensor(tensor))
            .expect("Execution failed");
        let result = outputs.into_single().expect("Expected single output");
        match result {
            Payload::Scalar(t) | Payload::Tensor(t) => {
                // [0, 0] softmax -> [0.5, 0.5]
                // [0, 0] softmax -> [0.5, 0.5]
                // trace -> 0.5 + 0.5 = 1.0
                assert_eq!(t.as_f32_slice().unwrap(), &[1.0]);
            }
            other => panic!("Expected Scalar/Tensor output, got {:?}", other),
        }
    }
    // ---------------------------------------------------------------------
    // Type-preserving 'each': each iteration keeps the payload variant, and
    // the loop result is restacked back into that same variant.
    // ---------------------------------------------------------------------

    fn stereo_planar_audio(num_samples: usize) -> Payload {
        let samples: Vec<f32> = (0..(2 * num_samples)).map(|x| x as f32).collect();
        let tensor = Tensor::from_f32_shape(&samples, vec![2, num_samples]).unwrap();
        Payload::Audio(
            Audio::new(
                tensor,
                48000,
                AudioChannelLayout::Stereo,
                AudioLayout::Planar,
            )
            .unwrap(),
        )
    }

    #[test]
    fn test_each_over_planar_stereo_audio_preserves_audio_payload() {
        let morf_src = r#"
            from audio_essentials/latest import gain
            accept Audio audio

            $audio >> each ($channel) {
                $channel >> gain(linear=1.0)
            } >> emit
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to parse pipeline");
        let outputs = pipeline
            .run(stereo_planar_audio(64))
            .expect("Pipeline execution failed");
        let result = outputs.into_single().unwrap();

        match result {
            Payload::Audio(a) => {
                assert_eq!(a.tensor.shape.as_slice(), &[2, 64]);
                assert_eq!(a.channels(), 2);
                assert_eq!(a.sample_rate, 48000);
                assert_eq!(a.layout, AudioLayout::Planar);
            }
            other => panic!(
                "Expected Audio payload, got {}",
                core_types::payload_kind_name(&other)
            ),
        }
    }

    #[test]
    fn test_each_over_audio_binds_audio_to_loop_variable() {
        // The body uses an audio-native action, which only succeeds if the loop
        // variable is a real Audio payload rather than a bare Tensor.
        let morf_src = r#"
            from audio_essentials/latest import gain
            accept Audio audio

            $audio >> each ($channel) {
                $channel >> gain(linear=2.0)
            } >> emit
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to parse pipeline");
        let outputs = pipeline
            .run(stereo_planar_audio(32))
            .expect("Pipeline execution failed");
        let result = outputs.into_single().unwrap();

        match result {
            Payload::Audio(a) => {
                assert_eq!(a.tensor.shape.as_slice(), &[2, 32]);
                let samples = a.tensor.as_f32_slice().unwrap();
                assert!(
                    (samples[1] - 2.0).abs() < 1e-6,
                    "gain should have doubled sample 1"
                );
            }
            other => panic!(
                "Expected Audio payload, got {}",
                core_types::payload_kind_name(&other)
            ),
        }
    }

    #[test]
    fn test_each_over_chw_image_iterates_channels() {
        let morf_src = r#"
            from base/latest import identity
            accept Image $image

            $image >> each ($channel) {
                $channel >> identity
            } >> emit
        "#;

        // CHW [C, H, W] = [3, 4, 5] iterates the 3 channels.
        let data: Vec<u8> = (0..60).collect();
        let tensor =
            Tensor::from_rvec_u8(RVec::from(data), vec![3, 4, 5], TensorDType::U8).unwrap();
        let image = Image::new(tensor, ColorSpace::Rgb, ImageLayout::Chw).unwrap();

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to parse pipeline");
        let outputs = pipeline
            .run(Payload::Image(image))
            .expect("Pipeline execution failed");
        let result = outputs.into_single().unwrap();

        // A grayscale channel is still an image, so the restack yields an Image.
        match result {
            Payload::Image(img) => {
                assert_eq!(img.tensor.shape.as_slice(), &[3, 4, 5]);
                assert_eq!(img.color_space, ColorSpace::Rgb);
                assert_eq!(img.layout, ImageLayout::Chw);
            }
            other => panic!(
                "Expected Image, got {}",
                core_types::payload_kind_name(&other)
            ),
        }
    }

    #[test]
    fn test_each_rejects_a_body_that_changes_the_payload_type() {
        // The body converts an Image into a Tensor, so the loop no longer
        // produces the type it was iterating and the restack must refuse it.
        let morf_src = r#"
            from base/latest import to_tensor
            accept Image $image

            $image >> each ($channel) {
                $channel >> to_tensor
            } >> emit
        "#;

        let data: Vec<u8> = (0..60).collect();
        let tensor =
            Tensor::from_rvec_u8(RVec::from(data), vec![3, 4, 5], TensorDType::U8).unwrap();
        let image = Image::new(tensor, ColorSpace::Rgb, ImageLayout::Chw).unwrap();

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to parse pipeline");
        let err = pipeline
            .run(Payload::Image(image))
            .expect_err("a type-changing body should be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("every iteration must return Image"),
            "unexpected diagnostic: {}",
            msg
        );
    }

    #[test]
    fn test_each_over_hwc_image_iterates_channels_not_rows() {
        let morf_src = r#"
            from base/latest import identity
            accept Image image

            $image >> each ($channel) {
                $channel >> identity
            } >> emit
        "#;

        // HWC [H, W, C] = [40, 50, 3] iterates 3 channels, not 40 rows.
        let data: Vec<u8> = (0..6000).map(|x| (x % 256) as u8).collect();
        let tensor =
            Tensor::from_rvec_u8(RVec::from(data), vec![40, 50, 3], TensorDType::U8).unwrap();
        let image = Image::new(tensor, ColorSpace::Rgb, ImageLayout::Hwc).unwrap();

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to parse pipeline");
        let outputs = pipeline
            .run(Payload::Image(image))
            .expect("Pipeline execution failed");
        let result = outputs.into_single().unwrap();

        match result {
            Payload::Image(img) => {
                assert_eq!(img.tensor.shape.as_slice(), &[40, 50, 3]);
                assert_eq!(img.color_space, ColorSpace::Rgb);
                assert_eq!(img.layout, ImageLayout::Hwc);
            }
            other => panic!(
                "Expected Image, got {}",
                core_types::payload_kind_name(&other)
            ),
        }
    }

    #[test]
    fn test_each_over_mono_audio_iterates_samples_and_restacks_to_audio() {
        // Mono is rank-1 [N], so axis 0 is the sample axis. Per-sample slices
        // are rank-0, which Audio cannot hold, so the loop variable is a Tensor
        // and the restacked result is Audio again.
        let morf_src = r#"
            import base/latest
            accept Audio audio

            $audio >> each ($sample) {
                $sample >> identity
            } >> emit
        "#;

        let samples: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let tensor = Tensor::from_f32_shape(&samples, vec![16]).unwrap();
        let mono =
            Audio::new(tensor, 44100, AudioChannelLayout::Mono, AudioLayout::Planar).unwrap();

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to parse pipeline");
        let outputs = pipeline
            .run(Payload::Audio(mono))
            .expect("Pipeline execution failed");
        let result = outputs.into_single().unwrap();

        match result {
            Payload::Audio(a) => {
                assert_eq!(a.tensor.shape.as_slice(), &[16]);
                assert_eq!(a.sample_rate, 44100);
            }
            other => panic!(
                "Expected Audio, got {}",
                core_types::payload_kind_name(&other)
            ),
        }
    }

    #[test]
    fn test_each_over_interleaved_audio_iterates_channel_axis() {
        // Interleaved [N, C] = [8, 2] iterates axis 1, giving 2 iterations.
        let morf_src = r#"
            from audio_essentials/latest import gain
            accept Audio audio

            $audio >> each ($channel) {
                $channel >> gain(linear=1.0)
            } >> emit
        "#;

        let samples: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let tensor = Tensor::from_f32_shape(&samples, vec![8, 2]).unwrap();
        let audio = Audio::new(
            tensor,
            48000,
            AudioChannelLayout::Stereo,
            AudioLayout::Interleaved,
        )
        .unwrap();

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to parse pipeline");
        let outputs = pipeline
            .run(Payload::Audio(audio))
            .expect("Pipeline execution failed");
        let result = outputs.into_single().unwrap();

        match result {
            Payload::Audio(a) => {
                assert_eq!(a.layout, AudioLayout::Interleaved);
                assert_eq!(a.sample_rate, 48000);
            }
            other => panic!(
                "Expected Audio, got {}",
                core_types::payload_kind_name(&other)
            ),
        }
    }

    #[test]
    fn test_each_on_raw_bytes_payload_errors_at_runtime() {
        let morf_src = r#"
            import base/latest
            accept RawBytes raw

            $raw >> each ($b) {
                $b >> identity
            } >> emit
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to parse pipeline");
        let payload = Payload::Data {
            buffer: RVec::from(vec![1u8, 2, 3, 4]),
        };
        let err = pipeline
            .run(payload)
            .expect_err("each on a Data payload should fail");
        assert!(
            err.to_string().contains("raw-bytes"),
            "expected a raw-bytes diagnostic, got: {}",
            err
        );
    }

    #[test]
    fn test_each_on_scalar_tensor_errors() {
        let morf_src = r#"
            import base/latest
            accept Scalar tensor_in

            $tensor_in >> each ($x) {
                $x >> identity
            } >> emit
        "#;

        let mut pipeline = Morflow::from_str(morf_src).expect("Failed to parse pipeline");
        let tensor = Tensor::from_f32_shape(&[1.0], vec![]).unwrap();
        let err = pipeline
            .run(Payload::Tensor(tensor))
            .expect_err("each on a rank-0 tensor should fail");
        assert!(
            err.to_string().contains("scalar"),
            "expected a scalar diagnostic, got: {}",
            err
        );
    }
}
