pub mod ast;
pub mod parser;

use chumsky::error::Simple;
use chumsky::Parser;

pub use ast::*;

/// Parses a `.morf` pipeline specification into an AST `Pipeline`.
pub fn parse(source: &str) -> Result<Pipeline, Vec<Simple<char>>> {
    parser::parser().parse(source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_pipeline() {
        let src = r#"
            load_audio("sample.wav")
                >> resample(rate=44100)
                >> gain(db=+3.5)
                >> export("output.wav")
        "#;
        let res = parse(src);
        assert!(res.is_ok(), "Failed to parse: {:?}", res.err());
        let pipeline = res.unwrap();
        assert_eq!(pipeline.statements.len(), 1);
        let Statement::Flow(chain) = &pipeline.statements[0];
        assert_eq!(chain.steps.len(), 4);
    }

    #[test]
    fn test_tap_and_variables() {
        let src = r#"
            load_image("input.png")
                >> resize(width=1920, height=1080) >> $clean_image
                >> edge_detect(sensitivity=0.8) >> $edges

            blend($clean_image, overlay=$edges, opacity=0.5)
                >> save_image("blended.png")
        "#;
        let res = parse(src);
        assert!(res.is_ok(), "Failed to parse: {:?}", res.err());
        let pipeline = res.unwrap();
        assert_eq!(pipeline.statements.len(), 2);
    }

    #[test]
    fn test_slicing() {
        let src = r#"
            $audio_buffer[0:48000]
                >> normalize
                >> save_audio("preview.wav")

            $tensor[:, 0:3, 100:200, 100:200]
                >> denoise
                >> $cropped_clean
        "#;
        let res = parse(src);
        assert!(res.is_ok(), "Failed to parse: {:?}", res.err());
        let pipeline = res.unwrap();
        assert_eq!(pipeline.statements.len(), 2);
    }

    #[test]
    fn test_each_loop_with_named_var() {
        let src = r#"
            $video_tensor[axis=0] >> each ($frame) {
                $frame
                    >> detect_faces(confidence=0.9)
                    >> blur_region(mask="faces")
            } >> encode_video("censored.mp4")
        "#;
        let res = parse(src);
        assert!(res.is_ok(), "Failed to parse: {:?}", res.err());
        let pipeline = res.unwrap();
        assert_eq!(pipeline.statements.len(), 1);
    }

    #[test]
    fn test_branching_if_else() {
        let src = r#"
            $payload
                >> analyze_quality >> $metrics
                >> if ($metrics.snr < 20.0) {
                    spectral_subtract(noise_profile="ambient")
                        >> lowpass(cutoff_hz=8000)
                } else {
                    identity
                }
                >> limiter(ceiling_db=-0.1)
                >> export("final.wav")
        "#;
        let res = parse(src);
        assert!(res.is_ok(), "Failed to parse: {:?}", res.err());
        let pipeline = res.unwrap();
        assert_eq!(pipeline.statements.len(), 1);
    }

    #[test]
    fn test_full_morf_document() {
        let src = r#"
        # Real-world Morflow Pipeline
        pipeline "AudioEnhanceAndSplit" {
            // 1. Load source and tap original
            load_audio("source.flac") >> $raw_audio

            // 2. Extract 5-second slice for calibration
            $raw_audio[0:240000]
                >> compute_noise_profile
                >> $noise_profile

            // 3. Main processing with branch
            $raw_audio
                >> denoise(profile=$noise_profile)
                >> if ($raw_audio.peak > 0.0) {
                    compressor(ratio=4.0, attack_ms=15)
                        >> soft_clip
                } else {
                    normalize(target_lufs=-14.0)
                }
                >> $mastered

            // 4. Multi-channel loop across dimension 1
            $mastered[axis=1] >> each ($channel) {
                $channel
                    >> highpass(freq=30)
                    >> stereo_widen(amount=1.2)
            } >> $final_mix

            // 5. Output sink
            $final_mix >> export("master_out.wav")
        }
        "#;
        let res = parse(src);
        assert!(res.is_ok(), "Failed to parse: {:?}", res.err());
        let pipeline = res.unwrap();
        assert_eq!(pipeline.name, Some("AudioEnhanceAndSplit".to_string()));
        assert_eq!(pipeline.statements.len(), 5);
    }

    #[test]
    fn test_load_example_morf_file() {
        let example_content =
            include_str!("../../examples/rust/audio_processing/audio_pipeline.morf");
        let res = parse(example_content);
        assert!(res.is_ok(), "Failed to parse example file: {:?}", res.err());
        let pipeline = res.unwrap();
        assert_eq!(pipeline.statements.len(), 1);
        assert_eq!(pipeline.params.len(), 2);
    }

    #[test]
    fn test_pipeline_with_parameters() {
        let src = r#"
            accept $input_audio
            accept $sample_rate = 44100

            $input_audio
                >> resample(rate=$sample_rate)
                >> export("out.wav")
        "#;
        let res = parse(src);
        assert!(res.is_ok(), "Failed to parse: {:?}", res.err());
        let pipeline = res.unwrap();
        assert_eq!(pipeline.params.len(), 2);
        assert_eq!(pipeline.params[0].name, "input_audio");
        assert_eq!(pipeline.params[0].default_value, None);
        assert_eq!(pipeline.params[1].name, "sample_rate");
        assert_eq!(pipeline.params[1].default_value, Some(Value::Int(44100)));
        assert_eq!(pipeline.statements.len(), 1);
    }

    #[test]
    fn test_load_audio_split_example() {
        let example_content =
            include_str!("../../examples/rust/audio_split/audio_split_pipeline.morf");
        let res = parse(example_content);
        assert!(
            res.is_ok(),
            "Failed to parse audio_split_pipeline.morf: {:?}",
            res.err()
        );
        let pipeline = res.unwrap();
        assert_eq!(pipeline.params.len(), 1);
        assert_eq!(pipeline.params[0].name, "input_audio");
        assert_eq!(pipeline.statements.len(), 2);
    }

    #[test]
    fn test_load_image_pipeline_example() {
        let example_content =
            include_str!("../../examples/rust/image_processing/image_pipeline.morf");
        let res = parse(example_content);
        assert!(
            res.is_ok(),
            "Failed to parse image_pipeline.morf: {:?}",
            res.err()
        );
        let pipeline = res.unwrap();
        assert_eq!(pipeline.params.len(), 3);
        assert_eq!(pipeline.params[0].name, "img_in");
        assert_eq!(pipeline.params[1].name, "target_width");
        assert_eq!(pipeline.params[2].name, "target_height");
        assert_eq!(pipeline.statements.len(), 1);
    }
}
