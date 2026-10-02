pub mod ast;
pub mod parser;

use chumsky::error::{Simple, SimpleReason};
use chumsky::Parser;

pub use ast::*;

/// Parses a `.morf` pipeline specification into an AST `Pipeline`.
pub fn parse(source: &str) -> Result<Pipeline, Vec<Simple<char>>> {
    parser::parser().parse(source)
}

/// Renders a parse error as a message.
///
/// `Simple`'s own `Display` ignores custom messages, which is where the
/// grammar puts its advice about things like a missing parameter type.
pub fn format_error(err: &Simple<char>) -> String {
    match err.reason() {
        SimpleReason::Custom(msg) => msg.clone(),
        SimpleReason::Unclosed { delimiter, .. } => {
            format!("unclosed delimiter '{}'", delimiter)
        }
        SimpleReason::Unexpected => {
            let found = match err.found() {
                Some(c) => format!("character '{}'", c),
                None => "end of input".to_string(),
            };
            let expected: Vec<String> = err
                .expected()
                .map(|c| match c {
                    Some(ch) => format!("'{}'", ch),
                    None => "end of input".to_string(),
                })
                .collect();
            match expected.len() {
                0 => format!("unexpected {}", found),
                1 => format!("unexpected {}, expected {}", found, expected[0]),
                _ => format!(
                    "unexpected {}, expected one of {}",
                    found,
                    expected.join(", ")
                ),
            }
        }
    }
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
        import base/latest
        from audio_basics/latest import load_audio, compute_noise_profile, denoise, compressor, soft_clip, normalize, highpass, stereo_widen, export

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
        "#;
        let res = parse(src);
        assert!(res.is_ok(), "Failed to parse: {:?}", res.err());
        let pipeline = res.unwrap();
        assert_eq!(pipeline.imports.len(), 2);
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
        assert_eq!(pipeline.params[0].param_type, ParamType::Bytes);
        assert_eq!(pipeline.params[1].param_type, ParamType::IntArg);
    }

    #[test]
    fn test_pipeline_with_parameters() {
        let src = r#"
            accept Bytes $input_audio
            accept IntArg $sample_rate = 44100

            $input_audio
                >> resample(rate=$sample_rate)
                >> export("out.wav")
        "#;
        let res = parse(src);
        assert!(res.is_ok(), "Failed to parse: {:?}", res.err());
        let pipeline = res.unwrap();
        assert_eq!(pipeline.params.len(), 2);
        assert_eq!(pipeline.params[0].name, "input_audio");
        assert_eq!(pipeline.params[0].param_type, ParamType::Bytes);
        assert_eq!(pipeline.params[0].default_value, None);
        assert_eq!(pipeline.params[1].name, "sample_rate");
        assert_eq!(pipeline.params[1].param_type, ParamType::IntArg);
        assert_eq!(pipeline.params[1].default_value, Some(Value::Int(44100)));
        assert_eq!(pipeline.statements.len(), 1);
    }

    #[test]
    fn test_every_param_type_parses() {
        let src = r#"
            accept Bytes $bytes
            accept IntArg $int
            accept FloatArg $float
            accept StrArg $str
            accept BoolArg $bool
            accept Scalar $scalar
            accept Composite $composite
            accept Tensor $tensor
            accept Image $image
            accept Audio $audio

            $bytes >> identity >> emit
        "#;
        let pipeline = parse(src).unwrap_or_else(|e| panic!("Failed to parse: {:?}", e));
        let types: Vec<ParamType> = pipeline
            .params
            .iter()
            .map(|p| p.param_type.clone())
            .collect();
        assert_eq!(
            types,
            vec![
                ParamType::Bytes,
                ParamType::IntArg,
                ParamType::FloatArg,
                ParamType::StrArg,
                ParamType::BoolArg,
                ParamType::Scalar,
                ParamType::Composite,
                ParamType::Tensor(ParamShape::AnyRank),
                ParamType::Image(ParamShape::AnyRank),
                ParamType::Audio(ParamShape::AnyRank),
            ]
        );
    }

    #[test]
    fn test_shape_suffixes_parse() {
        let src = r#"
            accept Tensor[rank=2] $ranked
            accept Tensor[*,*,3] $channels_last
            accept Tensor[2, 3] $fixed
            accept Image[rank=3] $image
            accept Audio[rank=2] $audio
            accept Audio[] $unpinned

            $ranked >> identity >> emit
        "#;
        let pipeline = parse(src).unwrap_or_else(|e| panic!("Failed to parse: {:?}", e));
        assert_eq!(
            pipeline.params[0].param_type,
            ParamType::Tensor(ParamShape::Ranked {
                dims: vec![ParamDim::Any, ParamDim::Any]
            })
        );
        assert_eq!(
            pipeline.params[1].param_type,
            ParamType::Tensor(ParamShape::Ranked {
                dims: vec![ParamDim::Any, ParamDim::Any, ParamDim::Fixed(3)]
            })
        );
        assert_eq!(
            pipeline.params[2].param_type,
            ParamType::Tensor(ParamShape::Ranked {
                dims: vec![ParamDim::Fixed(2), ParamDim::Fixed(3)]
            })
        );
        assert_eq!(
            pipeline.params[3].param_type,
            ParamType::Image(ParamShape::Ranked {
                dims: vec![ParamDim::Any, ParamDim::Any, ParamDim::Any]
            })
        );
        assert_eq!(
            pipeline.params[4].param_type,
            ParamType::Audio(ParamShape::Ranked {
                dims: vec![ParamDim::Any, ParamDim::Any]
            })
        );
        // Empty brackets say nothing, so they leave the rank unpinned.
        assert_eq!(
            pipeline.params[5].param_type,
            ParamType::Audio(ParamShape::AnyRank)
        );
    }

    #[test]
    fn test_param_type_display_normalises() {
        let all_any = ParamType::Tensor(ParamShape::Ranked {
            dims: vec![ParamDim::Any, ParamDim::Any],
        });
        assert_eq!(all_any.to_string(), "Tensor[rank=2]");
        assert_eq!(
            ParamType::Image(ParamShape::Ranked {
                dims: vec![ParamDim::Any, ParamDim::Any, ParamDim::Fixed(3)]
            })
            .to_string(),
            "Image[*, *, 3]"
        );
        assert_eq!(ParamType::Scalar.to_string(), "Scalar");
        assert_eq!(ParamType::StrArg.to_string(), "StrArg");
    }

    #[test]
    fn test_accept_without_a_type_is_rejected() {
        let src = "accept $input_audio\n\n$input_audio >> identity >> emit";
        let errs = parse(src).unwrap_err();
        let messages: Vec<String> = errs.iter().map(format_error).collect();
        assert!(
            messages
                .iter()
                .any(|m| m.contains("every 'accept' needs a type")),
            "unexpected errors: {:?}",
            messages
        );
    }

    #[test]
    fn test_rank_zero_is_rejected() {
        for keyword in ["Tensor", "Image", "Audio"] {
            let src = format!("accept {}[rank=0] $value", keyword);
            let errs = parse(&src).unwrap_err();
            let messages: Vec<String> = errs.iter().map(format_error).collect();
            assert!(
                messages.iter().any(|m| m.contains("the type 'Scalar'")),
                "unexpected errors for {}: {:?}",
                keyword,
                messages
            );
        }
    }

    #[test]
    fn test_rank_sugar_cannot_be_mixed_with_dims() {
        let src = "accept Tensor[rank=2,*] $value";
        assert!(
            parse(src).is_err(),
            "rank=2 cannot be combined with a dimension list"
        );
    }

    #[test]
    fn test_unknown_param_type_is_rejected() {
        assert!(parse("accept Video $clip").is_err());
        assert!(parse("accept Int $count").is_err());
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
        assert_eq!(pipeline.statements.len(), 3);
    }

    #[test]
    fn test_action_pack_imports() {
        let src = r#"
            import base/latest
            import audio_basics/latest as audio
            from image_basics/0.1.0 import resize, color_adjust as ca, gaussian_blur
            import base/0.1.0/identity as ident

            accept Image $img_in

            $img_in
                >> base/identity
                >> resize(512, 512)
                >> ca(contrast=1.1)
                >> image_basics/gaussian_blur(sigma=1.5)
                >> ident
                >> emit
        "#;
        let res = parse(src);
        assert!(res.is_ok(), "Failed to parse: {:?}", res.err());
        let pipeline = res.unwrap();
        assert_eq!(pipeline.imports.len(), 4);

        match &pipeline.imports[0] {
            ImportStmt::Package(pkg) => {
                assert_eq!(pkg.package, "base");
                assert_eq!(pkg.version, "latest");
                assert_eq!(pkg.alias, None);
            }
            _ => panic!("Expected Package import"),
        }

        match &pipeline.imports[1] {
            ImportStmt::Package(pkg) => {
                assert_eq!(pkg.package, "audio_basics");
                assert_eq!(pkg.version, "latest");
                assert_eq!(pkg.alias, Some("audio".to_string()));
            }
            _ => panic!("Expected Package import with alias"),
        }

        match &pipeline.imports[2] {
            ImportStmt::Items(items) => {
                assert_eq!(items.package, "image_basics");
                assert_eq!(items.version, "0.1.0");
                assert_eq!(items.items.len(), 3);
                assert_eq!(items.items[0].name, "resize");
                assert_eq!(items.items[0].alias, None);
                assert_eq!(items.items[1].name, "color_adjust");
                assert_eq!(items.items[1].alias, Some("ca".to_string()));
                assert_eq!(items.items[2].name, "gaussian_blur");
                assert_eq!(items.items[2].alias, None);
            }
            _ => panic!("Expected Items import"),
        }

        match &pipeline.imports[3] {
            ImportStmt::Items(items) => {
                assert_eq!(items.package, "base");
                assert_eq!(items.version, "0.1.0");
                assert_eq!(items.items.len(), 1);
                assert_eq!(items.items[0].name, "identity");
                assert_eq!(items.items[0].alias, Some("ident".to_string()));
            }
            _ => panic!("Expected single item import"),
        }

        assert_eq!(pipeline.params.len(), 1);
        assert_eq!(pipeline.statements.len(), 1);
        let Statement::Flow(flow) = &pipeline.statements[0];
        assert_eq!(flow.steps.len(), 7);
        if let FlowStep::Action(call) = &flow.steps[1] {
            assert_eq!(call.name, "base/identity");
        }
        if let FlowStep::Action(call) = &flow.steps[2] {
            assert_eq!(call.name, "resize");
        }
        if let FlowStep::Action(call) = &flow.steps[3] {
            assert_eq!(call.name, "ca");
        }
        if let FlowStep::Action(call) = &flow.steps[4] {
            assert_eq!(call.name, "image_basics/gaussian_blur");
        }
        if let FlowStep::Action(call) = &flow.steps[5] {
            assert_eq!(call.name, "ident");
        }
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

#[cfg(test)]
mod plugin_tests {
    #[test]
    fn plugin_versions_are_explicit() {
        for version in ["0.1.0", "latest"] {
            let ast = super::parse(&format!(
                "plugin opencv-bridge/{version}\naccept Tensor $x\n$x >> emit\n"
            ))
            .unwrap();
            assert_eq!(ast.plugins[0].name, "opencv-bridge");
            assert_eq!(ast.plugins[0].version, version);
        }
        let errors =
            super::parse("plugin opencv-bridge\naccept Tensor $x\n$x >> emit\n").unwrap_err();
        assert!(errors
            .iter()
            .map(super::format_error)
            .any(|message| message.contains("Plugin version is required")));
        assert!(super::parse("import image_opencv\naccept Tensor $x\n$x >> emit\n").is_err());
        assert!(
            super::parse("from image_opencv import resize\naccept Tensor $x\n$x >> emit\n")
                .is_err()
        );
    }
}
