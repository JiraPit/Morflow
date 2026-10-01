//! Compare complete and partially known predictions against native execution.
//! Every partial prediction must cover the same concrete output; this catches
//! rank/dimension promises that become incorrect when information is removed.
use core_types::{
    ActionArgs, Dimension, RBox, Shape, ShapeResult, Tuple2, ValueShape, ValueShapeResult,
};
use pipeline::{ActionIdentity, ActionRegistry, Audio, LoadedAction, Payload, Tensor};

fn args(values: &[(&str, &str)]) -> ActionArgs {
    ActionArgs {
        positional: vec![].into(),
        named: values
            .iter()
            .map(|(k, v)| Tuple2((*k).into(), (*v).into()))
            .collect(),
    }
}
fn tensor(dims: &[usize]) -> Payload {
    let count = dims.iter().product();
    // Positive diagonal matrices also exercise successful inversion/Cholesky.
    let mut values = vec![0.5; count];
    if dims.len() >= 2 && dims[dims.len() - 2] == dims[dims.len() - 1] {
        let n = dims[dims.len() - 1];
        if n > 0 {
            for matrix in values.chunks_mut(n * n) {
                for i in 0..n {
                    matrix[i * n + i] = 4.;
                }
            }
        }
    }
    Payload::from_tensor(Tensor::from_f32_shape(&values, dims.to_vec()).unwrap())
}
fn partial(input: &ValueShape, mask: usize) -> ValueShape {
    match input {
        ValueShape::Leaf { kind, shape } => ValueShape::Leaf {
            kind: *kind,
            shape: shape
                .as_ref()
                .into_option()
                .map(|s| {
                    Shape::new(s.dims().iter().enumerate().map(|(i, d)| {
                        if mask & (1 << i) != 0 {
                            Dimension::Unknown
                        } else {
                            *d
                        }
                    }))
                })
                .into(),
        },
        ValueShape::Composite(items) => {
            ValueShape::composite(items.iter().map(|s| partial(s, mask)))
        }
        ValueShape::Unknown => ValueShape::Unknown,
    }
}
fn verify(loaded: &LoadedAction, input: &ValueShape, args: &ActionArgs, output: &Payload) {
    // Check the component view against the same ordered output prediction.
    if let (Some(shape), Payload::Composite(items)) = (input.shape(), output) {
        if let Some(components) = loaded.output_components(shape, args) {
            assert_eq!(components.len(), items.len(), "{}", loaded.name);
            for (component, item) in components.iter().zip(items) {
                let shape = match &component.shape {
                    ShapeResult::Ok(shape) => Some(shape.clone()),
                    ShapeResult::Unknown => None,
                    ShapeResult::Invalid(e) => panic!("{}: {e}", loaded.name),
                };
                ValueShape::Leaf {
                    kind: component.kind,
                    shape: shape.into(),
                }
                .verify(item)
                .unwrap();
            }
        }
    }
    if let Some(result) = loaded.output_value_result(input, args) {
        match result {
            ValueShapeResult::Ok(expected) => expected
                .verify(output)
                .unwrap_or_else(|e| panic!("{} input={input:?} args={args:?}: {e}", loaded.name)),
            ValueShapeResult::Invalid(e) => panic!(
                "{} rejected a successful input={input:?} args={args:?}: {e}",
                loaded.name
            ),
            ValueShapeResult::Unknown => (),
        }
    } else if let Some(shape) = input.shape() {
        match loaded.output_result(shape, args) {
            ShapeResult::Ok(expected) => {
                let actual = ValueShape::from_payload(output);
                let actual = actual.shape().unwrap();
                assert_eq!(
                    expected.rank(),
                    actual.rank(),
                    "{} input={input:?} args={args:?}",
                    loaded.name
                );
                for (expected, actual) in expected.dims().iter().zip(actual.dims()) {
                    assert!(expected.compatible(*actual), "{} input={input:?} args={args:?}: predicted {expected:?}, actual {actual:?}",loaded.name);
                }
            }
            ShapeResult::Invalid(e) => {
                panic!("{} rejected a successful input={input:?}: {e}", loaded.name)
            }
            ShapeResult::Unknown => (),
        }
    }
}
fn compare(loaded: &LoadedAction, input: Payload, parameters: ActionArgs) {
    let shape = ValueShape::from_payload(&input);
    let output = loaded.process(Payload::WithArgs {
        payload: RBox::new(input),
        args: parameters.clone(),
    });
    assert!(
        !matches!(output, Payload::Error(_)),
        "{} input={shape:?} args={parameters:?}: {output:?}",
        loaded.name
    );
    for mask in 0..8 {
        verify(loaded, &partial(&shape, mask), &parameters, &output);
    }
    // Static arguments can be runtime variables. Removing an argument's value
    // must not make a stronger rank or dimension claim than the concrete call.
    for index in 0..parameters.named.len() {
        let mut dynamic = parameters.clone();
        dynamic.named[index].1 = "$runtime".into();
        verify(loaded, &shape, &dynamic, &output);
        verify(loaded, &partial(&shape, 7), &dynamic, &output);
    }
}
fn action(registry: &ActionRegistry, pack: &str, name: &str) -> std::sync::Arc<LoadedAction> {
    registry
        .get_or_load(&ActionIdentity::new(pack, "latest", name).unwrap())
        .unwrap()
}
#[test]
fn every_native_action_has_a_sound_complete_and_partial_prediction() {
    let registry = ActionRegistry::default();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("actions");
    let mut count = 0;
    for pack in std::fs::read_dir(root).unwrap() {
        let pack = pack.unwrap();
        if !pack.path().is_dir() {
            continue;
        }
        let pack_name = pack.file_name().to_string_lossy().into_owned();
        for entry in std::fs::read_dir(pack.path()).unwrap() {
            let entry = entry.unwrap();
            if !entry.path().join("Cargo.toml").exists() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            let loaded = action(&registry, &pack_name, &name);
            let input = if ["dot", "matmul", "outer", "cosine_similarity", "concat"]
                .contains(&name.as_str())
            {
                Payload::Composite(vec![tensor(&[2, 2]), tensor(&[2, 2])].into())
            } else if pack_name == "audio_basics" && name != "to_audio" {
                Payload::Audio(Audio::from_f32_planar(&[0.5; 34], 2, 48000).unwrap())
            } else if pack_name == "image_basics" {
                tensor(&[5, 7, 3])
            } else {
                tensor(&[2, 2])
            };
            let parameters = match name.as_str() {
                "reshape" => args(&[("shape", "[2,-1]")]),
                "permute" => args(&[("dims", "[1,0]")]),
                "repeat" => args(&[("repeats", "[2,3]")]),
                "stft" => args(&[("n_fft", "16"), ("hop_size", "4")]),
                "resample" => args(&[("from_rate", "48000"), ("to_rate", "24000")]),
                _ => args(&[]),
            };
            compare(&loaded, input, parameters);
            count += 1;
        }
    }
    assert_eq!(count, 86);
}
#[test]
fn reduction_empty_and_singleton_shapes_and_dynamic_axes_match_execution() {
    let registry = ActionRegistry::default();
    for name in [
        "argmax", "argmin", "max", "min", "mean", "std", "sum", "var", "norm", "cumsum",
    ] {
        let loaded = action(&registry, "tensor_stats", name);
        for dims in [
            vec![],
            vec![1],
            vec![3],
            vec![1, 1],
            vec![0, 3],
            vec![3, 0],
            vec![2, 0, 3],
        ] {
            for keepdim in ["false", "true"] {
                // argmax/min default to the last axis; other default reductions
                // are global. Do not reduce an empty axis for non-sum reducers.
                if !["argmax", "argmin"].contains(&name) || dims.last() != Some(&0) {
                    compare(&loaded, tensor(&dims), args(&[("keepdim", keepdim)]));
                }
                for (axis, length) in dims.iter().enumerate() {
                    if *length == 0 && !["sum", "cumsum"].contains(&name) {
                        continue;
                    }
                    compare(
                        &loaded,
                        tensor(&dims),
                        args(&[("axis", &axis.to_string()), ("keepdim", keepdim)]),
                    );
                }
            }
        }
    }
}
#[test]
fn empty_matrices_and_shape_preserving_kernels_match_execution() {
    let registry = ActionRegistry::default();
    for pack in ["math_basics", "nn_basics", "linalg_basics"] {
        let names: &[&str] = match pack {
            "math_basics" => &[
                "abs", "add", "clamp", "cos", "div", "exp", "log", "mul", "neg", "pow", "rsqrt",
                "sign", "sin", "sqrt", "sub", "tan",
            ],
            "nn_basics" => &[
                "gelu",
                "leaky_relu",
                "log_softmax",
                "relu",
                "sigmoid",
                "silu",
                "softmax",
                "tanh",
            ],
            _ => &["cholesky", "inv", "det", "trace", "qr"],
        };
        for name in names {
            let loaded = action(&registry, pack, name);
            for dims in [vec![0, 0], vec![0, 2, 2], vec![2, 0, 0]] {
                if (*name == "cholesky" || *name == "qr") && dims.len() != 2 {
                    continue;
                }
                compare(&loaded, tensor(&dims), args(&[]));
            }
        }
    }
    let loaded = action(&registry, "audio_basics", "to_audio");
    compare(&loaded, tensor(&[0, 2]), args(&[("channels", "2")]));
    let loaded = action(&registry, "tensor_basics", "reshape");
    compare(&loaded, tensor(&[0, 3]), args(&[("shape", "[2,-1]")]));
    assert!(matches!(
        loaded.output_result(
            &Shape::new([0.into(), Dimension::Unknown]),
            &args(&[("shape", "[2,3]")])
        ),
        ShapeResult::Invalid(_)
    ));
}
#[test]
fn image_layout_inference_and_transform_arguments_match_partial_predictions() {
    let registry = ActionRegistry::default();
    for dims in [
        vec![5, 7],
        vec![5, 7, 1],
        vec![5, 7, 3],
        vec![3, 5, 7],
        vec![2, 3, 5],
        vec![1, 2, 1],
    ] {
        for (name, parameters) in [
            (
                "crop",
                args(&[("x", "1"), ("y", "1"), ("width", "3"), ("height", "2")]),
            ),
            ("pad", args(&[("pad", "1")])),
            ("resize", args(&[("width", "4"), ("height", "3")])),
            ("crop", args(&[("width", "0")])),
            (
                "resize",
                args(&[
                    ("width", "4"),
                    ("height", "3"),
                    ("keep_aspect_ratio", "true"),
                ]),
            ),
            ("resize", args(&[("scale", "0.7")])),
            ("rotate", args(&[("angle", "45")])),
            ("rotate", args(&[("angle", "90")])),
            ("to_image", args(&[("color", "rgba"), ("layout", "chw")])),
            ("to_image", args(&[("color", "gray")])),
            ("to_tensor", args(&[("color", "rgb"), ("layout", "chw")])),
        ] {
            let loaded = action(
                &registry,
                if name == "to_tensor" {
                    "base"
                } else {
                    "image_basics"
                },
                name,
            );
            compare(&loaded, tensor(&dims), parameters);
        }
    }
}

#[test]
fn compound_shapes_cover_broadcasts_empty_axes_and_independent_unknown_dimensions() {
    let registry = ActionRegistry::default();
    for (pack, name, left, right, parameters) in [
        (
            "linalg_basics",
            "matmul",
            vec![2, 1, 3, 4],
            vec![1, 5, 4, 2],
            args(&[]),
        ),
        (
            "linalg_basics",
            "matmul",
            vec![0, 3, 4],
            vec![1, 4, 2],
            args(&[]),
        ),
        (
            "linalg_basics",
            "matmul",
            vec![2, 3, 0],
            vec![0, 4],
            args(&[]),
        ),
        ("linalg_basics", "outer", vec![0, 3], vec![2], args(&[])),
        ("linalg_basics", "dot", vec![0, 3], vec![0], args(&[])),
        (
            "nn_basics",
            "cosine_similarity",
            vec![2, 0],
            vec![2, 0],
            args(&[]),
        ),
        (
            "tensor_basics",
            "concat",
            vec![0, 3],
            vec![2, 3],
            args(&[("axis", "0")]),
        ),
        (
            "tensor_basics",
            "concat",
            vec![2, 0],
            vec![2, 3],
            args(&[("axis", "-1")]),
        ),
    ] {
        let loaded = action(&registry, pack, name);
        let input = Payload::Composite(vec![tensor(&left), tensor(&right)].into());
        let shape = ValueShape::from_payload(&input);
        let output = loaded.process(Payload::WithArgs {
            payload: RBox::new(input),
            args: parameters.clone(),
        });
        assert!(!matches!(output, Payload::Error(_)), "{name}: {output:?}");
        for a in 0..(1 << left.len()) {
            for b in 0..(1 << right.len()) {
                let partial = ValueShape::composite([partial(&shape[0], a), partial(&shape[1], b)]);
                verify(&loaded, &partial, &parameters, &output);
            }
        }
    }
}
#[test]
fn audio_lengths_channels_and_conversion_metadata_match_execution() {
    let registry = ActionRegistry::default();
    for channels in [1, 2] {
        for samples in [0, 3, 16, 17] {
            for name in [
                "biquad_filter",
                "compressor",
                "delay",
                "gain",
                "limiter",
                "noise_gate",
                "normalize",
                "stereo_widen",
                "resample",
                "stft",
                "to_pcm",
                "to_wav",
                "to_tensor",
            ] {
                let loaded = action(
                    &registry,
                    if name == "to_tensor" {
                        "base"
                    } else {
                        "audio_basics"
                    },
                    name,
                );
                let input = Payload::Audio(
                    Audio::from_f32_planar(&vec![0.5; channels * samples], channels, 48000)
                        .unwrap(),
                );
                let parameters = match name {
                    "resample" => args(&[("from_rate", "48000"), ("to_rate", "24000")]),
                    "stft" => args(&[("n_fft", "16"), ("hop_size", "4")]),
                    _ => args(&[]),
                };
                compare(&loaded, input, parameters);
            }
        }
    }
    let to_image = action(&registry, "image_basics", "to_image");
    let to_tensor = action(&registry, "base", "to_tensor");
    for layout in ["hwc", "chw"] {
        for color in ["gray", "rgb", "rgba"] {
            let image = to_image.process(Payload::WithArgs {
                payload: RBox::new(tensor(&[5, 7, 3])),
                args: args(&[("layout", layout), ("color", color)]),
            });
            assert!(matches!(image, Payload::Image(_)));
            compare(&to_tensor, image.clone(), args(&[]));
            for target in ["gray", "rgb", "rgba"] {
                compare(
                    &to_tensor,
                    image.clone(),
                    args(&[("color", target), ("layout", layout)]),
                );
            }
        }
    }
}
#[test]
fn known_invalid_empty_dimensions_are_rejected_statically() {
    let registry = ActionRegistry::default();
    let resample = action(&registry, "audio_basics", "resample");
    assert!(matches!(
        resample.output_result(&Shape::new([0, 3]), &args(&[])),
        ShapeResult::Invalid(_)
    ));
    let to_audio = action(&registry, "audio_basics", "to_audio");
    assert!(matches!(
        to_audio.output_value_result(
            &ValueShape::tensor(Shape::new([2, 3])),
            &args(&[("channels", "4")])
        ),
        Some(ValueShapeResult::Invalid(_))
    ));
    for name in [
        "argmax", "argmin", "max", "min", "mean", "std", "var", "norm",
    ] {
        let loaded = action(&registry, "tensor_stats", name);
        assert!(
            matches!(
                loaded.output_result(&Shape::new([2, 0]), &args(&[("axis", "1")])),
                ShapeResult::Invalid(_)
            ),
            "{name}"
        );
    }
    for name in [
        "blend",
        "color_adjust",
        "crop",
        "edge_detect",
        "flip",
        "gaussian_blur",
        "morphology",
        "pad",
        "resize",
        "rotate",
        "sharpen",
        "threshold",
        "to_image",
    ] {
        let loaded = action(&registry, "image_basics", name);
        assert!(
            matches!(
                loaded.output_result(
                    &Shape::new([Dimension::Unknown, 0.into(), 3.into()]),
                    &args(&[])
                ),
                ShapeResult::Invalid(_)
            ),
            "{name}"
        );
    }
}
