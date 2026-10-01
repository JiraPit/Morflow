//! Differential tests load the compiled native binaries through the verified
//! cache and compare their contracts against real execution.
use core_types::{ActionArgs, RBox, Shape, ShapeResult, Tuple2};
use pipeline::{ActionIdentity, ActionRegistry, Audio, Payload, RVec, Tensor};

fn args(values: &[(&str, &str)]) -> ActionArgs {
    ActionArgs {
        positional: RVec::new(),
        named: values
            .iter()
            .map(|(k, v)| Tuple2((*k).into(), (*v).into()))
            .collect(),
    }
}
fn dims(payload: &Payload) -> Option<&[usize]> {
    match payload.unwrap_payload() {
        Payload::Tensor(t) | Payload::Scalar(t) => Some(t.shape.as_slice()),
        Payload::Image(i) => Some(i.tensor.shape.as_slice()),
        Payload::Audio(a) => Some(a.tensor.shape.as_slice()),
        _ => None,
    }
}
fn tensor(shape: &[usize]) -> Payload {
    let n = shape.iter().product();
    Payload::from_tensor(Tensor::from_f32_shape(&vec![1.0; n], shape.to_vec()).unwrap())
}
fn compare(
    registry: &ActionRegistry,
    pack: &str,
    action: &str,
    input: Payload,
    values: &[(&str, &str)],
) {
    let id = ActionIdentity::new(pack, "latest", action).unwrap();
    let loaded = registry.get_or_load(&id).unwrap();
    let args = args(values);
    let prediction =
        loaded.output_result(&Shape::new(dims(&input).unwrap().iter().copied()), &args);
    assert!(
        !matches!(prediction, ShapeResult::Invalid(_)),
        "{id}: {prediction:?}"
    );
    let output = loaded.process(Payload::WithArgs {
        payload: RBox::new(input),
        args,
    });
    assert!(!matches!(output, Payload::Error(_)), "{id}: {output:?}");
    if let ShapeResult::Ok(expected) = prediction {
        let actual = dims(&output).unwrap();
        assert_eq!(expected.rank(), actual.len(), "{id}");
        for (e, a) in expected.dims().iter().zip(actual) {
            if !e.is_unknown() {
                assert_eq!(e, a, "{id}");
            }
        }
    }
}
#[test]
fn tensor_and_scalar_contracts_match_native_execution() {
    let registry = ActionRegistry::default();
    for action in [
        "abs", "add", "clamp", "cos", "div", "exp", "log", "mul", "neg", "pow", "rsqrt", "sign",
        "sin", "sqrt", "sub", "tan",
    ] {
        compare(&registry, "math_basics", action, tensor(&[2, 3]), &[]);
        compare(
            &registry,
            "math_basics",
            action,
            Payload::scalar_f32(1.0),
            &[],
        );
    }
    for (action, input, values) in [
        ("cast", vec![2, 3], vec![]),
        ("flatten", vec![2, 3, 4], vec![("start_dim", "1")]),
        ("permute", vec![2, 3, 4], vec![("dims", "[2,0,1]")]),
        ("repeat", vec![2, 3], vec![("repeats", "[0,2]")]),
        ("reshape", vec![2, 3], vec![("shape", "[3,-1]")]),
        ("roll", vec![2, 3], vec![("shift", "1"), ("axis", "1")]),
        ("squeeze", vec![1, 2, 1], vec![]),
        ("squeeze", vec![1], vec![]),
        ("transpose", vec![2, 3, 4], vec![]),
        ("unsqueeze", vec![2, 3], vec![("dim", "1")]),
    ] {
        compare(&registry, "tensor_basics", action, tensor(&input), &values);
    }
    compare(
        &registry,
        "tensor_basics",
        "unsqueeze",
        tensor(&[2, 3]),
        &[],
    );
    compare(
        &registry,
        "tensor_basics",
        "unsqueeze",
        tensor(&[2, 3]),
        &[("axis", "0"), ("dim", "1")],
    );
    for action in [
        "argmax", "argmin", "max", "min", "mean", "std", "sum", "var", "norm", "cumsum",
    ] {
        for values in [
            vec![],
            vec![("axis", "1")],
            vec![("axis", "-1"), ("keepdim", "true")],
        ] {
            compare(&registry, "tensor_stats", action, tensor(&[2, 3]), &values);
        }
    }
    for action in [
        "gelu",
        "layer_norm",
        "leaky_relu",
        "log_softmax",
        "relu",
        "rms_norm",
        "sigmoid",
        "silu",
        "softmax",
        "tanh",
    ] {
        compare(&registry, "nn_basics", action, tensor(&[2, 3]), &[]);
        compare(
            &registry,
            "nn_basics",
            action,
            Payload::scalar_f32(1.0),
            &[],
        );
    }
    for action in ["avg_pool2d", "max_pool2d"] {
        compare(
            &registry,
            "nn_basics",
            action,
            tensor(&[2, 4, 6]),
            &[("kernel_size", "2"), ("stride", "2")],
        );
    }
}
#[test]
fn matrix_contracts_include_batches_and_diagonal_offsets() {
    let registry = ActionRegistry::default();
    let matrix =
        || Payload::Tensor(Tensor::from_f32_shape(&[4.0, 1.0, 1.0, 4.0], vec![2, 2]).unwrap());
    for action in ["cholesky", "inv", "det", "trace"] {
        compare(&registry, "linalg_basics", action, matrix(), &[]);
    }
    for action in ["inv", "det", "trace"] {
        let input = Payload::Tensor(
            Tensor::from_f32_shape(&[4.0, 1.0, 1.0, 4.0, 4.0, 1.0, 1.0, 4.0], vec![2, 2, 2])
                .unwrap(),
        );
        compare(&registry, "linalg_basics", action, input, &[]);
    }
    compare(
        &registry,
        "linalg_basics",
        "diag",
        tensor(&[3, 4]),
        &[("diagonal", &isize::MIN.to_string())],
    );
    for shape in [vec![3], vec![3, 4]] {
        for offset in ["-1", "0", "1", "10"] {
            compare(
                &registry,
                "linalg_basics",
                "diag",
                tensor(&shape),
                &[("diagonal", offset)],
            );
        }
    }
}
#[test]
fn image_contracts_cover_layout_changes_and_arbitrary_rotation() {
    let registry = ActionRegistry::default();
    for shape in [vec![8, 10], vec![8, 10, 3], vec![3, 8, 10]] {
        for action in [
            "blend",
            "color_adjust",
            "edge_detect",
            "flip",
            "gaussian_blur",
            "morphology",
            "sharpen",
            "threshold",
        ] {
            compare(&registry, "image_basics", action, tensor(&shape), &[]);
        }
        compare(
            &registry,
            "image_basics",
            "crop",
            tensor(&shape),
            &[("x", "1"), ("y", "2"), ("width", "5"), ("height", "4")],
        );
        compare(
            &registry,
            "image_basics",
            "pad",
            tensor(&shape),
            &[("pad", "1")],
        );
        compare(
            &registry,
            "image_basics",
            "resize",
            tensor(&shape),
            &[("width", "5"), ("height", "3")],
        );
        compare(
            &registry,
            "image_basics",
            "resize",
            tensor(&shape),
            &[
                ("width", "5"),
                ("height", "5"),
                ("keep_aspect_ratio", "true"),
            ],
        );
        for angle in ["0", "45", "90", "180", "270"] {
            compare(
                &registry,
                "image_basics",
                "rotate",
                tensor(&shape),
                &[("angle", angle)],
            );
        }
        for color in ["gray", "rgb", "rgba"] {
            for layout in ["hwc", "chw"] {
                compare(
                    &registry,
                    "image_basics",
                    "to_image",
                    tensor(&shape),
                    &[("color", color), ("layout", layout)],
                );
            }
        }
    }
}
#[test]
fn audio_contracts_match_mono_and_stereo_execution() {
    let registry = ActionRegistry::default();
    for channels in [1, 2] {
        for action in [
            "biquad_filter",
            "compressor",
            "delay",
            "gain",
            "limiter",
            "noise_gate",
            "normalize",
            "stereo_widen",
        ] {
            let input = Audio::from_f32_planar(&vec![0.5; channels * 64], channels, 48000).unwrap();
            compare(
                &registry,
                "audio_basics",
                action,
                Payload::Audio(input),
                &[],
            );
        }
        for length in [8, 64] {
            let input =
                Audio::from_f32_planar(&vec![0.5; channels * length], channels, 48000).unwrap();
            compare(
                &registry,
                "audio_basics",
                "stft",
                Payload::Audio(input),
                &[("n_fft", "16"), ("hop_size", "4")],
            );
        }
        let input = Audio::from_f32_planar(&vec![0.5; channels * 101], channels, 48000).unwrap();
        compare(
            &registry,
            "audio_basics",
            "resample",
            Payload::Audio(input),
            &[("from_rate", "48000"), ("to_rate", "24000")],
        );
    }
}
#[test]
fn invalid_shapes_fail_before_execution_and_dynamic_arguments_are_unknown() {
    let registry = ActionRegistry::default();
    for (pack, action, shape, values) in [
        ("linalg_basics", "cholesky", vec![2, 3], vec![]),
        ("linalg_basics", "inv", vec![2, 3], vec![]),
        ("tensor_stats", "cumsum", vec![2, 3], vec![("axis", "-3")]),
        ("tensor_basics", "transpose", vec![], vec![]),
        (
            "tensor_basics",
            "permute",
            vec![2, 3],
            vec![("dims", "[0,0]")],
        ),
        ("tensor_basics", "unsqueeze", vec![2, 3], vec![("dim", "4")]),
        (
            "tensor_basics",
            "transpose",
            vec![2, 3],
            vec![("dim0", "4")],
        ),
        (
            "nn_basics",
            "max_pool2d",
            vec![2, 2],
            vec![("kernel_size", "4")],
        ),
        ("image_basics", "resize", vec![2, 3], vec![("width", "0")]),
    ] {
        let id = ActionIdentity::new(pack, "latest", action).unwrap();
        let loaded = registry.get_or_load(&id).unwrap();
        let args = args(&values);
        assert!(
            matches!(
                loaded.output_result(&Shape::new(shape.clone()), &args),
                ShapeResult::Invalid(_)
            ),
            "{id}"
        );
        assert!(
            matches!(
                loaded.process(Payload::WithArgs {
                    payload: RBox::new(tensor(&shape)),
                    args
                }),
                Payload::Error(_)
            ),
            "{id}"
        );
    }
    for (pack, action, key) in [
        ("tensor_basics", "transpose", "dim0"),
        ("tensor_stats", "sum", "axis"),
        ("image_basics", "resize", "width"),
    ] {
        let loaded = registry
            .get_or_load(&ActionIdentity::new(pack, "latest", action).unwrap())
            .unwrap();
        assert!(matches!(
            loaded.output_result(&Shape::new([2, 3]), &args(&[(key, "$dynamic")])),
            ShapeResult::Ok(_)
        ));
    }
}

#[test]
fn norm_positional_order_and_metadata_conversions_are_conservative() {
    let registry = ActionRegistry::default();
    let loaded = registry
        .get_or_load(&ActionIdentity::new("tensor_stats", "latest", "norm").unwrap())
        .unwrap();
    let args = ActionArgs {
        positional: vec!["2".into(), "1".into()].into(),
        named: RVec::new(),
    };
    let prediction = loaded.output_result(&Shape::new([2, 3]), &args);
    assert!(matches!(prediction,ShapeResult::Ok(ref shape) if shape.dims()==[2]));
    let output = loaded.process(Payload::WithArgs {
        payload: RBox::new(tensor(&[2, 3])),
        args,
    });
    assert_eq!(dims(&output).unwrap(), &[2]);
    let to_tensor = registry
        .get_or_load(&ActionIdentity::new("base", "latest", "to_tensor").unwrap())
        .unwrap();
    assert!(matches!(
        to_tensor.output_result(&Shape::new([2, 64]), &ActionArgs::default()),
        ShapeResult::Ok(ref shape) if shape.dims() == [2, 64]
    ));
    compare(
        &registry,
        "base",
        "to_tensor",
        Payload::Audio(Audio::from_f32_planar(&[0.0; 128], 2, 48000).unwrap()),
        &[],
    );
    let to_audio = registry
        .get_or_load(&ActionIdentity::new("audio_basics", "latest", "to_audio").unwrap())
        .unwrap();
    assert!(matches!(
        to_audio.output_result(&Shape::new([2, 64]), &ActionArgs::default()),
        ShapeResult::Ok(ref shape) if shape.dims() == [2, 64]
    ));
    compare(
        &registry,
        "audio_basics",
        "to_audio",
        tensor(&[2, 64]),
        &[("channels", "2"), ("layout", "planar")],
    );
}

#[test]
fn strided_image_views_follow_the_same_shape_contracts() {
    let registry = ActionRegistry::default();
    let view = || {
        let source = Tensor::from_f32_shape(&[1.0; 240], vec![8, 10, 3]).unwrap();
        let view = source.transpose(0, 1).unwrap();
        assert!(!view.is_contiguous());
        Payload::Tensor(view)
    };
    compare(
        &registry,
        "image_basics",
        "resize",
        view(),
        &[("width", "5"), ("height", "3")],
    );
    compare(
        &registry,
        "image_basics",
        "to_image",
        view(),
        &[("color", "rgba"), ("layout", "chw")],
    );
    compare(&registry, "image_basics", "pad", view(), &[("pad", "1")]);
    compare(
        &registry,
        "image_basics",
        "rotate",
        view(),
        &[("angle", "45")],
    );
    compare(
        &registry,
        "image_basics",
        "crop",
        view(),
        &[("width", "5"), ("height", "3")],
    );
}

#[test]
fn qr_validates_input_and_returns_correct_composite_components() {
    let registry = ActionRegistry::default();
    let loaded = registry
        .get_or_load(&ActionIdentity::new("linalg_basics", "latest", "qr").unwrap())
        .unwrap();
    assert!(matches!(
        loaded.output_result(&Shape::new([3]), &ActionArgs::default()),
        ShapeResult::Invalid(_)
    ));
    assert!(matches!(
        loaded.output_result(&Shape::new([3, 2]), &ActionArgs::default()),
        ShapeResult::Unknown
    ));
    match loaded.process(tensor(&[3, 2])) {
        Payload::Composite(items) => {
            assert_eq!(dims(&items[0]).unwrap(), &[3, 2]);
            assert_eq!(dims(&items[1]).unwrap(), &[2, 2]);
        }
        other => panic!("Expected QR components, got {other:?}"),
    }
}

#[test]
fn boundary_shapes_and_argument_precedence_match_execution() {
    let registry = ActionRegistry::default();
    for action in [
        "argmax", "argmin", "max", "min", "mean", "std", "sum", "var", "norm", "cumsum",
    ] {
        for shape in [vec![], vec![1], vec![1, 1], vec![2, 1]] {
            for values in [vec![], vec![("keepdim", "true")]] {
                compare(&registry, "tensor_stats", action, tensor(&shape), &values);
            }
        }
        compare(
            &registry,
            "tensor_stats",
            action,
            tensor(&[2, 3]),
            &[("axis", "1"), ("dim", "ignored")],
        );
    }
    for action in ["flatten", "squeeze", "unsqueeze"] {
        compare(&registry, "tensor_basics", action, tensor(&[]), &[]);
    }
    for shape in [vec![1, 1], vec![1, 2, 1], vec![2, 3, 2], vec![2, 3, 5]] {
        for action in ["crop", "pad", "resize", "rotate", "to_image"] {
            compare(&registry, "image_basics", action, tensor(&shape), &[]);
        }
        compare(
            &registry,
            "image_basics",
            "crop",
            tensor(&shape),
            &[("x", "100"), ("y", "100")],
        );
        compare(
            &registry,
            "image_basics",
            "rotate",
            tensor(&shape),
            &[("angle", "0.0005")],
        );
    }
}

#[test]
fn large_static_dimensions_and_opaque_outputs_are_checked() {
    let registry = ActionRegistry::default();
    for (action, shape, values) in [
        ("resize", vec![1, 16_777_217], vec![]),
        ("rotate", vec![100_000, 100_000], vec![("angle", "0.0005")]),
    ] {
        let loaded = registry
            .get_or_load(&ActionIdentity::new("image_basics", "latest", action).unwrap())
            .unwrap();
        assert!(
            matches!(loaded.output_result(&Shape::new(shape.clone()), &args(&values)), ShapeResult::Ok(ref result) if result.dims() == shape)
        );
    }
    compare(&registry, "base", "identity", tensor(&[2, 3]), &[]);
    for action in ["to_pcm", "to_wav"] {
        let loaded = registry
            .get_or_load(&ActionIdentity::new("audio_basics", "latest", action).unwrap())
            .unwrap();
        assert!(matches!(
            loaded.output_result(&Shape::new([2, 8]), &ActionArgs::default()),
            ShapeResult::Unknown
        ));
        let input = Audio::from_f32_planar(&[0.5; 16], 2, 48000).unwrap();
        assert!(
            matches!(loaded.process(Payload::Audio(input)), Payload::Data { ref buffer } if !buffer.is_empty())
        );
    }
    // The channel product can overflow even when the sample count fits.
    let loaded = registry
        .get_or_load(&ActionIdentity::new("audio_basics", "latest", "resample").unwrap())
        .unwrap();
    assert!(matches!(
        loaded.output_result(
            &Shape::new([4, usize::MAX / 8]),
            &args(&[("from_rate", "1"), ("to_rate", "4")])
        ),
        ShapeResult::Invalid(_)
    ));
}
