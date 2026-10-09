//! Requires locally packaged image_opencv actions and a shared OpenCV bridge.
use core_types::{
    ActionArgs, InputDescriptor, Payload, RBox, ShapeCheckResult, Tensor, TensorDType, Tuple2,
};
use morflow::{ActionIdentity, ActionRegistry, Morflow};
fn plugins() -> std::sync::Arc<morflow::plugins::PluginSet> {
    static PLUGINS: std::sync::OnceLock<std::sync::Arc<morflow::plugins::PluginSet>> =
        std::sync::OnceLock::new();
    PLUGINS
        .get_or_init(|| {
            morflow::plugins::PluginSet::prepare(
                &[parser::ast::PluginDecl {
                    name: "opencv-bridge".into(),
                    version: "0.1.2".into(),
                }],
                &morflow::plugins::search_paths(),
            )
            .unwrap()
        })
        .clone()
}
fn args(pairs: &[(&str, &str)]) -> ActionArgs {
    ActionArgs {
        positional: Default::default(),
        named: pairs
            .iter()
            .map(|(k, v)| Tuple2((*k).into(), (*v).into()))
            .collect(),
    }
}
fn execute(
    registry: &ActionRegistry,
    pack: &str,
    name: &str,
    input: Tensor,
    arguments: ActionArgs,
) -> Tensor {
    let action = registry
        .get_or_load(
            &ActionIdentity::new(
                pack,
                if pack == "image_basics" {
                    "latest"
                } else {
                    "0.1.2"
                },
                name,
            )
            .unwrap(),
        )
        .unwrap();
    let action = if pack == "image_opencv" {
        action.with_plugins(plugins()).unwrap()
    } else {
        action
    };
    let descriptor = InputDescriptor::from_payload(&Payload::Tensor(input.clone()));
    let ShapeCheckResult::Ready { output, .. } = action.shapecheck(descriptor, arguments.clone())
    else {
        panic!("{name} did not check");
    };
    let result = action.process(Payload::WithArgs {
        payload: RBox::new(Payload::Tensor(input)),
        args: arguments,
    });
    output.verify(&result).unwrap();
    let Payload::Tensor(tensor) = result else {
        panic!("{name} did not return Tensor");
    };
    tensor
}
#[test]
fn filters_match_basics_for_all_modes_dtypes_and_preserve_inputs() {
    let registry = ActionRegistry::default();
    for dtype in [TensorDType::F32, TensorDType::U8] {
        let input = if dtype == TensorDType::F32 {
            Tensor::from_f32_vec(
                (0..7 * 9 * 3)
                    .map(|n| ((n * 17) % 251) as f32 / 255.)
                    .collect(),
                vec![7, 9, 3],
            )
            .unwrap()
        } else {
            Tensor::from_rvec_u8(
                (0..7 * 9 * 3).map(|n| ((n * 17) % 251) as u8).collect(),
                vec![7, 9, 3],
                dtype,
            )
            .unwrap()
        };
        let original = input.to_contiguous_bytes();
        let mut cases = vec![
            ("gaussian_blur", args(&[("sigma", "1.2"), ("radius", "2")])),
            ("gaussian_blur", args(&[("mode", "box"), ("radius", "2")])),
            ("sharpen", args(&[("strength", "0.7"), ("sigma", "1.2")])),
        ];
        for mode in ["sobel", "sobel_x", "sobel_y", "prewitt", "laplacian"] {
            cases.push(("edge_detect", args(&[("mode", mode), ("strength", "0.5")])));
        }
        for mode in ["dilate", "erode", "open", "close", "gradient"] {
            for shape in ["rect", "cross", "ellipse"] {
                cases.push((
                    "morphology",
                    args(&[
                        ("op", mode),
                        ("shape", shape),
                        ("iterations", "2"),
                        ("kernel_size", "4"),
                    ]),
                ));
            }
        }
        for (name, arguments) in cases {
            let expected = execute(
                &registry,
                "image_basics",
                name,
                input.clone(),
                arguments.clone(),
            );
            let actual = execute(&registry, "image_opencv", name, input.clone(), arguments);
            let tolerance = if dtype == TensorDType::U8 { 1.01 } else { 2e-5 };
            for (a, b) in actual.to_vec_f32().iter().zip(expected.to_vec_f32()) {
                assert!((a - b).abs() <= tolerance, "{name} {dtype:?}: {a} vs {b}");
            }
        }
        assert_eq!(input.to_contiguous_bytes(), original);
    }
}
#[test]
fn chw_strided_and_singleton_channel_outputs_match_hwc() {
    let registry = ActionRegistry::default();
    let hwc = Tensor::from_f32_vec(
        (0..6 * 8 * 3).map(|n| n as f32 / 200.).collect(),
        vec![6, 8, 3],
    )
    .unwrap();
    let chw = hwc.permute(&[2, 0, 1]).unwrap();
    for name in [
        "resize",
        "gaussian_blur",
        "morphology",
        "edge_detect",
        "sharpen",
    ] {
        let arguments = if name == "resize" {
            args(&[("width", "5"), ("height", "7")])
        } else {
            ActionArgs::default()
        };
        let expected = execute(
            &registry,
            "image_opencv",
            name,
            hwc.clone(),
            arguments.clone(),
        );
        let actual = execute(&registry, "image_opencv", name, chw.clone(), arguments)
            .permute(&[1, 2, 0])
            .unwrap();
        assert_eq!(actual.to_vec_f32(), expected.to_vec_f32());
        let singleton = Tensor::from_f32_vec(vec![0.5; 48], vec![6, 8, 1]).unwrap();
        let result = execute(
            &registry,
            "image_opencv",
            name,
            singleton,
            ActionArgs::default(),
        );
        assert_eq!(result.shape.as_slice(), [6, 8, 1]);
    }
}
#[test]
fn resize_filters_dimensions_and_nearest_pixel_selection() {
    let registry = ActionRegistry::default();
    let input = Tensor::from_f32_vec((0..16).map(|n| n as f32).collect(), vec![4, 4]).unwrap();
    let nearest = execute(
        &registry,
        "image_opencv",
        "resize",
        input.clone(),
        args(&[("width", "2"), ("height", "2"), ("filter", "nearest")]),
    );
    assert_eq!(nearest.to_vec_f32(), [0., 2., 8., 10.]);
    for filter in ["bilinear", "bicubic", "area"] {
        let constant = Tensor::from_f32_vec(vec![0.5; 48], vec![6, 8, 1]).unwrap();
        let result = execute(
            &registry,
            "image_opencv",
            "resize",
            constant,
            args(&[("width", "4"), ("height", "3"), ("filter", filter)]),
        );
        assert_eq!(result.shape.as_slice(), [3, 4, 1]);
        assert!(result.to_vec_f32().iter().all(|x| (*x - 0.5).abs() < 1e-6));
    }
    let result = execute(
        &registry,
        "image_opencv",
        "resize",
        input,
        args(&[("scale", "0.5")]),
    );
    assert_eq!(result.shape.as_slice(), [2, 2]);
}
#[test]
fn invalid_arguments_and_dimensions_fail_during_shapecheck() {
    let registry = ActionRegistry::default();
    for (name, arguments) in [
        ("gaussian_blur", args(&[("sigma", "NaN")])),
        ("morphology", args(&[("kernel_size", "9999999999")])),
        ("resize", args(&[("width", "2147483648")])),
    ] {
        let action = registry
            .get_or_load(&ActionIdentity::new("image_opencv", "0.1.2", name).unwrap())
            .unwrap();
        let descriptor = InputDescriptor::from_payload(&Payload::Tensor(
            Tensor::from_f32_vec(vec![0.; 36], vec![6, 6]).unwrap(),
        ));
        assert!(matches!(
            action.shapecheck(descriptor, arguments),
            ShapeCheckResult::Invalid { .. }
        ));
    }
}
#[test]
fn missing_system_library_is_deferred_until_execution() {
    use morflow::artifact::{atomic_write, digest, receipt_path};
    use morflow::plugins::{PluginIdentity, PluginReceipt};
    const FLAG: &str = "MORFLOW_PLUGIN_TEST_CHILD";
    if std::env::var_os(FLAG).is_none() {
        let root = tempfile::tempdir().unwrap();
        let id = PluginIdentity::new("opencv-bridge", "0.1.2").unwrap();
        let bytes = b"verified plugin fixture that cannot be dynamically loaded";
        atomic_write(&id.path(root.path()), bytes).unwrap();
        let receipt = PluginReceipt {
            identity: id.clone(),
            concrete_version: "0.1.2".into(),
            repository: "test".into(),
            sha256: digest(bytes),
        };
        atomic_write(
            &receipt_path(&id.path(root.path())),
            &serde_json::to_vec(&receipt).unwrap(),
        )
        .unwrap();
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "missing_system_library_is_deferred_until_execution",
                "--nocapture",
            ])
            .env(FLAG, "1")
            .env("MORFLOW_PLUGINS_PATH", root.path())
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        return;
    }
    let source="plugin opencv-bridge/0.1.2\nfrom image_opencv/0.1.2 import gaussian_blur\naccept Tensor[6,8,3] $x\n$x >> gaussian_blur >> emit";
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("offline.morf");
    std::fs::write(&path, source).unwrap();
    assert!(morflow::check::check_pipeline(&path, None).is_ok());
    for mut pipeline in [
        Morflow::from_str(source).unwrap(),
        Morflow::load(&path).unwrap(),
    ] {
        let error = pipeline
            .run(Payload::Tensor(
                Tensor::from_f32_vec(vec![0.5; 144], vec![6, 8, 3]).unwrap(),
            ))
            .unwrap_err();
        assert!(error.to_string().contains("Cannot load plugin"), "{error}");
    }
}
#[test]
fn mixed_pack_example_loads_once_and_runs_repeatedly() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let mut pipeline = Morflow::load(root.join("examples/python/opencv/pipeline.morf")).unwrap();
    let tensor = Tensor::from_f32_vec(vec![0.5; 480 * 640 * 3], vec![480, 640, 3]).unwrap();
    for _ in 0..3 {
        let result = pipeline.run(Payload::Tensor(tensor.clone())).unwrap();
        let Payload::Tensor(output) = result.single().unwrap() else {
            panic!("Expected Tensor");
        };
        assert_eq!(output.shape.as_slice(), [320, 240, 3]);
    }
}

#[test]
fn dynamic_shapes_and_arguments_remain_deferred() {
    let registry = ActionRegistry::default();
    for name in [
        "gaussian_blur",
        "morphology",
        "edge_detect",
        "sharpen",
        "resize",
    ] {
        let action = registry
            .get_or_load(&ActionIdentity::new("image_opencv", "0.1.2", name).unwrap())
            .unwrap();
        let mut descriptor = InputDescriptor::from_payload(&Payload::Tensor(
            Tensor::from_f32_vec(vec![0.; 144], vec![6, 8, 3]).unwrap(),
        ));
        descriptor.value = core_types::ValueShape::Leaf {
            kind: core_types::DataType::Tensor,
            shape: Some(core_types::Shape::new([
                core_types::Dimension::Unknown,
                core_types::Dimension::Unknown,
                3.into(),
            ]))
            .into(),
        };
        assert!(matches!(
            action.shapecheck(descriptor, ActionArgs::default()),
            ShapeCheckResult::Deferred { .. }
        ));
    }
    let action = registry
        .get_or_load(&ActionIdentity::new("image_opencv", "0.1.2", "gaussian_blur").unwrap())
        .unwrap();
    let descriptor = InputDescriptor::from_payload(&Payload::Tensor(
        Tensor::from_f32_vec(vec![0.; 36], vec![6, 6]).unwrap(),
    ));
    assert!(matches!(
        action.shapecheck(descriptor, args(&[("sigma", "$sigma")])),
        ShapeCheckResult::Deferred { .. }
    ));
}

#[test]
fn rotation_quarter_turns_match_basics_and_arbitrary_angles_match_predicted_canvas() {
    let registry = ActionRegistry::default();
    for dtype in [TensorDType::F32, TensorDType::U8] {
        let input = if dtype == TensorDType::F32 {
            Tensor::from_f32_vec((0..6 * 8 * 3).map(|n| n as f32).collect(), vec![6, 8, 3]).unwrap()
        } else {
            Tensor::from_rvec_u8(
                (0..6 * 8 * 3).map(|n| n as u8).collect(),
                vec![6, 8, 3],
                dtype,
            )
            .unwrap()
        };
        for angle in ["0", "90", "180", "270", "-90"] {
            let arguments = args(&[("angle", angle), ("expand", "false")]);
            let expected = execute(
                &registry,
                "image_basics",
                "rotate",
                input.clone(),
                arguments.clone(),
            );
            let actual = execute(
                &registry,
                "image_opencv",
                "rotate",
                input.clone(),
                arguments.clone(),
            );
            assert_eq!(actual.to_vec_f32(), expected.to_vec_f32());
            let chw = execute(
                &registry,
                "image_opencv",
                "rotate",
                input.permute(&[2, 0, 1]).unwrap(),
                arguments,
            )
            .permute(&[1, 2, 0])
            .unwrap();
            assert_eq!(chw.to_vec_f32(), actual.to_vec_f32());
        }
        for expand in ["true", "false"] {
            let result = execute(
                &registry,
                "image_opencv",
                "rotate",
                input.clone(),
                args(&[("angle", "33"), ("expand", expand), ("fill", "0.25")]),
            );
            assert_eq!(result.shape[2], 3);
            if expand == "false" {
                assert_eq!(result.shape.as_slice(), [6, 8, 3]);
            }
        }
    }
}

#[test]
fn native_dimension_and_channel_limits_are_checked_without_allocating_images() {
    let registry = ActionRegistry::default();
    for (name, shape, arguments) in [
        ("rotate", vec![30000, 30000], args(&[("angle", "45")])),
        ("rotate", vec![32767, 8], ActionArgs::default()),
        ("resize", vec![6, 8, 5], ActionArgs::default()),
    ] {
        let action = registry
            .get_or_load(&ActionIdentity::new("image_opencv", "0.1.2", name).unwrap())
            .unwrap();
        let descriptor = InputDescriptor {
            value: core_types::ValueShape::Leaf {
                kind: core_types::DataType::Tensor,
                shape: Some(core_types::Shape::new(shape)).into(),
            },
            metadata: core_types::shapecheck::Metadata::Tensor {
                dtype: TensorDType::F32,
            },
        };
        assert!(matches!(
            action.shapecheck(descriptor, arguments),
            ShapeCheckResult::Invalid { .. }
        ));
    }
}

#[test]
fn boundary_blur_radii_and_identity_morphology_match_basics() {
    let registry = ActionRegistry::default();
    let input = Tensor::from_f32_vec(
        (0..6 * 8).map(|n| (n % 5) as f32 / 5.).collect(),
        vec![6, 8],
    )
    .unwrap();
    for (name, arguments) in [
        ("gaussian_blur", args(&[("sigma", "0.33333334")])),
        ("sharpen", args(&[("sigma", "0.33333334")])),
        ("gaussian_blur", args(&[("radius", "0")])),
        ("sharpen", args(&[("radius", "0")])),
    ] {
        let expected = execute(
            &registry,
            "image_basics",
            name,
            input.clone(),
            arguments.clone(),
        );
        let actual = execute(&registry, "image_opencv", name, input.clone(), arguments);
        for (a, b) in actual.to_vec_f32().iter().zip(expected.to_vec_f32()) {
            assert!((a - b).abs() < 2e-5, "{name}: {a} vs {b}");
        }
    }
    for op in ["dilate", "erode", "open", "close", "gradient"] {
        for size in ["0", "1"] {
            let arguments = args(&[("op", op), ("kernel_size", size)]);
            let expected = execute(
                &registry,
                "image_basics",
                "morphology",
                input.clone(),
                arguments.clone(),
            );
            let actual = execute(
                &registry,
                "image_opencv",
                "morphology",
                input.clone(),
                arguments,
            );
            assert_eq!(actual.to_vec_f32(), expected.to_vec_f32());
        }
    }
}

#[test]
fn arbitrary_rotation_matches_inverse_mapping_on_a_linear_image() {
    let registry = ActionRegistry::default();
    let (height, width) = (16, 20);
    let input = Tensor::from_f32_vec(
        (0..height)
            .flat_map(|y| (0..width).map(move |x| x as f32 + y as f32 * 0.25))
            .collect(),
        vec![height, width],
    )
    .unwrap();
    for angle in ["33", "-33", "123"] {
        for expand in ["true", "false"] {
            let output = execute(
                &registry,
                "image_opencv",
                "rotate",
                input.clone(),
                args(&[("angle", angle), ("expand", expand), ("fill", "-0.75")]),
            );
            let (oh, ow) = (output.shape[0], output.shape[1]);
            let raw = angle.parse::<f32>().unwrap();
            let normalized = ((raw % 360.) + 360.) % 360.;
            let radians = f64::from(normalized).to_radians();
            let (s, c) = radians.sin_cos();
            let mut interior = 0;
            let mut exterior = 0;
            for (index, value) in output.to_vec_f32().into_iter().enumerate() {
                let dx = (index % ow) as f64 - ow as f64 / 2.;
                let dy = (index / ow) as f64 - oh as f64 / 2.;
                let x = dx * c + dy * s + width as f64 / 2.;
                let y = -dx * s + dy * c + height as f64 / 2.;
                if x >= 1. && x < (width - 2) as f64 && y >= 1. && y < (height - 2) as f64 {
                    // A bilinear sampler reproduces a linear ramp. OpenCV 4 uses
                    // fractional coordinates quantized to 1/32 pixel.
                    assert!(
                        (f64::from(value) - (x + y * 0.25)).abs() < 0.025,
                        "angle={angle}, expand={expand}: {value} at ({x},{y})"
                    );
                    interior += 1;
                } else if x < -1. || x > width as f64 || y < -1. || y > height as f64 {
                    assert_eq!(value, -0.75);
                    exterior += 1;
                }
            }
            assert!(interior > 20 && exterior > 0);
        }
    }
}

#[test]
fn offset_and_negative_stride_views_match_materialized_images() {
    let registry = ActionRegistry::default();
    let original = Tensor::from_f32_vec(
        (0..8 * 12 * 3).map(|n| n as f32 / 300.).collect(),
        vec![8, 12, 3],
    )
    .unwrap();
    let view = original
        .slice_range(0, 1, 7, 1)
        .unwrap()
        .slice_range(1, 1, 11, 2)
        .unwrap();
    let mut reversed = view.clone();
    reversed.byte_offset = (reversed.byte_offset as isize
        + (reversed.shape[1] - 1) as isize * reversed.strides[1])
        as usize;
    reversed.strides[1] = -reversed.strides[1];
    for input in [view, reversed] {
        let contiguous = Tensor::from_f32_vec(input.to_vec_f32(), input.shape.to_vec()).unwrap();
        for name in [
            "resize",
            "rotate",
            "gaussian_blur",
            "morphology",
            "edge_detect",
            "sharpen",
        ] {
            let expected = execute(
                &registry,
                "image_opencv",
                name,
                contiguous.clone(),
                ActionArgs::default(),
            );
            let actual = execute(
                &registry,
                "image_opencv",
                name,
                input.clone(),
                ActionArgs::default(),
            );
            assert_eq!(actual.shape, expected.shape);
            assert_eq!(actual.to_vec_f32(), expected.to_vec_f32(), "{name}");
        }
    }
}

#[test]
fn shape_preserving_filters_support_more_than_four_channels() {
    let registry = ActionRegistry::default();
    for channels in [5, 8] {
        let input = Tensor::from_f32_vec(
            (0..6 * 8 * channels)
                .map(|n| (n % 53) as f32 / 53.)
                .collect(),
            vec![6, 8, channels],
        )
        .unwrap();
        for (name, arguments) in [
            ("gaussian_blur", args(&[("sigma", "0.7")])),
            ("sharpen", args(&[("sigma", "0.7")])),
            ("morphology", args(&[("op", "erode")])),
            ("edge_detect", args(&[("mode", "sobel_x")])),
            ("edge_detect", args(&[("mode", "sobel_y")])),
            ("edge_detect", args(&[("mode", "laplacian")])),
            ("edge_detect", args(&[("mode", "prewitt")])),
            ("edge_detect", args(&[("mode", "sobel")])),
        ] {
            let output = execute(&registry, "image_opencv", name, input.clone(), arguments);
            assert_eq!(output.shape, input.shape);
            assert!(output.to_vec_f32().iter().all(|v| v.is_finite()));
        }
    }
}

#[cfg(unix)]
#[test]
fn pipeline_reload_selects_refreshed_plugin_and_existing_pipeline_retains_old_one() {
    use morflow::artifact::{atomic_write, digest, receipt_path};
    use morflow::plugins::{PluginIdentity, PluginReceipt};
    const FLAG: &str = "MORFLOW_PLUGIN_RELOAD_CHILD";
    if std::env::var_os(FLAG).is_none() {
        let root = tempfile::tempdir().unwrap();
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "pipeline_reload_selects_refreshed_plugin_and_existing_pipeline_retains_old_one",
                "--nocapture",
            ])
            .env(FLAG, "1")
            .env("MORFLOW_PLUGINS_PATH", root.path())
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        return;
    }
    let root = std::path::PathBuf::from(std::env::var_os("MORFLOW_PLUGINS_PATH").unwrap());
    let id = PluginIdentity::new("opencv-bridge", "latest").unwrap();
    let publish = |number, version: &str| {
        let source = root.join(format!("{number}.c"));
        let binary = root.join(format!("{number}.so"));
        std::fs::write(&source,format!("#include <stddef.h>\nint morflow_opencv_process(const void* input, void* output, int h, int w, int c, int depth, int oh, int ow, const void* options, char* error, size_t error_length) {{ for (size_t i=0;i<(size_t)oh*ow*c;i++) ((float*)output)[i]={number}.0f; return 0; }}")).unwrap();
        assert!(std::process::Command::new("cc")
            .args(["-shared", "-fPIC"])
            .arg(source)
            .arg("-o")
            .arg(&binary)
            .status()
            .unwrap()
            .success());
        let bytes = std::fs::read(binary).unwrap();
        let receipt = PluginReceipt {
            identity: id.clone(),
            concrete_version: version.into(),
            repository: "test".into(),
            sha256: digest(&bytes),
        };
        let _guard = morflow::artifact::CacheGuard::acquire(&root, &id.to_string()).unwrap();
        atomic_write(&id.path(&root), &bytes).unwrap();
        atomic_write(
            &receipt_path(&id.path(&root)),
            &serde_json::to_vec(&receipt).unwrap(),
        )
        .unwrap();
    };
    let source="plugin opencv-bridge/latest\nfrom image_opencv/0.1.2 import gaussian_blur\naccept Tensor[6,8,3] $x\n$x >> gaussian_blur >> emit\n";
    publish(1, "0.1.0");
    let mut old = Morflow::from_str(source).unwrap();
    publish(2, "0.1.2");
    let mut new = Morflow::from_str(source).unwrap();
    let run = |pipeline: &mut morflow::MorflowPipeline| {
        pipeline
            .run(Payload::Tensor(
                Tensor::from_f32_vec(vec![0.5; 144], vec![6, 8, 3]).unwrap(),
            ))
            .unwrap()
            .into_single()
            .unwrap()
            .as_tensor()
            .unwrap()
            .to_vec_f32()
    };
    // Both plugins were selected before either process call opened the library.
    assert!(run(&mut old).iter().all(|v| *v == 1.));
    assert!(run(&mut new).iter().all(|v| *v == 2.));
    assert!(run(&mut old).iter().all(|v| *v == 1.));
}

#[test]
fn plugin_actions_require_a_declaration_before_execution() {
    let source =
        "from image_opencv/0.1.2 import resize\naccept Tensor[6,8,3] $x\n$x >> resize(4,3) >> emit";
    match Morflow::from_str(source) {
        Err(error) => assert!(error.to_string().contains("undeclared"), "{error}"),
        Ok(_) => panic!("Missing plugin declaration must be rejected at load time"),
    }
}

#[test]
fn action_plans_only_contain_their_own_parameters() {
    let registry = ActionRegistry::default();
    let tensor = Tensor::from_f32_vec(vec![0.; 6 * 8 * 3], vec![6, 8, 3]).unwrap();
    for (name, keys) in [
        ("resize", vec!["mode"]),
        ("gaussian_blur", vec!["mode", "radius", "sigma"]),
        ("morphology", vec!["mode", "radius", "iterations", "shape"]),
        ("edge_detect", vec!["mode", "strength"]),
        ("sharpen", vec!["radius", "sigma", "strength"]),
        ("rotate", vec!["angle", "fill"]),
    ] {
        let action = registry
            .get_or_load(&ActionIdentity::new("image_opencv", "0.1.2", name).unwrap())
            .unwrap();
        let descriptor = InputDescriptor::from_payload(&Payload::Tensor(tensor.clone()));
        let ShapeCheckResult::Ready { prepared, .. } =
            action.shapecheck(descriptor, ActionArgs::default())
        else {
            panic!("{name} did not check");
        };
        for key in &keys {
            assert!(
                prepared
                    .fields
                    .iter()
                    .any(|Tuple2(k, _)| k.as_str() == *key),
                "{name}: missing {key}"
            );
        }
        let mut actual_keys: Vec<_> = prepared
            .fields
            .iter()
            .map(|Tuple2(k, _)| k.as_str())
            .collect();
        let mut expected_keys = keys;
        actual_keys.sort_unstable();
        expected_keys.sort_unstable();
        assert_eq!(
            actual_keys, expected_keys,
            "{name} contains unrelated parameters"
        );
        let unrelated = match name {
            "gaussian_blur" => "strength",
            "sharpen" => "angle",
            _ => "sigma",
        };
        let descriptor = InputDescriptor::from_payload(&Payload::Tensor(tensor.clone()));
        assert!(
            matches!(
                action.shapecheck(descriptor, args(&[(unrelated, "NaN")])),
                ShapeCheckResult::Ready { .. }
            ),
            "{name} parsed unrelated {unrelated}"
        );
        assert!(!prepared.fields.iter().any(|Tuple2(k, _)| k == "operation"));
        if name == "resize" {
            assert!(!prepared
                .fields
                .iter()
                .any(|Tuple2(k, _)| k == "sigma" || k == "radius"));
        }
    }
}
