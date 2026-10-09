use core_types::{
    ActionArgs, DataType, Dimension, PType, Shape, ShapeSpec, Tuple2, ValueShape, ValueShapeResult,
};
use morflow::{ActionIdentity, ActionRegistry, Morflow, Payload, Tensor};

fn args(values: &[(&str, &str)]) -> ActionArgs {
    ActionArgs {
        positional: vec![].into(),
        named: values
            .iter()
            .map(|(k, v)| Tuple2((*k).into(), (*v).into()))
            .collect(),
    }
}
fn action(
    registry: &ActionRegistry,
    pack: &str,
    name: &str,
) -> std::sync::Arc<morflow::LoadedAction> {
    registry
        .get_or_load(&ActionIdentity::new(pack, "latest", name).unwrap())
        .unwrap()
}
fn prediction(
    registry: &ActionRegistry,
    pack: &str,
    name: &str,
    input: Shape,
    args: ActionArgs,
) -> ValueShapeResult {
    let loaded = action(registry, pack, name);
    let kind = if loaded.input_type.intersects(DataType::Tensor) {
        DataType::Tensor
    } else {
        loaded.input_type
    };
    loaded
        .output_value_result(
            &ValueShape::Leaf {
                kind,
                shape: Some(input).into(),
            },
            &args,
        )
        .unwrap()
}

#[test]
fn every_rank_determined_action_preserves_rank_with_no_known_lengths() {
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
            if [
                "identity",
                "to_wav",
                "to_pcm",
                "squeeze",
                "resample",
                "stft",
                "to_audio",
                "to_image",
                "to_tensor",
            ]
            .contains(&name.as_str())
            {
                continue;
            }
            let input = Shape::unknown(if pack_name == "audio_basics" { 1 } else { 2 });
            let parameters = match name.as_str() {
                "reshape" => args(&[("shape", "[2,-1]")]),
                "permute" => args(&[("dims", "[1,0]")]),
                "repeat" => args(&[("repeats", "[2,3]")]),
                _ => args(&[]),
            };
            let result = prediction(&registry, &pack_name, &name, input, parameters);
            let ValueShapeResult::Ok(shape) = result else {
                panic!("{pack_name}/{name}: {result:?}");
            };
            if name == "qr" {
                assert_eq!(shape[0].shape().unwrap().rank(), 2);
                assert_eq!(shape[1].shape().unwrap().rank(), 2);
            } else {
                let expected_rank = match (pack_name.as_str(), name.as_str()) {
                    ("audio_basics", _) => 1,
                    ("tensor_stats", "cumsum") => 2,
                    ("tensor_stats", "argmax" | "argmin") => 1,
                    ("tensor_stats", _) => 0,
                    ("linalg_basics" | "linalg_blas", "det" | "trace") => 0,
                    ("linalg_basics" | "linalg_blas", "dot" | "diag")
                    | ("tensor_basics", "flatten") => 1,
                    ("tensor_basics", "unsqueeze") => 3,
                    _ => 2,
                };
                assert_eq!(
                    shape.shape().unwrap().rank(),
                    expected_rank,
                    "{pack_name}/{name}"
                );
            }
            count += 1;
        }
    }
    assert_eq!(count, 93);
}

#[test]
fn genuinely_length_dependent_ranks_stay_unknown() {
    let registry = ActionRegistry::default();
    for (pack, name, rank) in [
        ("tensor_basics", "squeeze", 3),
        ("audio_basics", "resample", 2),
        ("audio_basics", "stft", 1),
        ("audio_basics", "to_audio", 2),
        ("image_basics", "to_image", 3),
        ("base", "to_tensor", 3),
    ] {
        assert!(
            matches!(
                prediction(&registry, pack, name, Shape::unknown(rank), args(&[])),
                ValueShapeResult::Ok(ref shape) if shape.shape().is_none()
            ),
            "{pack}/{name}"
        );
    }
}

#[test]
fn unknown_arguments_preserve_rank_when_they_do_not_determine_it() {
    let registry = ActionRegistry::default();
    for (pack, name, key, rank) in [
        ("tensor_basics", "transpose", "dim0", 2),
        ("tensor_basics", "permute", "dims", 2),
        ("tensor_basics", "unsqueeze", "axis", 3),
        ("tensor_basics", "roll", "axis", 2),
        ("tensor_stats", "sum", "axis", 1),
        ("tensor_stats", "norm", "axis", 1),
        ("nn_basics", "softmax", "axis", 2),
        ("nn_basics", "avg_pool2d", "kernel", 2),
        ("image_basics", "resize", "width", 2),
        ("linalg_basics", "diag", "k", 1),
    ] {
        let ValueShapeResult::Ok(shape) = prediction(
            &registry,
            pack,
            name,
            Shape::unknown(2),
            args(&[(key, "$value")]),
        ) else {
            panic!("{pack}/{name}");
        };
        assert_eq!(shape.shape().unwrap().rank(), rank, "{pack}/{name}");
    }
    let ValueShapeResult::Ok(shape) = prediction(
        &registry,
        "tensor_stats",
        "norm",
        Shape::unknown(2),
        ActionArgs {
            positional: vec!["$p".into()].into(),
            named: vec![].into(),
        },
    ) else {
        panic!()
    };
    assert_eq!(shape.shape().unwrap().rank(), 0);
}

#[test]
fn zero_dimensions_roundtrip_and_validate_as_real_lengths() {
    let shape = Shape::new([Dimension::Known(0), Dimension::Unknown, Dimension::Known(3)]);
    assert_eq!(shape.element_count(), Some(0));
    assert_eq!(shape.to_string(), "[0, *, 3]");
    let ty = PType::Tensor(ShapeSpec::from(shape.clone()));
    assert_eq!(ValueShape::from_ptype(&ty).shape(), Some(&shape));
    assert_eq!(ValueShape::from_ptype(&ty).to_ptype(), ty);
    let expected = ValueShape::tensor(Shape::new([0, 3]));
    assert!(expected
        .verify(&Payload::Tensor(
            Tensor::from_f32_shape(&[], vec![0, 3]).unwrap()
        ))
        .is_ok());
    assert!(expected
        .verify(&Payload::Tensor(
            Tensor::from_f32_shape(&[1.; 3], vec![1, 3]).unwrap()
        ))
        .is_err());
    let registry = ActionRegistry::default();
    let concat = action(&registry, "tensor_basics", "concat");
    let input = ValueShape::composite([
        ValueShape::tensor(Shape::new([0, 3])),
        ValueShape::tensor(Shape::new([2, 3])),
    ]);
    let Some(ValueShapeResult::Ok(shape)) = concat.output_value_result(&input, &args(&[])) else {
        panic!()
    };
    assert_eq!(shape.shape().unwrap().known_dims(), Some(vec![2, 3]));
    let input = ValueShape::composite([
        ValueShape::tensor(Shape::new([Dimension::Unknown, 3.into()])),
        ValueShape::tensor(Shape::new([2, 3])),
    ]);
    let Some(ValueShapeResult::Ok(shape)) = concat.output_value_result(&input, &args(&[])) else {
        panic!()
    };
    assert_eq!(
        shape.shape().unwrap().dims(),
        [Dimension::Unknown, 3.into()]
    );
}

fn check(source: &str) -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let file = temp.path().join("dynamic.morf");
    std::fs::write(&file, source)?;
    morflow::check::check_pipeline(
        &file,
        Some(std::path::Path::new(&std::env::var(
            "MORFLOW_ACTIONS_PATH",
        )?)),
    )
}

#[test]
fn each_preserves_row_dimensions_and_infers_the_restacked_output() {
    let prefix="import tensor_basics/latest\naccept Tensor[*,3] $input\n$input >> each ($row) { $row >> reshape(shape=\"[3]\") }";
    assert!(check(&format!("{prefix} >> transpose >> emit")).is_ok());
    // Row length remains three, so this is rejected statically inside each.
    assert!(check("import tensor_basics/latest\naccept Tensor[*,3] $input\n$input >> each ($row) { $row >> reshape(shape=\"[4]\") } >> emit").is_err());
    // Restacking restores the batch dimension and its known length.
    let fixed = prefix.replace("[*,3]", "[2,3]");
    assert!(check(&format!("{fixed} >> reshape(shape=\"[3]\") >> emit")).is_err());
    let mut flow = Morflow::from_str(&format!("{prefix} >> transpose >> emit")).unwrap();
    let out = flow
        .run(Payload::Tensor(
            Tensor::from_f32_shape(&[1.; 6], vec![2, 3]).unwrap(),
        ))
        .unwrap();
    assert!(matches!(out.single().unwrap(),Payload::Tensor(t) if t.shape.as_slice()==[3,2]));
    let loop_type = PType::Tensor(ShapeSpec::parse("[*,3]").unwrap())
        .each_loop_var()
        .unwrap();
    assert_eq!(loop_type.to_string(), "Tensor[3]");
}

#[test]
fn conversion_contracts_retain_rank_when_the_payload_kind_makes_it_known() {
    let registry = ActionRegistry::default();
    let to_tensor = action(&registry, "base", "to_tensor");
    let audio = ValueShape::Leaf {
        kind: DataType::Audio,
        shape: Some(Shape::new([Dimension::Known(2), Dimension::Unknown])).into(),
    };
    let Some(ValueShapeResult::Ok(shape)) = to_tensor.output_value_result(&audio, &args(&[]))
    else {
        panic!()
    };
    assert_eq!(shape.shape().unwrap().rank(), 2);
    let ValueShapeResult::Ok(shape) = prediction(
        &registry,
        "audio_basics",
        "to_audio",
        Shape::unknown(1),
        args(&[]),
    ) else {
        panic!()
    };
    assert_eq!(shape.shape().unwrap().rank(), 1);
    let ValueShapeResult::Ok(shape) = prediction(
        &registry,
        "audio_basics",
        "resample",
        Shape::new([Dimension::Known(2), Dimension::Unknown]),
        args(&[]),
    ) else {
        panic!()
    };
    assert_eq!(shape.shape().unwrap().rank(), 2);
    let ValueShapeResult::Ok(shape) = prediction(
        &registry,
        "tensor_basics",
        "squeeze",
        Shape::unknown(1),
        args(&[]),
    ) else {
        panic!()
    };
    assert_eq!(shape.shape().unwrap().rank(), 1);
}
