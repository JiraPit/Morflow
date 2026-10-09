use core_types::{ActionArgs, DataType, Shape, ValueShape, ValueShapeResult};
use morflow::{ActionIdentity, ActionRegistry, Morflow, Payload, Tensor};

fn tensor(dims: &[usize], value: f32) -> Payload {
    Payload::Tensor(
        Tensor::from_f32_shape(&vec![value; dims.iter().product()], dims.to_vec()).unwrap(),
    )
}
fn pair(a: &[usize], b: &[usize]) -> Payload {
    Payload::Composite(vec![tensor(a, 1.), tensor(b, 2.)].into())
}
fn loaded(
    registry: &ActionRegistry,
    pack: &str,
    action: &str,
) -> std::sync::Arc<morflow::LoadedAction> {
    registry
        .get_or_load(&ActionIdentity::new(pack, "latest", action).unwrap())
        .unwrap()
}
fn compare(
    registry: &ActionRegistry,
    pack: &str,
    action: &str,
    input: Payload,
    args: ActionArgs,
    dims: &[usize],
) {
    let action = loaded(registry, pack, action);
    let result = action
        .output_value_result(&ValueShape::from_payload(&input), &args)
        .expect("New shape export required");
    let ValueShapeResult::Ok(expected) = result else {
        panic!("Unexpected contract: {result:?}")
    };
    assert_eq!(expected.shape().unwrap().dims(), dims);
    let out = action.process(Payload::WithArgs {
        payload: core_types::RBox::new(input),
        args,
    });
    expected.verify(&out).unwrap();
    assert!(matches!(out,Payload::Tensor(ref t) if t.shape.as_slice()==dims));
}

#[test]
fn native_contracts_cover_all_five_ordered_input_actions() {
    let registry = ActionRegistry::default();
    compare(
        &registry,
        "linalg_basics",
        "dot",
        pair(&[2, 3], &[6]),
        ActionArgs::default(),
        &[1],
    );
    compare(
        &registry,
        "linalg_basics",
        "outer",
        pair(&[2], &[3]),
        ActionArgs::default(),
        &[2, 3],
    );
    compare(
        &registry,
        "linalg_basics",
        "matmul",
        pair(&[2, 3], &[3, 4]),
        ActionArgs::default(),
        &[2, 4],
    );
    compare(
        &registry,
        "nn_basics",
        "cosine_similarity",
        pair(&[2, 3, 4], &[2, 3, 4]),
        ActionArgs::default(),
        &[2, 3],
    );
    let args = ActionArgs {
        positional: vec!["-1".into()].into(),
        named: vec![].into(),
    };
    compare(
        &registry,
        "tensor_basics",
        "concat",
        Payload::Composite(
            vec![
                tensor(&[2, 3], 1.),
                tensor(&[2, 4], 2.),
                tensor(&[2, 5], 3.),
            ]
            .into(),
        ),
        args,
        &[2, 12],
    );
    // Tensor-only forms keep their existing self-operation or pass-through semantics.
    for (pack, action, input, output) in [
        ("linalg_basics", "dot", vec![2, 2], vec![1]),
        ("linalg_basics", "outer", vec![2, 2], vec![4, 4]),
        ("linalg_basics", "matmul", vec![2, 2], vec![2, 2]),
        ("nn_basics", "cosine_similarity", vec![2, 3], vec![2, 3]),
        ("tensor_basics", "concat", vec![2, 3], vec![2, 3]),
    ] {
        compare(
            &registry,
            pack,
            action,
            tensor(&input, 1.),
            ActionArgs::default(),
            &output,
        );
    }
}

#[test]
fn non_f32_tensor_components_keep_shape_and_value_counts() {
    let registry = ActionRegistry::default();
    let a = Payload::Tensor(Tensor::from_i32_vec(vec![1, 2], vec![2]).unwrap());
    let b = Payload::Tensor(
        Tensor::from_rvec_u8(vec![3, 4, 5].into(), vec![3], core_types::TensorDType::U8).unwrap(),
    );
    let out =
        loaded(&registry, "linalg_basics", "outer").process(Payload::Composite(vec![a, b].into()));
    let Payload::Tensor(out) = out else {
        panic!("Expected outer product")
    };
    assert_eq!(out.shape.as_slice(), [2, 3]);
    assert_eq!(out.to_vec_f32(), [3., 4., 5., 6., 8., 10.]);
}

#[test]
fn matmul_broadcasts_batches_in_order_and_rejects_incompatible_batches() {
    let registry = ActionRegistry::default();
    let action = loaded(&registry, "linalg_basics", "matmul");
    let a = Payload::Tensor(
        Tensor::from_f32_shape(&[vec![1.; 6], vec![2.; 6]].concat(), vec![2, 1, 2, 3]).unwrap(),
    );
    let b = Payload::Tensor(
        Tensor::from_f32_shape(
            &(1..=4).flat_map(|i| vec![i as f32; 6]).collect::<Vec<_>>(),
            vec![1, 4, 3, 2],
        )
        .unwrap(),
    );
    let input = Payload::Composite(vec![a, b].into());
    let ValueShapeResult::Ok(shape) = action
        .output_value_result(&ValueShape::from_payload(&input), &ActionArgs::default())
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(shape.shape().unwrap().dims(), [2, 4, 2, 2]);
    let out = action.process(input);
    shape.verify(&out).unwrap();
    let Payload::Tensor(out) = out else { panic!() };
    let expected = (1..=2)
        .flat_map(|a| (1..=4).flat_map(move |b| vec![(3 * a * b) as f32; 4]))
        .collect::<Vec<_>>();
    assert_eq!(out.to_vec_f32(), expected);
    let invalid = pair(&[2, 2, 3], &[3, 3, 4]);
    assert!(matches!(
        action.output_value_result(&ValueShape::from_payload(&invalid), &ActionArgs::default()),
        Some(ValueShapeResult::Invalid(_))
    ));
    assert!(matches!(action.process(invalid), Payload::Error(_)));
}

#[test]
fn arity_types_dimensions_and_overflow_fail_before_processing() {
    let registry = ActionRegistry::default();
    for (pack, action, input) in [
        ("linalg_basics", "dot", pair(&[2], &[3])),
        ("linalg_basics", "matmul", pair(&[2, 3], &[2, 4])),
        ("nn_basics", "cosine_similarity", pair(&[2, 3], &[3, 2])),
        ("tensor_basics", "concat", pair(&[2, 3], &[3, 4])),
        ("tensor_basics", "concat", Payload::Composite(vec![].into())),
    ] {
        let action = loaded(&registry, pack, action);
        assert!(matches!(
            action.output_value_result(&ValueShape::from_payload(&input), &ActionArgs::default()),
            Some(ValueShapeResult::Invalid(_))
        ));
        assert!(matches!(action.process(input), Payload::Error(_)));
    }
    for (pack, name) in [
        ("linalg_basics", "dot"),
        ("linalg_basics", "outer"),
        ("linalg_basics", "matmul"),
        ("nn_basics", "cosine_similarity"),
    ] {
        let action = loaded(&registry, pack, name);
        assert!(matches!(
            action.process(Payload::arg("bad input")),
            Payload::Error(_)
        ));
        for input in [
            Payload::Composite(vec![tensor(&[2, 2], 1.)].into()),
            Payload::Composite(vec![tensor(&[2, 2], 1.); 3].into()),
            Payload::Composite(vec![Payload::scalar_f32(1.), tensor(&[2, 2], 1.)].into()),
        ] {
            assert!(matches!(action.process(input), Payload::Error(_)), "{name}");
        }
        assert!(matches!(
            action.output_value_result(
                &ValueShape::unranked(DataType::Composite),
                &ActionArgs::default()
            ),
            Some(ValueShapeResult::Ok(ref shape)) if shape.shape().is_none()
        ));
    }
    let huge = ValueShape::composite([
        ValueShape::tensor(Shape::new([usize::MAX])),
        ValueShape::tensor(Shape::new([2])),
    ]);
    assert!(matches!(
        loaded(&registry, "linalg_basics", "outer")
            .output_value_result(&huge, &ActionArgs::default()),
        Some(ValueShapeResult::Invalid(_))
    ));
}

#[test]
fn empty_inputs_do_not_panic_in_native_kernels() {
    let registry = ActionRegistry::default();
    for (pack, name, input, expected) in [
        ("linalg_basics", "dot", pair(&[0], &[0]), vec![1]),
        ("linalg_basics", "outer", pair(&[2], &[0]), vec![2, 0]),
        (
            "linalg_basics",
            "matmul",
            pair(&[2, 0], &[0, 3]),
            vec![2, 3],
        ),
        (
            "linalg_basics",
            "matmul",
            pair(&[0, 2], &[2, 3]),
            vec![0, 3],
        ),
        (
            "nn_basics",
            "cosine_similarity",
            pair(&[2, 0], &[2, 0]),
            vec![2],
        ),
    ] {
        let out = loaded(&registry, pack, name).process(input);
        assert!(
            matches!(out,Payload::Tensor(ref t) if t.shape.as_slice()==expected),
            "{name}: {out:?}"
        );
    }
}

fn check(source: &str) -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("compound.morf");
    std::fs::write(&path, source)?;
    morflow::check::check_pipeline(
        &path,
        Some(std::path::Path::new(&std::env::var(
            "MORFLOW_ACTIONS_PATH",
        )?)),
    )
}
#[test]
fn typed_composite_declarations_and_qr_output_infer_downstream_shapes() {
    for (imports, ty, call, output) in [
        (
            "linalg_basics",
            "Composite[Tensor[2,3],Tensor[3,4]]",
            "matmul",
            "[8]",
        ),
        (
            "linalg_basics",
            "Composite[Tensor[2,3],Tensor[6]]",
            "dot",
            "[1]",
        ),
        (
            "linalg_basics",
            "Composite[Tensor[2],Tensor[3]]",
            "outer",
            "[6]",
        ),
        (
            "nn_basics",
            "Composite[Tensor[2,3],Tensor[2,3]]",
            "cosine_similarity",
            "[2]",
        ),
        (
            "tensor_basics",
            "Composite[Tensor[2,3],Tensor[4,3],Tensor[1,3]]",
            "concat",
            "[21]",
        ),
    ] {
        let prefix=format!("import {imports}/latest\nimport tensor_basics/latest as tensor\naccept {ty} $parts\n$parts >> {call}");
        assert!(
            check(&format!(
                "{prefix} >> tensor.reshape(shape=\"{output}\") >> emit"
            ))
            .is_ok(),
            "{call}"
        );
        assert!(
            check(&format!(
                "{prefix} >> tensor.reshape(shape=\"[99]\") >> emit"
            ))
            .is_err(),
            "{call}"
        );
    }
    let source="import linalg_basics/latest\naccept Tensor[3,2] $matrix\n$matrix >> qr >> $parts\n$parts >> matmul >> emit";
    assert!(check(source).is_ok());
    let mut flow = Morflow::from_str(source).unwrap();
    let out = flow.run(tensor(&[3, 2], 1.)).unwrap();
    assert!(matches!(out.single().unwrap(),Payload::Tensor(t) if t.shape.as_slice()==[3,2]));
    let mut nested = Morflow::from_str(
        "accept Composite[Bytes,Composite[Tensor[2,3],Scalar]] $parts\n$parts[1][0] >> emit",
    )
    .unwrap();
    let input = Payload::Composite(
        vec![
            Payload::Data {
                buffer: vec![].into(),
            },
            Payload::Composite(vec![tensor(&[2, 3], 1.), Payload::scalar_f32(2.)].into()),
        ]
        .into(),
    );
    assert!(nested.run(input).is_ok());
    let invalid = Payload::Composite(
        vec![
            Payload::Data {
                buffer: vec![].into(),
            },
            Payload::Composite(vec![tensor(&[3, 2], 1.), Payload::scalar_f32(2.)].into()),
        ]
        .into(),
    );
    assert!(nested.run(invalid).is_err());
    assert!(Morflow::from_str("accept Composite[IntArg] $parts\n$parts >> emit").is_err());
}
