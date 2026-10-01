use pipeline::{Audio, Morflow, Payload, Tensor};

fn matrix() -> Payload {
    Payload::Tensor(Tensor::from_f32_shape(&[1., 0., 0., 1., 1., 1.], vec![3, 2]).unwrap())
}

#[test]
fn qr_components_can_be_selected_and_indexed_further() {
    let mut flow = Morflow::from_str(
        r#"
        import linalg_essentials/latest
        accept Tensor[3,2] $matrix
        $matrix >> qr >> $parts
        $parts[0] >> emit("q")
        $parts[1] >> emit("r")
        $parts[0][0] >> emit("row")
        $parts[1][0,0] >> emit("entry")
    "#,
    )
    .unwrap();
    let out = flow.run(matrix()).unwrap();
    for (name, expected) in [("q", vec![3, 2]), ("r", vec![2, 2]), ("row", vec![2])] {
        let Payload::Tensor(t) = &out[name] else {
            panic!("Expected Tensor {name}");
        };
        assert_eq!(t.shape.as_slice(), expected);
    }
    let Payload::Scalar(t) = &out["entry"] else {
        panic!("Expected scalar entry");
    };
    assert!((t.as_f32_slice().unwrap()[0] - 2f32.sqrt()).abs() < 1e-5);
    // Q * R reconstructs the input, so selecting components keeps their bytes intact.
    let (Payload::Tensor(q), Payload::Tensor(r)) = (&out["q"], &out["r"]) else {
        unreachable!()
    };
    let q = q.to_vec_f32();
    let r = r.to_vec_f32();
    for (i, expected) in [1., 0., 0., 1., 1., 1.].into_iter().enumerate() {
        let reconstructed = (0..2)
            .map(|k| q[(i / 2) * 2 + k] * r[k * 2 + i % 2])
            .sum::<f32>();
        assert!((reconstructed - expected).abs() < 1e-5);
    }
}

#[test]
fn mixed_and_nested_composites_preserve_payloads_and_bracket_groups() {
    let nested = Payload::Composite(vec![Payload::scalar_f32(7.)].into());
    let audio = Payload::Audio(Audio::from_f32_planar(&[0.5; 8], 1, 48000).unwrap());
    let input = Payload::Composite(
        vec![
            Payload::Data {
                buffer: b"abc".to_vec().into(),
            },
            matrix(),
            nested,
            audio,
        ]
        .into(),
    );
    let mut flow = Morflow::from_str(
        r#"
        accept Composite $parts
        $parts[0] >> emit("bytes")
        $parts[1][:,1][0] >> emit("entry")
        $parts[2][0] >> emit("nested")
        $parts[3] >> emit("audio")
    "#,
    )
    .unwrap();
    let out = flow.run(input).unwrap();
    assert!(matches!(&out["bytes"], Payload::Data { buffer } if buffer.as_slice() == b"abc"));
    assert!(matches!(&out["nested"], Payload::Scalar(t) if t.as_f32_slice().unwrap() == [7.]));
    assert!(matches!(&out["entry"], Payload::Scalar(t) if t.as_f32_slice().unwrap() == [0.]));
    assert!(
        matches!(&out["audio"], Payload::Audio(a) if a.sample_rate == 48000 && a.tensor.shape.as_slice() == [8])
    );
}

#[test]
fn invalid_composite_indexes_and_slice_forms_fail_explicitly() {
    for (reference, expected) in [
        ("$parts[-1]", "nonnegative"),
        ("$parts[1]", "out of bounds"),
        ("$parts[:]", "integer indexes only"),
        ("$parts[0:1]", "integer indexes only"),
        ("$parts[axis=0]", "integer indexes only"),
        ("$parts[0][0]", "Cannot slice"),
    ] {
        let source = format!("accept Composite $parts\n{reference} >> emit");
        let mut flow = Morflow::from_str(&source).unwrap();
        let error = flow
            .run(Payload::Composite(vec![Payload::scalar_f32(1.)].into()))
            .unwrap_err();
        assert!(error.to_string().contains(expected), "{reference}: {error}");
    }
    let mut empty = Morflow::from_str("accept Composite $parts\n$parts[0] >> emit").unwrap();
    assert!(empty
        .run(Payload::Composite(vec![].into()))
        .unwrap_err()
        .to_string()
        .contains("out of bounds"));
}

fn check(source: &str) -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("composite.morf");
    std::fs::write(&path, source)?;
    let cache = std::env::var("MORFLOW_ACTIONS_PATH").unwrap();
    pipeline::check::check_pipeline(&path, Some(std::path::Path::new(&cache)))
}

#[test]
fn checker_tracks_qr_shapes_and_rejects_bad_selected_calls() {
    let prefix = "import linalg_essentials/latest\nimport tensor_essentials/latest\naccept Tensor[3,2] $matrix\n$matrix >> qr >> $parts\n";
    assert!(check(&format!(
        "{prefix}$parts[0] >> reshape(shape=\"[6]\") >> emit"
    ))
    .is_ok());
    assert!(check(&format!(
        "{prefix}$parts[1] >> reshape(shape=\"[4]\") >> emit"
    ))
    .is_ok());
    assert!(check(&format!(
        "{prefix}reshape($parts[1], shape=\"[4]\") >> emit"
    ))
    .is_ok());
    assert!(check(&format!(
        "{prefix}$parts[1] >> reshape(shape=\"[6]\") >> emit"
    ))
    .is_err());
    assert!(check(&format!("{prefix}$parts[2] >> emit"))
        .unwrap_err()
        .to_string()
        .contains("out of bounds"));
    assert!(check(&format!(
        "{prefix}$parts[0][:,1][0] >> reshape(shape=\"[1]\") >> emit"
    ))
    .is_ok());
    assert!(check("accept Composite $parts\n$parts[0] >> emit").is_ok());
    assert!(check("accept Composite $parts\n$parts[:] >> emit").is_err());
}
