//! Run after building the two BLAS packs and preparing local test actions.
use core_types::{ActionArgs, InputDescriptor, RString, ShapeCheckResult};
use pipeline::{ActionIdentity, ActionRegistry, Morflow, Payload, RVec, Tensor};
fn matrix() -> Payload {
    Payload::Tensor(Tensor::from_f32_shape(&[4., 1., 1., 3.], vec![2, 2]).unwrap())
}
fn action_input(name: &str) -> (Payload, ActionArgs) {
    let mut args = ActionArgs::default();
    let payload = match name {
        "matmul" | "concat" => Payload::Composite(RVec::from(vec![matrix(), matrix()])),
        "dot" | "outer" => {
            let vector = Payload::Tensor(Tensor::from_f32_slice(&[1., 2.]));
            Payload::Composite(RVec::from(vec![vector.clone(), vector]))
        }
        "repeat" => {
            args.positional.push(RString::from("2,1"));
            matrix()
        }
        "roll" => {
            args.positional.push(RString::from("1"));
            matrix()
        }
        _ => matrix(),
    };
    (payload, args)
}
#[test]
fn all_blas_contracts_match_basics_and_runtime_outputs_match_the_predictions() {
    let registry = ActionRegistry::default();
    for (pack, names) in [
        (
            "linalg",
            vec!["matmul", "dot", "outer", "inv", "det", "qr", "cholesky"],
        ),
        ("tensor", vec!["concat", "repeat", "roll"]),
    ] {
        for name in names {
            let basic = registry
                .get_or_load(
                    &ActionIdentity::new(&format!("{pack}_basics"), "latest", name).unwrap(),
                )
                .unwrap();
            let blas = registry
                .get_or_load(&ActionIdentity::new(&format!("{pack}_blas"), "0.1.2", name).unwrap())
                .unwrap();
            let blas = blas.with_plugins(plugins()).unwrap();
            let (input, args) = action_input(name);
            let describe = || InputDescriptor::from_payload(&input);
            let ShapeCheckResult::Ready {
                output: expected, ..
            } = basic.shapecheck(describe(), args.clone())
            else {
                panic!("basic {name} did not check")
            };
            let ShapeCheckResult::Ready {
                output: predicted, ..
            } = blas.shapecheck(describe(), args.clone())
            else {
                panic!("blas {name} did not check")
            };
            assert_eq!(expected, predicted, "{pack}/{name}");
            let result = blas.process(Payload::WithArgs {
                payload: core_types::RBox::new(input),
                args,
            });
            predicted.verify(&result).unwrap();
        }
    }
}
#[test]
fn file_and_string_loads_reuse_mixed_pack_pipeline_and_preserve_input() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let path = root.join("pipeline/tests/fixtures/blas.morf");
    let source = std::fs::read_to_string(&path).unwrap();
    let mut file = Morflow::load(&path).unwrap();
    let mut string = Morflow::from_str(&source).unwrap();
    let a = Tensor::from_f32_shape(&[1., 2., 3., 4., 5., 6.], vec![2, 3]).unwrap();
    let b = Tensor::from_f32_shape(
        &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        vec![3, 4],
    )
    .unwrap();
    let input = Payload::Composite(vec![Payload::Tensor(a.clone()), Payload::Tensor(b)].into());
    for _ in 0..3 {
        for flow in [&mut file, &mut string] {
            let result = flow.run(input.clone()).unwrap();
            let Payload::Tensor(output) = result.single().unwrap() else {
                panic!("Expected tensor")
            };
            assert_eq!(output.shape.as_slice(), [8, 2]);
            assert_eq!(
                output.to_vec_f32(),
                vec![
                    38., 83., 44., 98., 50., 113., 56., 128., 38., 83., 44., 98., 50., 113., 56.,
                    128.
                ]
            );
        }
    }
    assert_eq!(a.to_vec_f32(), vec![1., 2., 3., 4., 5., 6.]);
}
#[test]
fn missing_openblas_allows_offline_check_and_both_load_signatures_then_fails_on_run() {
    const FLAG: &str = "MORFLOW_BLAS_PIPELINE_CONTRACT_CHILD";
    if std::env::var_os(FLAG).is_none() {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "missing_openblas_allows_offline_check_and_both_load_signatures_then_fails_on_run",
            ])
            .env(FLAG, "1")
            .env(
                "MORFLOW_OPENBLAS_LIBRARY",
                "/definitely/missing/openblas.so",
            )
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
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let path = root.join("pipeline/tests/fixtures/blas.morf");
    let source = std::fs::read_to_string(&path).unwrap();
    pipeline::check::check_pipeline(&path, None).unwrap();
    let mut file = Morflow::load(&path).unwrap();
    let mut string = Morflow::from_str(&source).unwrap();
    let a = Tensor::from_f32_shape(&[1.; 6], vec![2, 3]).unwrap();
    let b = Tensor::from_f32_shape(&[1.; 12], vec![3, 4]).unwrap();
    let input = Payload::Composite(vec![Payload::Tensor(a), Payload::Tensor(b)].into());
    for flow in [&mut file, &mut string] {
        let error = flow.run(input.clone()).unwrap_err().to_string();
        assert!(error.contains("MORFLOW_OPENBLAS_LIBRARY"), "{error}");
    }
}

#[test]
fn aliased_basics_and_blas_matmul_coexist_in_one_pipeline() {
    let mut pipeline=Morflow::from_str("plugin openblas/0.1.1\nimport linalg_basics/latest as basic\nimport linalg_blas/0.1.2 as accelerated\naccept Tensor[2,2] $matrix\n$matrix >> basic/matmul >> emit(\"basic\")\n$matrix >> accelerated/matmul >> emit(\"blas\")").unwrap();
    let result = pipeline.run(matrix()).unwrap();
    let Payload::Tensor(basic) = result.get("basic").unwrap() else {
        panic!("Expected tensor")
    };
    let Payload::Tensor(blas) = result.get("blas").unwrap() else {
        panic!("Expected tensor")
    };
    assert_eq!(basic.to_vec_f32(), vec![17., 7., 7., 10.]);
    assert_eq!(basic.to_vec_f32(), blas.to_vec_f32());
}

fn plugins() -> std::sync::Arc<pipeline::plugins::PluginSet> {
    pipeline::plugins::PluginSet::prepare(
        &[parser::ast::PluginDecl {
            name: "openblas".into(),
            version: "0.1.1".into(),
        }],
        &pipeline::plugins::search_paths(),
    )
    .unwrap()
}

#[test]
fn blas_actions_require_a_plugin_declaration() {
    for (pack, name) in [
        ("linalg_blas", "matmul"),
        ("linalg_blas", "dot"),
        ("linalg_blas", "outer"),
        ("linalg_blas", "inv"),
        ("linalg_blas", "det"),
        ("linalg_blas", "qr"),
        ("linalg_blas", "cholesky"),
        ("tensor_blas", "concat"),
        ("tensor_blas", "repeat"),
        ("tensor_blas", "roll"),
    ] {
        let source = format!(
            "from {pack}/0.1.1 import {name}\naccept Tensor[2,2] $x\n$x >> {name} >> emit\n"
        );
        let error = Morflow::from_str(&source).err().unwrap().to_string();
        assert!(
            error.contains("openblas") && error.contains("undeclared"),
            "{pack}/{name}: {error}"
        );
    }
}
