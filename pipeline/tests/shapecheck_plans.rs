use core_types::{ActionArgs, InputDescriptor, Shape, ShapeCheckResult, Tuple2, ValueShape};
use morflow::{ActionIdentity, ActionRegistry, Payload, Tensor};

fn args(values: &[(&str, &str)]) -> ActionArgs {
    ActionArgs {
        positional: vec![].into(),
        named: values
            .iter()
            .map(|(k, v)| Tuple2((*k).into(), (*v).into()))
            .collect(),
    }
}
fn input() -> Payload {
    Payload::Tensor(Tensor::from_f32_shape(&[0., 1., 2., 3., 4., 5.], vec![2, 3]).unwrap())
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
fn run(action: &morflow::LoadedAction, input: Payload, args: ActionArgs) -> Payload {
    action.process(Payload::WithArgs {
        payload: core_types::RBox::new(input),
        args,
    })
}
#[test]
fn normalized_softmax_axes_preserve_existing_negative_axis_behavior() {
    let registry = ActionRegistry::default();
    for name in ["softmax", "log_softmax"] {
        let action = loaded(&registry, "nn_basics", name);
        let input = input();
        let args = args(&[("axis", "-99")]);
        let ShapeCheckResult::Ready { prepared, .. } =
            action.shapecheck(InputDescriptor::from_payload(&input), args.clone())
        else {
            panic!("expected ready")
        };
        assert_eq!(prepared.unsigned("axis"), Some(0));
        let Payload::Tensor(output) = run(&action, input, args) else {
            panic!("expected tensor")
        };
        let values = output.to_vec_f32();
        for column in 0..3 {
            let sum = if name == "softmax" {
                values[column] + values[column + 3]
            } else {
                values[column].exp() + values[column + 3].exp()
            };
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }
}
#[test]
fn pooling_uses_normalized_parameters_and_predicted_geometry() {
    let registry = ActionRegistry::default();
    for name in ["max_pool2d", "avg_pool2d"] {
        let action = loaded(&registry, "nn_basics", name);
        let input = input();
        let args = args(&[("kernel", "0"), ("stride", "0")]);
        let ShapeCheckResult::Ready { prepared, .. } =
            action.shapecheck(InputDescriptor::from_payload(&input), args.clone())
        else {
            panic!("expected ready")
        };
        assert_eq!(prepared.unsigned("kernel"), Some(1));
        assert_eq!(prepared.unsigned("stride"), Some(1));
        assert_eq!(prepared.output_dims().unwrap(), vec![2, 3]);
        let Payload::Tensor(output) = run(&action, input, args) else {
            panic!("expected tensor")
        };
        assert_eq!(output.to_vec_f32(), vec![0., 1., 2., 3., 4., 5.]);
        assert!(matches!(
            run(&action, self::input(), self::args(&[("kernel", "99")])),
            Payload::Error(_)
        ));
    }
}
#[test]
fn deferred_reductions_and_ready_rolls_keep_axes_in_order() {
    let registry = ActionRegistry::default();
    let sum = loaded(&registry, "tensor_stats", "sum");
    let arguments = args(&[("axis", "-1")]);
    let ShapeCheckResult::Deferred { output, .. } = sum.shapecheck(
        InputDescriptor::partial(ValueShape::tensor(Shape::new([
            core_types::Dimension::Unknown,
            3.into(),
        ]))),
        arguments.clone(),
    ) else {
        panic!("expected deferred")
    };
    assert_eq!(output.shape().unwrap().rank(), 1);
    let concrete = input();
    let ShapeCheckResult::Ready { prepared, .. } =
        sum.shapecheck(InputDescriptor::from_payload(&concrete), arguments.clone())
    else {
        panic!("expected ready")
    };
    assert_eq!(prepared.unsigned("axis"), Some(1));
    let Payload::Tensor(output) = run(&sum, concrete, arguments) else {
        panic!("expected tensor")
    };
    assert_eq!(output.to_vec_f32(), vec![3., 12.]);
    let roll = loaded(&registry, "tensor_basics", "roll");
    let Payload::Tensor(output) = run(&roll, input(), args(&[("shift", "1"), ("axis", "-1")]))
    else {
        panic!("expected tensor")
    };
    assert_eq!(output.to_vec_f32(), vec![2., 0., 1., 5., 3., 4.]);
}
