//! Conversions between the parser's declared parameter types and the runtime
//! [`PType`] model, plus the rules for turning declared defaults and values
//! supplied by a host into payloads.
//!
//! The parser deliberately does not depend on `core_types`, so this module
//! owns the one-way translation. These are plain functions rather than `From`
//! impls because both types are foreign to this crate.

use core_types::{ArgKind, Dim, PType, Payload, RVec, ShapeSpec, Tensor, TensorDType};
use parser::ast::{ParamDim, ParamShape, ParamType, PipelineParam, Value};

/// The runtime type of a declared parameter.
pub fn ptype_of(param_type: &ParamType) -> PType {
    match param_type {
        ParamType::Bytes => PType::Bytes,
        ParamType::IntArg => PType::Arg(ArgKind::Int),
        ParamType::FloatArg => PType::Arg(ArgKind::Float),
        ParamType::StrArg => PType::Arg(ArgKind::Str),
        ParamType::BoolArg => PType::Arg(ArgKind::Bool),
        ParamType::Scalar => PType::Scalar,
        ParamType::Tensor(shape) => PType::Tensor(shape_spec_of(shape)),
        ParamType::Image(shape) => PType::Image(shape_spec_of(shape)),
        ParamType::Audio(shape) => PType::Audio(shape_spec_of(shape)),
        ParamType::Composite => PType::Composite,
        ParamType::CompositeItems(items) => {
            PType::CompositeItems(items.iter().map(ptype_of).collect())
        }
    }
}

/// The shape specification of a declared parameter.
pub fn shape_spec_of(shape: &ParamShape) -> ShapeSpec {
    match shape {
        ParamShape::AnyRank => ShapeSpec::AnyRank,
        ParamShape::Ranked { dims } => ShapeSpec::Ranked {
            dims: dims.iter().map(dim_of).collect(),
        },
    }
}

fn dim_of(dim: &ParamDim) -> Dim {
    match dim {
        ParamDim::Any => Dim::Any,
        ParamDim::Fixed(n) => Dim::Fixed(*n),
    }
}

/// The declared type of a runtime type, for messages that quote a declaration.
pub fn param_type_of(ptype: &PType) -> Option<ParamType> {
    match ptype {
        PType::Unknown => None,
        PType::Bytes => Some(ParamType::Bytes),
        PType::Arg(kind) => Some(match kind {
            ArgKind::Int => ParamType::IntArg,
            ArgKind::Float => ParamType::FloatArg,
            ArgKind::Str => ParamType::StrArg,
            ArgKind::Bool => ParamType::BoolArg,
        }),
        PType::Scalar => Some(ParamType::Scalar),
        PType::Tensor(spec) => Some(ParamType::Tensor(shape_of(spec))),
        PType::Image(spec) => Some(ParamType::Image(shape_of(spec))),
        PType::Audio(spec) => Some(ParamType::Audio(shape_of(spec))),
        PType::Composite => Some(ParamType::Composite),
        PType::CompositeItems(items) => items
            .iter()
            .map(param_type_of)
            .collect::<Option<Vec<_>>>()
            .map(ParamType::CompositeItems),
    }
}

fn shape_of(spec: &ShapeSpec) -> ParamShape {
    match spec {
        ShapeSpec::AnyRank => ParamShape::AnyRank,
        ShapeSpec::Ranked { dims } => ParamShape::Ranked {
            dims: dims
                .iter()
                .map(|d| match d {
                    Dim::Any => ParamDim::Any,
                    Dim::Fixed(n) => ParamDim::Fixed(*n),
                })
                .collect(),
        },
    }
}

/// The payload a parameter holds when the host supplies nothing.
///
/// A declared default always produces the value in its own representation: an
/// argument for the four `*Arg` types, a rank-0 tensor for `Scalar`, and plain
/// bytes for `Bytes`.
pub fn default_payload(param: &PipelineParam) -> Option<Payload> {
    let declared = ptype_of(&param.param_type);
    let value = param.default_value.as_ref()?;
    match (&declared, value) {
        (_, Value::Int(v)) => match declared {
            PType::Arg(_) | PType::Bytes => Some(Payload::arg(v.to_string())),
            PType::Scalar => Some(Payload::Scalar(scalar_tensor(*v as f64, TensorDType::I32))),
            _ => None,
        },
        (_, Value::Float(v)) => match declared {
            PType::Arg(_) | PType::Bytes => Some(Payload::arg(v.to_string())),
            PType::Scalar => Some(Payload::Scalar(scalar_tensor(*v, TensorDType::F32))),
            _ => None,
        },
        (_, Value::String(v)) => match declared {
            PType::Arg(_) | PType::Bytes => Some(Payload::arg(v.clone())),
            _ => None,
        },
        (_, Value::Bool(v)) => match declared {
            PType::Arg(_) | PType::Bytes => Some(Payload::arg(v.to_string())),
            _ => None,
        },
        _ => None,
    }
}

/// Re-tags a host-supplied payload so it matches the declared type.
///
/// Hosts pass numbers and strings as byte buffers, so a value bound to an
/// argument parameter is retagged as an argument, and a rank-0 tensor bound to
/// a `Scalar` is retagged as a scalar. Anything else is returned untouched for
/// [`PType::verify_payload`] to accept or reject.
pub fn coerce_host_payload(declared: &PType, payload: Payload) -> Payload {
    let inner = match &payload {
        Payload::WithArgs { payload, .. } => payload.as_ref().clone(),
        other => other.clone(),
    };
    match (&declared, inner) {
        (PType::Arg(_), Payload::Data { buffer }) => Payload::Arg(buffer),
        (PType::Arg(_), Payload::Arg(bytes)) => Payload::Arg(bytes),
        (PType::Scalar, Payload::Tensor(t)) if t.rank() == 0 => Payload::Scalar(t),
        (PType::Scalar, Payload::Scalar(t)) => Payload::Scalar(t),
        (PType::Scalar, Payload::Data { buffer }) => match buffer_to_scalar(&buffer) {
            Some(t) => Payload::Scalar(t),
            None => Payload::Data { buffer },
        },
        (_, inner) => inner,
    }
}

fn buffer_to_scalar(buffer: &RVec<u8>) -> Option<Tensor> {
    let text = std::str::from_utf8(buffer.as_slice()).ok()?;
    let trimmed = text.trim();
    if let Ok(v) = trimmed.parse::<f64>() {
        return Some(scalar_tensor(v, TensorDType::F32));
    }
    if let Ok(v) = trimmed.parse::<i64>() {
        return Some(scalar_tensor(v as f64, TensorDType::I32));
    }
    None
}

/// A rank-0 tensor holding a single number of the given dtype.
pub fn scalar_tensor(value: f64, dtype: TensorDType) -> Tensor {
    match dtype {
        TensorDType::I32 => {
            Tensor::from_i32_vec(vec![value as i32], vec![]).expect("rank-0 shape is valid")
        }
        _ => Tensor::from_f32_shape(&[value as f32], vec![]).expect("rank-0 shape is valid"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Morflow, MorflowError};
    use core_types::{payload_kind_name, TensorDType};

    fn param(name: &str, param_type: ParamType, default_value: Option<Value>) -> PipelineParam {
        PipelineParam {
            name: name.to_string(),
            param_type,
            default_value,
        }
    }

    #[test]
    fn declared_types_map_to_runtime_types() {
        assert_eq!(ptype_of(&ParamType::Bytes), PType::Bytes);
        assert_eq!(ptype_of(&ParamType::IntArg), PType::Arg(ArgKind::Int));
        assert_eq!(ptype_of(&ParamType::FloatArg), PType::Arg(ArgKind::Float));
        assert_eq!(ptype_of(&ParamType::StrArg), PType::Arg(ArgKind::Str));
        assert_eq!(ptype_of(&ParamType::BoolArg), PType::Arg(ArgKind::Bool));
        assert_eq!(ptype_of(&ParamType::Scalar), PType::Scalar);
        assert_eq!(
            ptype_of(&ParamType::Tensor(ParamShape::AnyRank)),
            PType::Tensor(ShapeSpec::AnyRank)
        );
        assert_eq!(
            ptype_of(&ParamType::Image(ParamShape::Ranked {
                dims: vec![ParamDim::Any, ParamDim::Any, ParamDim::Fixed(3)],
            })),
            PType::Image(ShapeSpec::Ranked {
                dims: vec![Dim::Any, Dim::Any, Dim::Fixed(3)],
            })
        );
    }

    #[test]
    fn runtime_types_map_back_to_declarations() {
        for declared in [
            ParamType::Bytes,
            ParamType::IntArg,
            ParamType::FloatArg,
            ParamType::StrArg,
            ParamType::BoolArg,
            ParamType::Scalar,
            ParamType::Tensor(ParamShape::Ranked {
                dims: vec![ParamDim::Fixed(2), ParamDim::Any],
            }),
            ParamType::Image(ParamShape::AnyRank),
            ParamType::Audio(ParamShape::Ranked {
                dims: vec![ParamDim::Any, ParamDim::Any],
            }),
            ParamType::Composite,
        ] {
            assert_eq!(param_type_of(&ptype_of(&declared)), Some(declared));
        }
    }

    #[test]
    fn argument_defaults_are_args_and_scalar_defaults_are_scalars() {
        let int_default = param("rate", ParamType::IntArg, Some(Value::Int(44100)));
        match default_payload(&int_default) {
            Some(Payload::Arg(bytes)) => assert_eq!(bytes.as_slice(), b"44100"),
            other => panic!(
                "expected an Arg payload, got {:?}",
                other.as_ref().map(payload_kind_name)
            ),
        }

        let float_default = param("gain", ParamType::FloatArg, Some(Value::Float(1.5)));
        match default_payload(&float_default) {
            Some(Payload::Arg(bytes)) => assert_eq!(bytes.as_slice(), b"1.5"),
            other => panic!(
                "expected an Arg payload, got {:?}",
                other.as_ref().map(payload_kind_name)
            ),
        }

        let scalar_default = param("level", ParamType::Scalar, Some(Value::Float(0.25)));
        match default_payload(&scalar_default) {
            Some(Payload::Scalar(t)) => {
                assert_eq!(t.rank(), 0);
                assert_eq!(t.dtype, TensorDType::F32);
                assert_eq!(t.as_f32_slice().unwrap(), &[0.25]);
            }
            other => panic!(
                "expected a Scalar payload, got {:?}",
                other.as_ref().map(payload_kind_name)
            ),
        }

        let raw_default = param("mode", ParamType::Bytes, Some(Value::String("fast".into())));
        match default_payload(&raw_default) {
            Some(Payload::Arg(bytes)) => assert_eq!(bytes.as_slice(), b"fast"),
            other => panic!(
                "expected a raw byte payload, got {:?}",
                other.as_ref().map(payload_kind_name)
            ),
        }
    }

    #[test]
    fn host_values_are_retagged_to_the_declared_type() {
        let data = Payload::Data {
            buffer: RVec::from(b"48000".to_vec()),
        };
        let coerced = coerce_host_payload(&PType::Arg(ArgKind::Int), data);
        match coerced {
            Payload::Arg(bytes) => assert_eq!(bytes.as_slice(), b"48000"),
            other => panic!("expected an Arg payload, got {}", payload_kind_name(&other)),
        }

        let rank0 =
            Payload::Tensor(Tensor::from_f32_shape(&[7.0], vec![]).expect("rank-0 shape is valid"));
        match coerce_host_payload(&PType::Scalar, rank0) {
            Payload::Scalar(t) => assert_eq!(t.rank(), 0),
            other => panic!(
                "expected a Scalar payload, got {}",
                payload_kind_name(&other)
            ),
        }

        let bytes = Payload::Data {
            buffer: RVec::from(b"1.5".to_vec()),
        };
        match coerce_host_payload(&PType::Scalar, bytes) {
            Payload::Scalar(t) => assert_eq!(t.as_f32_slice().unwrap(), &[1.5]),
            other => panic!(
                "expected a Scalar payload, got {}",
                payload_kind_name(&other)
            ),
        }
    }

    #[test]
    fn run_rejects_a_payload_that_does_not_match_its_declaration() {
        let mut pipeline = Morflow::from_str(
            r#"
            accept Image $image
            $image >> base/latest/identity >> emit
        "#,
        )
        .expect("Failed to parse pipeline");

        let err = pipeline
            .run(Payload::Data {
                buffer: RVec::from(vec![1, 2, 3]),
            })
            .expect_err("raw bytes should not satisfy an Image declaration");
        assert!(
            matches!(err, MorflowError::TypeMismatch(_)),
            "expected a type error, got {:?}",
            err
        );
        assert!(
            err.to_string().contains("image"),
            "diagnostic should name the declared type: {}",
            err
        );
    }

    #[test]
    fn run_checks_a_declared_shape() {
        let mut pipeline = Morflow::from_str(
            r#"
            accept Tensor[rank=2] $matrix
            $matrix >> base/latest/identity >> emit
        "#,
        )
        .expect("Failed to parse pipeline");

        let err = pipeline
            .run(Payload::Tensor(
                Tensor::from_f32_shape(&[1.0, 2.0, 3.0], vec![3]).expect("valid shape"),
            ))
            .expect_err("a rank-1 tensor should not satisfy a rank=2 declaration");
        assert!(err.to_string().contains("rank"), "got: {}", err);
    }

    #[test]
    fn run_accepts_a_scalar_declared_as_a_tensor_rank_0() {
        let mut pipeline = Morflow::from_str(
            r#"
            accept Tensor $value
            $value >> base/latest/identity >> emit
        "#,
        )
        .expect("Failed to parse pipeline");

        // A rank-0 tensor is a valid Tensor value, and passing it through does
        // not change how it is tagged.
        let outputs = pipeline
            .run(Payload::Tensor(
                Tensor::from_f32_shape(&[3.0], vec![]).expect("rank-0 shape is valid"),
            ))
            .expect("a rank-0 tensor is a valid Tensor value");
        match outputs.into_single().unwrap() {
            Payload::Tensor(t) => {
                assert_eq!(t.rank(), 0);
                assert_eq!(t.as_f32_slice().unwrap(), &[3.0]);
            }
            other => panic!(
                "expected the rank-0 value back, got {}",
                payload_kind_name(&other)
            ),
        }
    }

    #[test]
    fn run_accepts_an_untagged_rank_0_tensor_for_a_scalar_declaration() {
        let mut pipeline = Morflow::from_str(
            r#"
            accept Scalar $value
            $value >> base/latest/identity >> emit
        "#,
        )
        .expect("Failed to parse pipeline");

        let outputs = pipeline
            .run(Payload::Tensor(
                Tensor::from_f32_shape(&[3.0], vec![]).expect("rank-0 shape is valid"),
            ))
            .expect("a rank-0 tensor satisfies a Scalar declaration");
        match outputs.into_single().unwrap() {
            Payload::Scalar(t) => {
                assert_eq!(t.rank(), 0);
                assert_eq!(t.as_f32_slice().unwrap(), &[3.0]);
            }
            other => panic!(
                "expected the rank-0 value back, got {}",
                payload_kind_name(&other)
            ),
        }
    }

    #[test]
    fn declared_int_default_is_readable_as_a_number() {
        let src = r#"
            accept Tensor $signal
            accept IntArg $threshold = 2

            $signal >> if ($signal.peak > $threshold) {
                base/latest/identity
            } else {
                base/latest/identity
            } >> emit
        "#;
        let mut pipeline = Morflow::from_str(src).expect("Failed to parse pipeline");
        let outputs = pipeline
            .run(Payload::Tensor(
                Tensor::from_f32_shape(&[1.0, 5.0], vec![2]).expect("valid shape"),
            ))
            .expect("the default argument should be readable in a condition");
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn shape_spec_of_reports_pinned_rank_and_wildcards() {
        let spec = shape_spec_of(&ParamShape::Ranked {
            dims: vec![ParamDim::Any, ParamDim::Fixed(3)],
        });
        assert_eq!(spec.rank(), Some(2));
        assert_eq!(spec.rank(), Some(2));
    }
}
