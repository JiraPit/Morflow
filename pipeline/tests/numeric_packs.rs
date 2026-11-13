//! Hidden tests for the math_basics, tensor_stats and nn_basics packs
//! (two-tensor arithmetic with broadcasting, its shape predictions, and support
//! for every tensor dtype, NaN like NumPy, exact integer statistics),
//! Python-style `$var[...]` slicing of tensors, images, audio and bytes, number
//! literals, `if`/`route` conditions (exact for every dtype, slices included),
//! `each(axis=N)` loops, and everything `morflow check` can tell from the
//! declared types.
//!
//! Every case lives in a JSON file under `numeric_packs_cases/`. A case is one of:
//! - `run`: a `.morf` pipeline, its inputs and the expected output (dtype,
//!   shape, payload kind and values) or an expected error;
//! - `check`: a `.morf` pipeline that `morflow check` must accept or reject;
//! - `contract`: an action's shape prediction for a partially known input.
//!
//! Each JSON file is one `#[test]`; a failing test lists every failing case.

use core_types::{
    ActionArgs, AudioChannelLayout, AudioLayout, DataType, Dimension, ImageLayout, RString, Shape,
    TensorDType, Tuple2, ValueShape, ValueShapeResult,
};
use morflow::{ActionIdentity, ActionRegistry, Audio, ColorSpace, Image, Morflow, Payload, Tensor};
use serde_json::Value;
use std::path::PathBuf;

fn cases_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/numeric_packs_cases")
}

fn dtype(name: &str) -> TensorDType {
    match name {
        "f32" => TensorDType::F32,
        "f64" => TensorDType::F64,
        "u8" => TensorDType::U8,
        "i8" => TensorDType::I8,
        "i16" => TensorDType::I16,
        "i32" => TensorDType::I32,
        "i64" => TensorDType::I64,
        "u32" => TensorDType::U32,
        "u64" => TensorDType::U64,
        other => panic!("unknown dtype {other}"),
    }
}

fn dtype_name(dtype: TensorDType) -> &'static str {
    match dtype {
        TensorDType::F32 => "f32",
        TensorDType::F64 => "f64",
        TensorDType::U8 => "u8",
        TensorDType::I8 => "i8",
        TensorDType::I16 => "i16",
        TensorDType::I32 => "i32",
        TensorDType::I64 => "i64",
        TensorDType::U32 => "u32",
        TensorDType::U64 => "u64",
    }
}

/// A float from JSON: a number, or "nan", "inf", "-inf".
fn float(value: &Value) -> f64 {
    match value {
        Value::String(s) => match s.as_str() {
            "nan" => f64::NAN,
            "inf" => f64::INFINITY,
            "-inf" => f64::NEG_INFINITY,
            other => panic!("bad float {other}"),
        },
        other => other.as_f64().expect("number"),
    }
}

fn integer(value: &Value) -> i128 {
    value
        .as_i64()
        .map(i128::from)
        .or_else(|| value.as_u64().map(i128::from))
        .expect("integer")
}

fn tensor(spec: &Value) -> Tensor {
    let dt = dtype(spec["dtype"].as_str().unwrap());
    let shape: Vec<usize> = spec["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d.as_u64().unwrap() as usize)
        .collect();
    let mut bytes = Vec::new();
    for v in spec["values"].as_array().unwrap() {
        match dt {
            TensorDType::F32 => bytes.extend((float(v) as f32).to_ne_bytes()),
            TensorDType::F64 => bytes.extend(float(v).to_ne_bytes()),
            TensorDType::U8 => bytes.extend((integer(v) as u8).to_ne_bytes()),
            TensorDType::I8 => bytes.extend((integer(v) as i8).to_ne_bytes()),
            TensorDType::I16 => bytes.extend((integer(v) as i16).to_ne_bytes()),
            TensorDType::I32 => bytes.extend((integer(v) as i32).to_ne_bytes()),
            TensorDType::I64 => bytes.extend((integer(v) as i64).to_ne_bytes()),
            TensorDType::U32 => bytes.extend((integer(v) as u32).to_ne_bytes()),
            TensorDType::U64 => bytes.extend((integer(v) as u64).to_ne_bytes()),
        }
    }
    Tensor::from_rvec_u8(bytes.into(), shape, dt).unwrap()
}

fn color_space(name: &str) -> ColorSpace {
    match name {
        "rgb" => ColorSpace::Rgb,
        "grayscale" => ColorSpace::Grayscale,
        other => panic!("unknown color space {other}"),
    }
}

fn image_layout(name: &str) -> ImageLayout {
    match name {
        "hwc" => ImageLayout::Hwc,
        "chw" => ImageLayout::Chw,
        other => panic!("unknown image layout {other}"),
    }
}

fn audio_layout(name: &str) -> AudioLayout {
    match name {
        "planar" => AudioLayout::Planar,
        "interleaved" => AudioLayout::Interleaved,
        other => panic!("unknown audio layout {other}"),
    }
}

fn byte_values(spec: &Value) -> Vec<u8> {
    spec.as_array()
        .unwrap()
        .iter()
        .map(|b| b.as_u64().unwrap() as u8)
        .collect()
}

fn payload(spec: &Value) -> Payload {
    if let Some(t) = spec.get("tensor") {
        Payload::Tensor(tensor(t))
    } else if let Some(t) = spec.get("scalar") {
        Payload::Scalar(tensor(t))
    } else if let Some(items) = spec.get("composite") {
        Payload::Composite(items.as_array().unwrap().iter().map(payload).collect())
    } else if let Some(i) = spec.get("image") {
        let image = Image::new(
            tensor(i),
            color_space(i["color"].as_str().unwrap()),
            image_layout(i["layout"].as_str().unwrap()),
        );
        Payload::Image(image.unwrap())
    } else if let Some(a) = spec.get("audio") {
        let t = tensor(a);
        let layout = audio_layout(a["layout"].as_str().unwrap());
        let channels = match (t.rank(), layout) {
            (1, _) => 1,
            (_, AudioLayout::Planar) => t.shape[0],
            (_, AudioLayout::Interleaved) => t.shape[1],
        };
        let audio = Audio::new(
            t,
            a["rate"].as_u64().unwrap() as u32,
            AudioChannelLayout::from_channel_count(channels),
            layout,
        );
        Payload::Audio(audio.unwrap())
    } else if let Some(b) = spec.get("bytes") {
        Payload::Data {
            buffer: byte_values(b).into(),
        }
    } else if let Some(text) = spec.get("arg") {
        Payload::Arg(text.as_str().unwrap().as_bytes().to_vec().into())
    } else {
        panic!("bad payload {spec}")
    }
}

enum Num {
    Int(i128),
    Float(f64),
}

fn values(t: &Tensor) -> Vec<Num> {
    let bytes = t.to_contiguous_bytes();
    let bytes = bytes.as_slice();
    macro_rules! read {
        ($ty:ty, $wrap:expr) => {
            bytes
                .chunks_exact(std::mem::size_of::<$ty>())
                .map(|c| $wrap(<$ty>::from_ne_bytes(c.try_into().unwrap())))
                .collect()
        };
    }
    match t.dtype {
        TensorDType::F32 => read!(f32, |v: f32| Num::Float(v as f64)),
        TensorDType::F64 => read!(f64, Num::Float),
        TensorDType::U8 => read!(u8, |v| Num::Int(v as i128)),
        TensorDType::I8 => read!(i8, |v| Num::Int(v as i128)),
        TensorDType::I16 => read!(i16, |v| Num::Int(v as i128)),
        TensorDType::I32 => read!(i32, |v| Num::Int(v as i128)),
        TensorDType::I64 => read!(i64, |v| Num::Int(v as i128)),
        TensorDType::U32 => read!(u32, |v| Num::Int(v as i128)),
        TensorDType::U64 => read!(u64, |v| Num::Int(v as i128)),
    }
}

fn describe(payload: &Payload) -> Value {
    match payload.unwrap_payload() {
        Payload::Tensor(t) | Payload::Scalar(t) => {
            let kind = if matches!(payload.unwrap_payload(), Payload::Scalar(_)) {
                "scalar"
            } else {
                "tensor"
            };
            let vals: Vec<Value> = values(t)
                .into_iter()
                .map(|n| match n {
                    Num::Int(i) => serde_json::json!(i as i64),
                    Num::Float(f) if f.is_nan() => Value::from("nan"),
                    Num::Float(f) if f.is_infinite() => {
                        Value::from(if f > 0.0 { "inf" } else { "-inf" })
                    }
                    Num::Float(f) => serde_json::json!(f),
                })
                .collect();
            serde_json::json!({"kind": kind, "dtype": dtype_name(t.dtype), "shape": t.shape.as_slice(), "values": vals})
        }
        Payload::Image(i) => {
            serde_json::json!({"kind": "image", "color": format!("{:?}", i.color_space), "layout": format!("{:?}", i.layout), "value": describe(&Payload::Tensor(i.tensor.clone()))})
        }
        Payload::Audio(a) => {
            serde_json::json!({"kind": "audio", "rate": a.sample_rate, "layout": format!("{:?}", a.layout), "value": describe(&Payload::Tensor(a.tensor.clone()))})
        }
        Payload::Data { buffer } => {
            serde_json::json!({"kind": "bytes", "values": buffer.as_slice()})
        }
        Payload::Error(e) => serde_json::json!({"error": e.to_string()}),
        other => serde_json::json!({"other": core_types::payload_kind_name(other)}),
    }
}

/// Compare an output with its expectation; `Err` explains the first difference.
fn compare(actual: &Payload, expected: &Value) -> Result<(), String> {
    let want_kind = expected.get("kind").and_then(Value::as_str);
    let (kind, t) = match actual.unwrap_payload() {
        Payload::Tensor(t) => ("tensor", t),
        Payload::Scalar(t) => ("scalar", t),
        Payload::Image(i) if want_kind == Some("image") => {
            if i.color_space != color_space(expected["color"].as_str().unwrap())
                || i.layout != image_layout(expected["layout"].as_str().unwrap())
            {
                return Err(format!(
                    "expected a {} {} image, got {:?} {:?}",
                    expected["color"], expected["layout"], i.color_space, i.layout
                ));
            }
            ("image", &i.tensor)
        }
        Payload::Audio(a) if want_kind == Some("audio") => {
            if u64::from(a.sample_rate) != expected["rate"].as_u64().unwrap()
                || a.layout != audio_layout(expected["layout"].as_str().unwrap())
            {
                return Err(format!(
                    "expected {} audio at {} Hz, got {:?} audio at {} Hz",
                    expected["layout"], expected["rate"], a.layout, a.sample_rate
                ));
            }
            ("audio", &a.tensor)
        }
        Payload::Data { buffer } if want_kind == Some("bytes") => {
            let want = byte_values(&expected["values"]);
            return if buffer.as_slice() == want.as_slice() {
                Ok(())
            } else {
                Err(format!(
                    "expected bytes {want:?}, got {:?}",
                    buffer.as_slice()
                ))
            };
        }
        Payload::Error(e) => return Err(format!("action returned an error: {e}")),
        other => {
            return Err(format!(
                "expected a {}, got {}",
                want_kind.unwrap_or("tensor"),
                core_types::payload_kind_name(other)
            ))
        }
    };
    if let Some(k) = want_kind {
        if k != kind {
            return Err(format!("expected a {k} payload, got a {kind}"));
        }
    }
    let want_dtype = expected["dtype"].as_str().unwrap();
    if dtype_name(t.dtype) != want_dtype {
        return Err(format!(
            "expected dtype {want_dtype}, got {}",
            dtype_name(t.dtype)
        ));
    }
    let want_shape: Vec<usize> = expected["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d.as_u64().unwrap() as usize)
        .collect();
    if t.shape.as_slice() != want_shape.as_slice() {
        return Err(format!(
            "expected shape {want_shape:?}, got {:?}",
            t.shape.as_slice()
        ));
    }
    let rtol = expected.get("rtol").map_or(0.0, float);
    let atol = expected.get("atol").map_or(0.0, float);
    let want = expected["values"].as_array().unwrap();
    let got = values(t);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let ok = match g {
            Num::Int(g) => *g == integer(w),
            Num::Float(g) => {
                let w = float(w);
                if w.is_nan() || g.is_nan() {
                    w.is_nan() && g.is_nan()
                } else if w.is_infinite() || g.is_infinite() {
                    *g == w
                } else {
                    (g - w).abs() <= atol + rtol * w.abs()
                }
            }
        };
        if !ok {
            let g = match g {
                Num::Int(v) => v.to_string(),
                Num::Float(v) => format!("{v:e}"),
            };
            return Err(format!(
                "value {i} is {g}, expected {w} (rtol {rtol}, atol {atol})"
            ));
        }
    }
    Ok(())
}

/// Runs `f`, turning a panic (a crash in the parser, checker or engine) into an error.
fn no_panic<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .unwrap_or_else(|_| Err("panicked".to_string()))
}

fn check_source(source: &str) -> Result<(), String> {
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    let path = dir.path().join("case.morf");
    std::fs::write(&path, source).map_err(|e| e.to_string())?;
    let cache = std::env::var("MORFLOW_ACTIONS_PATH")
        .ok()
        .map(PathBuf::from);
    morflow::check::check_pipeline(&path, cache.as_deref()).map_err(|e| e.to_string())
}

/// Result of loading and running a pipeline: `Err((stage, message))` on failure.
/// A panic is never an acceptable outcome, not even for a case that expects an error.
fn execute(case: &Value) -> Result<Payload, (&'static str, String)> {
    let mut stage = "load";
    let result = no_panic(|| execute_inner(case, &mut stage).map_err(|(_, e)| e));
    result.map_err(|e| (stage, e))
}

fn execute_inner(
    case: &Value,
    stage: &mut &'static str,
) -> Result<Payload, (&'static str, String)> {
    let source = case["morf"].as_str().unwrap();
    let mut pipeline = Morflow::from_str(source).map_err(|e| ("load", e.to_string()))?;
    *stage = "run";
    let inputs: Vec<Payload> = case["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(payload)
        .collect();
    let out = pipeline
        .run_args(inputs)
        .map_err(|e| ("run", e.to_string()))?
        .into_single()
        .map_err(|e| ("run", e.to_string()))?;
    if let Payload::Error(e) = out.unwrap_payload() {
        return Err(("run", e.to_string()));
    }
    Ok(out)
}

fn run_case(case: &Value) -> Result<(), String> {
    let expected = &case["expect"];
    let result = execute(case);
    if let Err((_, e)) = &result {
        if e == "panicked" {
            return Err("the pipeline panicked instead of returning an error".into());
        }
    }
    match expected.get("error").and_then(Value::as_str) {
        None => {
            let out = result.map_err(|(stage, e)| format!("{stage} failed: {e}"))?;
            compare(&out, expected)
        }
        Some("any") => match result {
            Err(_) => Ok(()),
            Ok(out) => Err(format!("expected an error, got {}", describe(&out))),
        },
        Some("static") => {
            let checked = no_panic(|| check_source(case["morf"].as_str().unwrap()));
            if checked.as_ref().is_err_and(|e| e == "panicked") {
                return Err("morflow check panicked".into());
            }
            if checked.is_ok() {
                return Err("morflow check accepted a pipeline it can prove is invalid".into());
            }
            match result {
                Err(_) => Ok(()),
                Ok(out) => Err(format!("the pipeline ran and produced {}", describe(&out))),
            }
        }
        Some(other) => panic!("bad error kind {other}"),
    }
}

fn check_case(case: &Value) -> Result<(), String> {
    let ok = no_panic(|| check_source(case["morf"].as_str().unwrap()));
    if ok.as_ref().is_err_and(|e| e == "panicked") {
        return Err("morflow check panicked".into());
    }
    match (case["ok"].as_bool().unwrap(), ok) {
        (true, Ok(())) | (false, Err(_)) => Ok(()),
        (true, Err(e)) => Err(format!("morflow check rejected it: {e}")),
        (false, Ok(())) => Err("morflow check accepted it".into()),
    }
}

fn value_shape(spec: &Value) -> ValueShape {
    if let Some(items) = spec.get("composite") {
        return ValueShape::composite(items.as_array().unwrap().iter().map(value_shape));
    }
    let (kind, dims) = if let Some(d) = spec.get("tensor") {
        (DataType::Tensor, d)
    } else if let Some(d) = spec.get("scalar") {
        (DataType::Scalar, d)
    } else {
        panic!("bad value shape {spec}")
    };
    match dims.as_array() {
        None => ValueShape::unranked(kind),
        Some(dims) => ValueShape::Leaf {
            kind,
            shape: abi_stable::std_types::ROption::RSome(Shape::new(dims.iter().map(|d| {
                d.as_u64()
                    .map_or(Dimension::Unknown, |n| Dimension::Known(n as usize))
            }))),
        },
    }
}

fn args(spec: Option<&Value>) -> ActionArgs {
    let mut named = core_types::RVec::new();
    if let Some(map) = spec.and_then(Value::as_object) {
        for (k, v) in map {
            let v = match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            named.push(Tuple2(RString::from(k.as_str()), RString::from(v)));
        }
    }
    ActionArgs {
        positional: core_types::RVec::new(),
        named,
    }
}

fn contract_case(registry: &ActionRegistry, case: &Value) -> Result<(), String> {
    let action = registry
        .get_or_load(
            &ActionIdentity::new(
                case["pack"].as_str().unwrap(),
                "latest",
                case["action"].as_str().unwrap(),
            )
            .unwrap(),
        )
        .map_err(|e| format!("cannot load action: {e}"))?;
    let input = value_shape(&case["input"]);
    let result = action
        .output_value_result(&input, &args(case.get("args")))
        .ok_or("no shape prediction")?;
    let expected = &case["expect"];
    if expected.get("invalid").is_some() {
        return match result {
            ValueShapeResult::Invalid(_) => Ok(()),
            other => Err(format!("expected the input to be rejected, got {other:?}")),
        };
    }
    if expected.get("unranked").is_some() {
        // Rank cannot be known: any prediction without a rank, or no prediction.
        return match result {
            ValueShapeResult::Unknown | ValueShapeResult::Ok(ValueShape::Unknown) => Ok(()),
            ValueShapeResult::Ok(ref v) if v.shape().is_none() && v.components().is_none() => {
                Ok(())
            }
            other => Err(format!("rank cannot be known, got {other:?}")),
        };
    }
    let ValueShapeResult::Ok(predicted) = result else {
        return Err(format!("expected a prediction, got {result:?}"));
    };
    if let Some(kind) = expected.get("kind").and_then(Value::as_str) {
        let want = if kind == "scalar" {
            DataType::Scalar
        } else {
            DataType::Tensor
        };
        match &predicted {
            ValueShape::Leaf { kind, .. } if *kind == want => {}
            other => return Err(format!("expected a {want} prediction, got {other:?}")),
        }
    }
    let want = expected["dims"].as_array().unwrap();
    // A Scalar is rank 0 whether or not the prediction spells out the empty shape.
    if want.is_empty()
        && predicted.shape().is_none()
        && matches!(predicted, ValueShape::Leaf { kind, .. } if kind == DataType::Scalar)
    {
        return Ok(());
    }
    let shape = predicted
        .shape()
        .ok_or_else(|| format!("expected a ranked prediction, got {predicted:?}"))?;
    if shape.dims().len() != want.len() {
        return Err(format!(
            "expected rank {}, predicted {:?}",
            want.len(),
            shape.dims()
        ));
    }
    for (axis, (got, want)) in shape.dims().iter().zip(want).enumerate() {
        // A number must be predicted exactly, null must stay unknown, and a
        // list allows any of its entries.
        let allowed: Vec<Option<usize>> = match want {
            Value::Array(options) => options
                .iter()
                .map(|o| o.as_u64().map(|n| n as usize))
                .collect(),
            other => vec![other.as_u64().map(|n| n as usize)],
        };
        if !allowed.contains(&got.known()) {
            return Err(format!(
                "axis {axis}: predicted {got:?}, allowed {allowed:?} (whole prediction {:?})",
                shape.dims()
            ));
        }
    }
    Ok(())
}

fn run_file(name: &str) {
    let path = cases_dir().join(format!("{name}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let cases: Vec<Value> = serde_json::from_str(&text).unwrap();
    let registry = ActionRegistry::default();
    let oracle = std::env::var_os("F2P_ORACLE_OUT");
    let mut observed = Vec::new();
    let mut failures = Vec::new();
    for case in &cases {
        let id = case["name"].as_str().unwrap();
        if oracle.is_some() {
            if case.get("morf").is_some() && case.get("inputs").is_some() {
                let got = match execute(case) {
                    Ok(out) => describe(&out),
                    Err((stage, e)) => serde_json::json!({"error": e, "stage": stage}),
                };
                observed.push(serde_json::json!({"name": id, "got": got}));
            }
            continue;
        }
        let outcome = match case["type"].as_str().unwrap() {
            "run" => run_case(case),
            "check" => check_case(case),
            "contract" => contract_case(&registry, case),
            other => panic!("bad case type {other}"),
        };
        if let Err(e) = outcome {
            failures.push(format!("  {id}: {e}"));
        }
    }
    if let Some(dir) = oracle {
        let out = PathBuf::from(dir).join(format!("{name}.json"));
        std::fs::write(out, serde_json::to_string_pretty(&observed).unwrap()).unwrap();
        return;
    }
    assert!(
        failures.is_empty(),
        "{} of {} cases in {name}.json failed:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

macro_rules! case_files {
    ($($name:ident),* $(,)?) => {
        $(#[test] fn $name() { run_file(stringify!($name)); })*
    };
}

// P2P: these pass before the change and must keep passing.
case_files!(
    // Existing F32 results stay exactly the same.
    f32_behaviour_unchanged,
    // Invalid composites are rejected (today every composite is rejected).
    binary_errors,
    // Conditions that can be evaluated still pick the right branch.
    condition_values,
);

// F2P: these fail before the change.
case_files!(
    binary_broadcast_values,
    binary_scalar_components,
    binary_dtype_promotion,
    binary_shape_predictions,
    binary_static_checks,
    math_dtypes,
    math_scalar_arguments,
    stats_dtypes,
    nn_dtypes,
    f64_precision,
    slicing_tensor_values,
    slicing_tensor_errors,
    slicing_tensor_checks,
    slicing_images,
    slicing_audio,
    slicing_bytes,
    number_literals,
    condition_errors,
    condition_checks,
    each_values,
    each_errors,
    each_checks,
    condition_exact,
    condition_slices,
    nan_values,
    integer_stats_exact,
    static_completeness,
    static_fuzz,
);
