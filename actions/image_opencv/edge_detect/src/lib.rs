mod interface {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../plugins/opencv-bridge/src/interface.rs"
    ));
}

use core_types::shapecheck::{PreparedArgs, PreparedValue};
use core_types::{
    ActionArgs, InputDescriptor, PreparedData, RString, ShapeCheckResult, Tensor, TensorDType,
    Tuple2,
};
use core_types::{DataType, Payload, Shape, ShapeResult};
use interface::{NativeProcess, Options};
use std::ffi::{c_char, CStr};

const OPERATION: i32 = interface::operation!(edge_detect);

fn shape_impl(input: Shape, _args: PreparedArgs) -> ShapeResult {
    use core_types::contract::{self};
    contract::finish((|| {
        if !matches!(input.rank(), 2 | 3) {
            return Err("Image operations require rank 2 or 3".into());
        }
        contract::image_nonempty(&input)?;
        Ok(input)
    })())
}

pub fn get_output_shape<A: Into<PreparedArgs>>(input: Shape, args: A) -> ShapeResult {
    let args = args.into();
    shape_impl(input, args)
}

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn shapecheck(
    input: core_types::InputDescriptor,
    args: core_types::ActionArgs,
) -> core_types::ShapeCheckResult {
    analyze(input, args)
}

#[no_mangle]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    process_impl(payload, prepared)
}

#[no_mangle]
pub extern "C" fn get_required_plugins() -> core_types::RVec<core_types::plugins::PluginRequirement>
{
    vec![core_types::plugins::PluginRequirement {
        name: interface::PLUGIN_NAME.into(),
        version: interface::PLUGIN_VERSION.into(),
    }]
    .into()
}

fn backend(prepared: &PreparedData) -> Result<NativeProcess, RString> {
    let runtime = prepared
        .runtime
        .as_ref()
        .into_option()
        .ok_or_else(|| RString::from("OpenCV requires an engine-provided plugin context"))?;
    let address = runtime.symbol(interface::PLUGIN_NAME, interface::PROCESS_SYMBOL)?;
    // The versioned plugin contract defines this exact native function signature.
    Ok(unsafe { std::mem::transmute::<usize, NativeProcess>(address) })
}

fn text<'a>(
    args: &'a PreparedArgs,
    keys: &[&str],
    pos: Option<usize>,
    default: &'a str,
) -> &'a str {
    keys.iter()
        .find_map(|key| args.get_named(key))
        .or_else(|| pos.and_then(|i| args.positional.get(i).map(|s| s.as_str())))
        .unwrap_or(default)
}

fn number(
    args: &PreparedArgs,
    keys: &[&str],
    pos: Option<usize>,
    default: &str,
) -> Result<f64, RString> {
    let value = text(args, keys, pos, default)
        .parse::<f32>()
        .map_err(|e| RString::from(e.to_string()))?;
    if !value.is_finite() {
        return Err("Image arguments must be finite".into());
    }
    Ok(f64::from(value))
}

fn options(args: &PreparedArgs) -> Result<Options, RString> {
    let mut o = Options {
        operation: OPERATION,
        mode: 0,
        radius: 0,
        iterations: 1,
        shape: 0,
        sigma: 1.0,
        strength: 1.0,
        angle: 90.0,
        fill: 0.0,
    };

    o.strength = number(args, &["strength", "scale"], None, "1")?;
    o.mode = match text(args, &["mode", "filter"], Some(0), "sobel")
        .to_lowercase()
        .as_str()
    {
        "sobel_x" | "sobelx" | "dx" => 1,
        "sobel_y" | "sobely" | "dy" => 2,
        "laplacian" | "laplace" => 3,
        "prewitt" => 4,
        _ => 0,
    };

    Ok(o)
}

fn analyze(input: InputDescriptor, args: ActionArgs) -> ShapeCheckResult {
    const NAME: &str = "edge_detect";
    if let core_types::shapecheck::Metadata::Tensor { dtype } = &input.metadata {
        if !matches!(dtype, TensorDType::F32 | TensorDType::U8) {
            return ShapeCheckResult::Invalid {
                reason: "OpenCV actions require F32 or U8 tensors".into(),
            };
        }
    }
    if let Some(dims) = input.value.shape() {
        if core_types::contract::image_dims(dims)
            .is_ok_and(|(_, _, c, _)| c.known().is_some_and(|n| n > 512))
        {
            return ShapeCheckResult::Invalid {
                reason: format!("OpenCV {NAME} supports at most {} channels", 512).into(),
            };
        }
        if dims
            .dims()
            .iter()
            .any(|d| d.known().is_some_and(|n| n > i32::MAX as usize))
        {
            return ShapeCheckResult::Invalid {
                reason: "OpenCV image dimensions must fit signed 32-bit integers".into(),
            };
        }
    }
    let dynamic = args
        .positional
        .iter()
        .chain(args.named.iter().map(|Tuple2(_, v)| v))
        .any(|v| v.starts_with('$'));
    let opts = if dynamic {
        None
    } else {
        match options(&PreparedArgs::from(args.clone())) {
            Ok(o) => Some(o),
            Err(reason) => return ShapeCheckResult::Invalid { reason },
        }
    };
    let mut result = core_types::shapecheck::analyze(
        input,
        args,
        NAME,
        DataType::Tensor,
        DataType::Tensor,
        Some(get_output_shape),
        None,
    );
    if let (ShapeCheckResult::Ready { prepared, .. }, Some(opts)) = (&mut result, opts) {
        prepared.fields.push(Tuple2(
            "mode".into(),
            PreparedValue::Unsigned(opts.mode as u64),
        ));
        prepared.fields.push(Tuple2(
            "strength".into(),
            PreparedValue::Float(opts.strength),
        ));
        if prepared
            .output_dims()
            .is_ok_and(|dims| dims.iter().any(|n| *n > i32::MAX as usize))
        {
            return ShapeCheckResult::Invalid {
                reason: "OpenCV output dimensions must fit signed 32-bit integers".into(),
            };
        }
    }
    result
}

fn process_impl(payload: Payload, prepared: PreparedData) -> Payload {
    match run(payload, prepared) {
        Ok(tensor) => Payload::Tensor(tensor),
        Err(error) => Payload::Error(error),
    }
}

fn run(payload: Payload, prepared: PreparedData) -> Result<Tensor, RString> {
    let backend = backend(&prepared)?;
    let payload = core_types::contract::image_input(payload, false)?.into_unwrapped();
    let Payload::Tensor(tensor) = payload else {
        return Err("OpenCV actions require Tensor input".into());
    };
    let output_shape = prepared.output_dims()?;
    if tensor
        .shape
        .iter()
        .chain(output_shape.iter())
        .any(|n| *n == 0 || *n > i32::MAX as usize)
    {
        return Err("OpenCV dimensions must be positive signed 32-bit integers".into());
    }
    let (_, _, _, chw) =
        core_types::contract::image_dims(&Shape::new(tensor.shape.iter().copied()))
            .map_err(|e| RString::from(format!("{e:?}")))?;
    let input = if chw {
        tensor.permute(&[1, 2, 0])?
    } else {
        tensor
    };
    let owned;
    let bytes = if input.is_contiguous() {
        input
            .as_bytes()
            .ok_or_else(|| RString::from("Invalid contiguous image storage"))?
    } else {
        owned = input.to_contiguous_bytes();
        owned.as_slice()
    };
    let dims = input.shape.as_slice();
    let (h, w, c) = (dims[0], dims[1], if dims.len() == 3 { dims[2] } else { 1 });
    if c > 512 {
        return Err(RString::from(
            "OpenCV edge_detect channel count exceeds its supported limit",
        ));
    }
    let (oh, ow) = if chw {
        (output_shape[1], output_shape[2])
    } else {
        (output_shape[0], output_shape[1])
    };
    let count = oh
        .checked_mul(ow)
        .and_then(|n| n.checked_mul(c))
        .ok_or_else(|| RString::from("OpenCV output size overflows"))?;
    let o = Options {
        operation: OPERATION,
        mode: prepared
            .unsigned("mode")
            .and_then(|n| i32::try_from(n).ok())
            .ok_or_else(|| RString::from("Missing mode execution plan"))?,
        radius: 0,
        iterations: 1,
        shape: 0,
        sigma: 1.0,
        strength: prepared
            .float("strength")
            .ok_or_else(|| RString::from("Missing strength execution plan"))?,
        angle: 90.0,
        fill: 0.0,
    };
    let mut floats = if input.dtype == TensorDType::F32 {
        vec![0f32; count]
    } else {
        Vec::new()
    };
    let mut integers = if input.dtype == TensorDType::U8 {
        vec![0u8; count]
    } else {
        Vec::new()
    };
    let pointer = if input.dtype == TensorDType::F32 {
        floats.as_mut_ptr().cast()
    } else {
        integers.as_mut_ptr().cast()
    };
    let mut error = [0 as c_char; 2048];
    // Slices live throughout the synchronous call. Checked shape products size the
    // output buffer; input storage was validated by Tensor and normalized above.
    let status = unsafe {
        backend(
            bytes.as_ptr().cast(),
            pointer,
            h as i32,
            w as i32,
            c as i32,
            if input.dtype == TensorDType::F32 {
                5
            } else {
                0
            },
            oh as i32,
            ow as i32,
            &o,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if status != 0 {
        return Err(format!(
            "OpenCV: {}",
            unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy()
        )
        .into());
    }
    let native_shape = if dims.len() == 2 {
        vec![oh, ow]
    } else {
        vec![oh, ow, c]
    };
    let result = if input.dtype == TensorDType::F32 {
        Tensor::from_f32_vec(floats, native_shape)?
    } else {
        Tensor::from_rvec_u8(integers.into(), native_shape, TensorDType::U8)?
    };
    if chw {
        result.permute(&[2, 0, 1])
    } else {
        Ok(result)
    }
}
