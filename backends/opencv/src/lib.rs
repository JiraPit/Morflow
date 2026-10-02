//! Typed action SDK for the pipeline-selected OpenCV plugin.
use core_types::shapecheck::{PreparedArgs, PreparedValue};
use core_types::{
    ActionArgs, DataType, InputDescriptor, Payload, PreparedData, RString, Shape, ShapeCheckResult,
    ShapeResult, Tensor, TensorDType, Tuple2,
};
use std::ffi::{c_char, c_void, CStr};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Options {
    pub operation: i32,
    pub mode: i32,
    pub radius: i32,
    pub iterations: i32,
    pub shape: i32,
    pub sigma: f64,
    pub strength: f64,
    pub angle: f64,
    pub fill: f64,
}
pub type NativeProcess = unsafe extern "C" fn(
    *const c_void,
    *mut c_void,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const Options,
    *mut c_char,
    usize,
) -> i32;
pub fn required_plugins() -> core_types::RVec<core_types::plugins::PluginRequirement> {
    vec![core_types::plugins::PluginRequirement {
        name: "opencv-bridge".into(),
        version: "^0.1.0".into(),
    }]
    .into()
}
fn backend(prepared: &PreparedData) -> Result<NativeProcess, RString> {
    let runtime = prepared
        .runtime
        .as_ref()
        .into_option()
        .ok_or_else(|| RString::from("OpenCV requires an engine-provided plugin context"))?;
    let address = runtime.symbol("opencv-bridge", "morflow_opencv_process")?;
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
fn options(name: &str, args: &PreparedArgs) -> Result<Options, RString> {
    let number = |keys: &[&str], pos, default| -> Result<f64, RString> {
        let value = text(args, keys, pos, default)
            .parse::<f32>()
            .map_err(|e| RString::from(e.to_string()))?;
        if !value.is_finite() {
            return Err("Image arguments must be finite".into());
        }
        Ok(f64::from(value))
    };
    let integer = |keys: &[&str], default: &str| -> Result<i32, RString> {
        let value = text(args, keys, None, default)
            .parse::<usize>()
            .map_err(|e| RString::from(e.to_string()))?;
        i32::try_from(value).map_err(|_| "OpenCV arguments must fit a signed 32-bit integer".into())
    };
    let sigma = number(
        &["sigma"],
        if name == "gaussian_blur" {
            Some(0)
        } else {
            None
        },
        "1",
    )?
    .max(0.01);
    let default_radius = f64::from((3.0_f32 * sigma as f32).ceil().max(1.0));
    if default_radius > f64::from((i32::MAX - 1) / 2) {
        return Err("Blur radius exceeds OpenCV limits".into());
    }
    let radius = integer(&["radius"], &(default_radius as i32).to_string())?;
    if radius > (i32::MAX - 1) / 2 {
        return Err("Blur radius exceeds OpenCV limits".into());
    }
    let mut o = Options {
        operation: 0,
        mode: 0,
        radius,
        iterations: 1,
        shape: 0,
        sigma,
        strength: 1.0,
        angle: 90.0,
        fill: 0.0,
    };
    match name {
        "rotate" => {
            o.operation = 5;
            let angle = number(&["angle", "angle_deg"], Some(0), "90")? as f32;
            o.angle = f64::from(((angle % 360.0) + 360.0) % 360.0);
            o.fill = number(&["fill", "fill_value"], None, "0")?;
        }
        "resize" => {
            o.mode = match text(args, &["filter"], None, "bilinear")
                .to_lowercase()
                .as_str()
            {
                "nearest" | "neighbor" => 0,
                "bicubic" | "cubic" => 2,
                "area" | "box" => 3,
                _ => 1,
            }
        }
        "gaussian_blur" => {
            o.operation = 1;
            o.mode = i32::from(
                text(args, &["mode", "type"], None, "gaussian").eq_ignore_ascii_case("box"),
            );
        }
        "morphology" => {
            o.operation = 2;
            let k = integer(&["kernel_size", "ksize"], "3")?.max(1);
            o.radius = k / 2;
            o.iterations = integer(&["iterations", "iter"], "1")?.max(1);
            o.mode = match text(args, &["op"], Some(0), "dilate")
                .to_lowercase()
                .as_str()
            {
                "erode" | "erosion" => 0,
                "open" | "opening" => 2,
                "close" | "closing" => 3,
                "gradient" | "grad" => 4,
                _ => 1,
            };
            o.shape = match text(args, &["shape"], None, "rect").to_lowercase().as_str() {
                "cross" => 1,
                "ellipse" | "circle" => 2,
                _ => 0,
            };
        }
        "edge_detect" => {
            o.operation = 3;
            o.strength = number(&["strength", "scale"], None, "1")?;
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
        }
        "sharpen" => {
            o.operation = 4;
            o.strength = number(&["strength", "amount"], Some(0), "1")?;
        }
        _ => return Err("Unknown OpenCV action".into()),
    }
    Ok(o)
}
pub fn analyze(
    input: InputDescriptor,
    args: ActionArgs,
    name: &str,
    shape: fn(Shape, PreparedArgs) -> ShapeResult,
) -> ShapeCheckResult {
    if let core_types::shapecheck::Metadata::Tensor { dtype } = &input.metadata {
        if !matches!(dtype, TensorDType::F32 | TensorDType::U8) {
            return ShapeCheckResult::Invalid {
                reason: "OpenCV actions require F32 or U8 tensors".into(),
            };
        }
    }
    if let Some(dims) = input.value.shape() {
        if name == "rotate"
            && dims
                .dims()
                .iter()
                .any(|d| d.known().is_some_and(|n| n >= 32767))
        {
            return ShapeCheckResult::Invalid {
                reason: "OpenCV rotation dimensions must be below 32767".into(),
            };
        }
        if core_types::contract::image_dims(dims).is_ok_and(|(_, _, c, _)| {
            c.known().is_some_and(|n| {
                n > if matches!(name, "rotate" | "resize") {
                    4
                } else {
                    512
                }
            })
        }) {
            return ShapeCheckResult::Invalid {
                reason: format!(
                    "OpenCV {name} supports at most {} channels",
                    if matches!(name, "rotate" | "resize") {
                        4
                    } else {
                        512
                    }
                )
                .into(),
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
        match options(name, &PreparedArgs::from(args.clone())) {
            Ok(o) => Some(o),
            Err(reason) => return ShapeCheckResult::Invalid { reason },
        }
    };
    let mut result = core_types::shapecheck::analyze(
        input,
        args,
        name,
        DataType::Tensor,
        DataType::Tensor,
        Some(shape),
        None,
    );
    if let (ShapeCheckResult::Ready { prepared, .. }, Some(opts)) = (&mut result, opts) {
        for (key, value) in [
            ("operation", opts.operation),
            ("mode", opts.mode),
            ("radius", opts.radius),
            ("iterations", opts.iterations),
            ("shape", opts.shape),
        ] {
            prepared
                .fields
                .push(Tuple2(key.into(), PreparedValue::Unsigned(value as u64)));
        }
        prepared
            .fields
            .push(Tuple2("angle".into(), PreparedValue::Float(opts.angle)));
        prepared
            .fields
            .push(Tuple2("fill".into(), PreparedValue::Float(opts.fill)));
        prepared
            .fields
            .push(Tuple2("sigma".into(), PreparedValue::Float(opts.sigma)));
        prepared.fields.push(Tuple2(
            "strength".into(),
            PreparedValue::Float(opts.strength),
        ));
        if name == "rotate"
            && prepared
                .output_dims()
                .is_ok_and(|dims| dims.iter().any(|n| *n >= 32767))
        {
            return ShapeCheckResult::Invalid {
                reason: "OpenCV rotation output dimensions must be below 32767".into(),
            };
        }
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
pub fn process(name: &str, payload: Payload, prepared: PreparedData) -> Payload {
    match run(name, payload, prepared) {
        Ok(tensor) => Payload::Tensor(tensor),
        Err(error) => Payload::Error(error),
    }
}
fn run(name: &str, payload: Payload, prepared: PreparedData) -> Result<Tensor, RString> {
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
    if c > if matches!(name, "rotate" | "resize") {
        4
    } else {
        512
    } {
        return Err(format!("OpenCV {name} channel count exceeds its supported limit").into());
    }
    let (oh, ow) = if chw {
        (output_shape[1], output_shape[2])
    } else {
        (output_shape[0], output_shape[1])
    };
    if name == "rotate" && [h, w, oh, ow].iter().any(|n| *n >= 32767) {
        return Err("OpenCV rotation dimensions must be below 32767".into());
    }
    let count = oh
        .checked_mul(ow)
        .and_then(|n| n.checked_mul(c))
        .ok_or_else(|| RString::from("OpenCV output size overflows"))?;
    let field = |name| {
        prepared
            .unsigned(name)
            .and_then(|n| i32::try_from(n).ok())
            .ok_or_else(|| RString::from("Missing OpenCV execution plan"))
    };
    let o = Options {
        operation: field("operation")?,
        mode: field("mode")?,
        radius: field("radius")?,
        iterations: field("iterations")?,
        shape: field("shape")?,
        angle: prepared
            .float("angle")
            .ok_or_else(|| RString::from("Missing angle"))?,
        fill: prepared
            .float("fill")
            .ok_or_else(|| RString::from("Missing fill"))?,
        sigma: prepared
            .float("sigma")
            .ok_or_else(|| RString::from("Missing sigma"))?,
        strength: prepared
            .float("strength")
            .ok_or_else(|| RString::from("Missing strength"))?,
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
