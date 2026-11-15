use core_types::numeric::{self, FloatElement, FloatVals};
use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, Shape, ShapeResult, Tensor};

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor | DataType::Scalar
}

fn shape_impl(input: Shape, args: PreparedArgs) -> ShapeResult {
    use core_types::contract::{self, arg, axis};
    contract::finish((|| {
        let dim = match arg::<isize>(&args, &["axis", "dim"], Some(0), Some(-1)) {
            Err(core_types::contract::Error::Unknown) => return Ok(input),
            other => other?.unwrap(),
        };
        if input.rank() > 0 {
            let dim = if dim < 0 {
                (input.rank() as isize).saturating_add(dim).max(0)
            } else {
                dim
            };
            axis(dim, input.rank(), false)?;
        }
        Ok(input)
    })())
}

pub fn get_output_shape<A: Into<PreparedArgs>>(input: Shape, args: A) -> ShapeResult {
    let args = args.into();
    shape_impl(input, args)
}

#[no_mangle]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    process_impl(payload, prepared)
}

fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let inner_payload = payload.into_unwrapped();
    let axis = prepared.unsigned("axis").unwrap_or(0) as usize;

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => {
            let result = match numeric::float_vals(&tensor) {
                FloatVals::F32(vals) => compute_log_softmax::<f32>(vals, &tensor.shape, axis),
                FloatVals::F64(vals) => compute_log_softmax::<f64>(vals, &tensor.shape, axis),
            };
            match result {
                Ok(t) => Payload::from_tensor(t),
                Err(e) => Payload::Error(e.into()),
            }
        }
        _ => Payload::Error(core_types::RString::from(
            "Action \'log_softmax\' requires a tensor or scalar value",
        )),
    }
}

/// F32 is computed in single precision, everything else in F64. A NaN in a
/// lane makes the whole lane NaN, like NumPy.
fn compute_log_softmax<T: FloatElement>(
    vals: Vec<T>,
    shape: &[usize],
    ax: usize,
) -> Result<Tensor, String> {
    let shape = shape.to_vec();
    if shape.is_empty() {
        let x = vals[0];
        let y = if x.is_nan() { x } else { T::zero() };
        return Tensor::from_vec_of(vec![y], shape).map_err(|e| e.to_string());
    }

    let axis_len = shape[ax];
    if axis_len == 0 || vals.is_empty() {
        return Tensor::from_vec_of(vals, shape).map_err(|e| e.to_string());
    }

    let out_vals = numeric::map_axis(&vals, &shape, ax, |lane: &[T], out: &mut [T]| {
        let mut max_val = T::neg_infinity();
        for &v in lane {
            max_val = max_val.nan_max(v);
        }
        let mut sum_exp = T::zero();
        for &v in lane {
            sum_exp = sum_exp + (v - max_val).exp();
        }

        let log_sum_exp = max_val + sum_exp.ln();
        for (a, &v) in lane.iter().enumerate() {
            out[a] = v - log_sum_exp;
        }
    });

    Tensor::from_vec_of(out_vals, shape).map_err(|e| e.to_string())
}

#[no_mangle]
pub extern "C" fn shapecheck(
    input: core_types::InputDescriptor,
    args: core_types::ActionArgs,
) -> core_types::ShapeCheckResult {
    let rank = input.value.shape().map(Shape::rank);
    let result = core_types::shapecheck::analyze(
        input,
        args,
        env!("CARGO_PKG_NAME"),
        get_input_type(),
        get_output_type(),
        Some(get_output_shape),
        None,
    );
    core_types::shapecheck::axis_plan(result, rank, 0, Some(-1), true)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn process(payload: Payload) -> Payload {
        core_types::shapecheck::execute(env!("CARGO_PKG_NAME"), shapecheck, super::process, payload)
    }
    use core_types::Tensor;

    #[test]
    fn test_log_softmax_action() {
        let tensor = Tensor::from_f32_slice(&[1.0, 2.0, 3.0]);
        let res = process(Payload::Tensor(tensor));
        if let Payload::Tensor(out) = res {
            let slice = out.as_f32_slice().unwrap();
            let exp_sum: f32 = slice.iter().map(|&x| x.exp()).sum();
            assert!((exp_sum - 1.0).abs() < 1e-4);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
