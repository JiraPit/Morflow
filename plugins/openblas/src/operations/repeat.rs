use core_types::Payload;
use core_types::Tensor;
pub fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    if let Err(error) = crate::require() {
        return Payload::Error(error);
    }
    let (inner_payload, args_opt) = (payload.into_unwrapped(), Some(&prepared.args));

    let mut repeats_str = None;
    if let Some(args) = &args_opt {
        repeats_str = args
            .get_named("repeats")
            .or_else(|| args.positional.first().map(|s| s.as_str()));
    }

    let Some(r_str) = repeats_str else {
        return Payload::Error("repeat action requires 'repeats' argument".into());
    };

    let repeats = match prepared.args.usize_list(r_str) {
        Ok(r) => r,
        Err(e) => return Payload::Error(e),
    };

    match inner_payload {
        Payload::Tensor(tensor) | Payload::Scalar(tensor) => match repeat_tensor(&tensor, &repeats)
        {
            Ok(t) => Payload::from_tensor(t),
            Err(e) => Payload::Error(e.into()),
        },
        _ => Payload::Error(core_types::RString::from(
            "Action \'repeat\' requires a tensor or scalar value",
        )),
    }
}

fn repeat_tensor(tensor: &Tensor, repeats: &[usize]) -> Result<Tensor, String> {
    crate::repeat(tensor, repeats).map_err(|e| e.to_string())
}
