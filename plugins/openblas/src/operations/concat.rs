use core_types::Payload;
pub fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    if let Err(error) = crate::require() {
        return Payload::Error(error);
    }
    let (inner_payload, args_opt) = (payload.into_unwrapped(), Some(&prepared.args));

    let mut axis = 0isize;
    if let Some(args) = &args_opt {
        if let Some(ax_str) = args
            .get_named("axis")
            .or_else(|| args.get_named("dim"))
            .or_else(|| args.positional.first().map(|s| s.as_str()))
        {
            if let Ok(ax) = prepared.args.parse::<isize>(ax_str) {
                axis = ax;
            }
        }
    }

    match inner_payload {
        Payload::Composite(items) => {
            let mut tensors = Vec::new();
            for item in items {
                match item {
                    Payload::Tensor(t) => tensors.push(t),
                    _ => {
                        return Payload::Error(
                            "All items in composite payload must be tensors for concat".into(),
                        )
                    }
                }
            }
            match crate::concat(&tensors, axis) {
                Ok(out) => Payload::Tensor(out),
                Err(e) => Payload::Error(e),
            }
        }
        Payload::Tensor(t) => Payload::Tensor(t),
        _ => Payload::Error(core_types::RString::from(
            "Action \'concat\' requires Payload::Composite or Payload::Tensor",
        )),
    }
}
