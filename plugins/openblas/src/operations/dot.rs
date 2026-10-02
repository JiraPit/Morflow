use core_types::Payload;
use core_types::Tensor;
pub fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    if let Err(error) = crate::require() {
        return Payload::Error(error);
    }
    let (inner_payload, _) = (payload.into_unwrapped(), Some(&prepared.args));

    match inner_payload {
        Payload::Composite(items) if items.len() == 2 => {
            let (t1, t2) = match (&items[0], &items[1]) {
                (Payload::Tensor(a), Payload::Tensor(b)) => (a, b),
                _ => return Payload::Error("dot expects 2 tensor inputs in composite".into()),
            };
            match compute_dot(t1, t2) {
                Ok(out) => Payload::Tensor(out),
                Err(e) => Payload::Error(e.into()),
            }
        }
        Payload::Tensor(t) => match compute_dot(&t, &t) {
            Ok(out) => Payload::Tensor(out),
            Err(e) => Payload::Error(e.into()),
        },
        other => other,
    }
}

fn compute_dot(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    crate::dot(a, b)
}
