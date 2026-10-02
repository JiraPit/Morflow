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
                _ => return Payload::Error("matmul expects 2 tensor inputs in composite".into()),
            };
            match compute_matmul(t1, t2, &prepared) {
                Ok(out) => Payload::Tensor(out),
                Err(e) => Payload::Error(e.into()),
            }
        }
        Payload::Tensor(t) => {
            // Self-multiplication A * A
            match compute_matmul(&t, &t, &prepared) {
                Ok(out) => Payload::Tensor(out),
                Err(e) => Payload::Error(e.into()),
            }
        }
        other => other,
    }
}

fn compute_matmul(
    a: &Tensor,
    b: &Tensor,
    prepared: &core_types::PreparedData,
) -> Result<Tensor, String> {
    crate::matmul(a, b, prepared)
}
