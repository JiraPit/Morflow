use core_types::Payload;
use core_types::RVec;
use core_types::Tensor;
pub fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    if let Err(error) = crate::require() {
        return Payload::Error(error);
    }
    let (inner_payload, _) = (payload.into_unwrapped(), Some(&prepared.args));

    match inner_payload {
        Payload::Tensor(tensor) => match compute_qr(&tensor) {
            Ok((q, r)) => {
                let mut items = RVec::new();
                items.push(Payload::Tensor(q));
                items.push(Payload::Tensor(r));
                Payload::Composite(items)
            }
            Err(e) => Payload::Error(e.into()),
        },
        other => other,
    }
}

fn compute_qr(a: &Tensor) -> Result<(Tensor, Tensor), String> {
    crate::qr(a)
}
