use core_types::shapecheck::PreparedArgs;
use core_types::{DataType, Payload, Tensor};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Composite | DataType::Tensor
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
}

pub fn get_output_value_shape<A: Into<PreparedArgs>>(
    input: core_types::ValueShape,
    args: A,
) -> core_types::ValueShapeResult {
    let args = args.into();
    core_types::composite_contract::cosine_similarity(input, args)
}

#[no_mangle]
pub extern "C" fn process(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    process_impl(payload, prepared)
}

fn process_impl(payload: Payload, prepared: core_types::PreparedData) -> Payload {
    let (inner_payload, args_opt) = (payload.into_unwrapped(), Some(&prepared.args));

    let mut eps = 1e-8f32;
    if let Some(args) = &args_opt {
        if let Some(e_str) = args.get_named("eps") {
            if let Ok(e) = prepared.args.parse::<f32>(e_str) {
                eps = e;
            }
        }
    }

    match inner_payload {
        Payload::Composite(items) if items.len() == 2 => {
            let (t1, t2) = match (&items[0], &items[1]) {
                (Payload::Tensor(a), Payload::Tensor(b)) => (a, b),
                _ => {
                    return Payload::Error(
                        "cosine_similarity expects 2 tensor inputs in composite".into(),
                    )
                }
            };
            match compute_cosine_similarity(t1, t2, eps) {
                Ok(out) => Payload::Tensor(out),
                Err(e) => Payload::Error(e.into()),
            }
        }
        Payload::Tensor(t) => Payload::Tensor(t),
        _ => Payload::Error(core_types::RString::from(
            "Action \'cosine_similarity\' requires Payload::Tensor",
        )),
    }
}

fn compute_cosine_similarity(t1: &Tensor, t2: &Tensor, eps: f32) -> Result<Tensor, String> {
    let vals1 = t1.to_vec_f32();
    let vals2 = t2.to_vec_f32();

    let r = t1.rank();
    if r == 0 {
        return Ok(Tensor::from_f32_slice(&[1.0]));
    }

    let feat_dim = *t1.shape.last().unwrap();
    if feat_dim == 0 {
        let shape = if r > 1 {
            t1.shape[..r - 1].to_vec()
        } else {
            vec![1]
        };
        return Tensor::from_f32_vec(vec![1.0; shape.iter().product()], shape)
            .map_err(|e| e.to_string());
    }

    let num_vectors = vals1.len() / feat_dim;
    let mut out_sims = vec![0.0f32; num_vectors];

    out_sims.par_iter_mut().enumerate().for_each(|(i, out)| {
        let offset = i * feat_dim;
        let v1 = &vals1[offset..offset + feat_dim];
        let v2 = &vals2[offset..offset + feat_dim];

        let mut dot = 0.0f32;
        let mut norm1_sq = 0.0f32;
        let mut norm2_sq = 0.0f32;

        for k in 0..feat_dim {
            dot += v1[k] * v2[k];
            norm1_sq += v1[k] * v1[k];
            norm2_sq += v2[k] * v2[k];
        }

        let denom = (norm1_sq.sqrt() * norm2_sq.sqrt()).max(eps);
        *out = dot / denom;
    });

    let out_shape = if r > 1 {
        t1.shape[0..r - 1].to_vec()
    } else {
        vec![1]
    };

    Tensor::from_f32_vec(out_sims, out_shape).map_err(|e| e.to_string())
}

#[no_mangle]
pub extern "C" fn get_action_abi_version() -> u32 {
    core_types::shapecheck::ACTION_ABI_VERSION
}
#[no_mangle]
pub extern "C" fn get_action_abi_layout() -> *const core_types::abi_stable::type_layout::TypeLayout
{
    <core_types::shapecheck::ActionAbiLayout as core_types::StableAbi>::LAYOUT
}
#[no_mangle]
pub extern "C" fn shapecheck(
    input: core_types::InputDescriptor,
    args: core_types::ActionArgs,
) -> core_types::ShapeCheckResult {
    core_types::shapecheck::analyze(
        input,
        args,
        env!("CARGO_PKG_NAME"),
        get_input_type(),
        get_output_type(),
        None,
        Some(get_output_value_shape),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn process(payload: Payload) -> Payload {
        core_types::shapecheck::execute(env!("CARGO_PKG_NAME"), shapecheck, super::process, payload)
    }
    use core_types::{RVec, Tensor};

    #[test]
    fn test_cosine_similarity_action() {
        let t1 = Tensor::from_f32_slice(&[1.0, 0.0]);
        let t2 = Tensor::from_f32_slice(&[1.0, 0.0]);
        let t3 = Tensor::from_f32_slice(&[0.0, 1.0]);

        let mut comp_ident = RVec::new();
        comp_ident.push(Payload::Tensor(t1.clone()));
        comp_ident.push(Payload::Tensor(t2));

        let res_ident = process(Payload::Composite(comp_ident));
        if let Payload::Tensor(out) = res_ident {
            assert!((out.as_f32_slice().unwrap()[0] - 1.0).abs() < 1e-4);
        } else {
            panic!("Expected Tensor output");
        }

        let mut comp_ortho = RVec::new();
        comp_ortho.push(Payload::Tensor(t1));
        comp_ortho.push(Payload::Tensor(t3));

        let res_ortho = process(Payload::Composite(comp_ortho));
        if let Payload::Tensor(out) = res_ortho {
            assert!(out.as_f32_slice().unwrap()[0].abs() < 1e-4);
        } else {
            panic!("Expected Tensor output");
        }
    }
}
