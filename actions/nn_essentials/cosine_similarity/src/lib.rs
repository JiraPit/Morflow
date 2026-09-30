use core_types::{DataType, Payload, Tensor};
use rayon::prelude::*;

#[no_mangle]
pub extern "C" fn get_input_type() -> DataType {
    DataType::Composite
}

#[no_mangle]
pub extern "C" fn get_output_type() -> DataType {
    DataType::Tensor
}

#[no_mangle]
pub extern "C" fn process(payload: Payload) -> Payload {
    let (inner_payload, args_opt) = payload.take_payload_and_args();

    let mut eps = 1e-8f32;
    if let Some(args) = &args_opt {
        if let Some(e_str) = args.get_named("eps") {
            if let Ok(e) = e_str.parse::<f32>() {
                eps = e;
            }
        }
    }

    match inner_payload {
        Payload::Composite(items) if items.len() >= 2 => {
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
    if t1.shape != t2.shape {
        return Err(format!(
            "Shape mismatch in cosine_similarity: {:?} vs {:?}",
            t1.shape.as_slice(),
            t2.shape.as_slice()
        ));
    }

    let vals1 = t1.to_vec_f32();
    let vals2 = t2.to_vec_f32();

    let r = t1.rank();
    if r == 0 {
        return Ok(Tensor::from_f32_slice(&[1.0]));
    }

    let feat_dim = *t1.shape.last().unwrap();
    if feat_dim == 0 {
        return Ok(Tensor::from_f32_slice(&[1.0]));
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

#[cfg(test)]
mod tests {
    use super::*;
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
