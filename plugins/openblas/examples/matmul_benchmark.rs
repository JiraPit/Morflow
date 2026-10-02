//! cargo run --release -p morflow-plugin-openblas --example matmul_benchmark
use core_types::{ActionArgs, InputDescriptor, Payload, RVec, ShapeCheckResult, Tensor};
use std::{hint::black_box, time::Instant};
fn measure(mut run: impl FnMut(), iterations: usize) -> f64 {
    for _ in 0..2 {
        run();
    }
    let begin = Instant::now();
    for _ in 0..iterations {
        run();
    }
    begin.elapsed().as_secs_f64() * 1000. / iterations as f64
}
fn main() {
    for side in [64, 256, 512] {
        let a = Tensor::from_f32_vec(vec![0.25; side * side], vec![side, side]).unwrap();
        let b = Tensor::from_f32_vec(vec![0.5; side * side], vec![side, side]).unwrap();
        let payload = Payload::Composite(RVec::from(vec![
            Payload::Tensor(a.clone()),
            Payload::Tensor(b.clone()),
        ]));
        let ShapeCheckResult::Ready { prepared, .. } = basic_matmul::shapecheck(
            InputDescriptor::from_payload(&payload),
            ActionArgs::default(),
        ) else {
            panic!("invalid inputs")
        };
        let basic = measure(
            || {
                black_box(basic_matmul::process(payload.clone(), prepared.clone()));
            },
            10,
        );
        let blas = measure(
            || {
                black_box(
                    morflow_openblas_plugin::matmul(black_box(&a), black_box(&b), &prepared)
                        .unwrap(),
                );
            },
            10,
        );
        println!(
            "matmul {side}x{side}: basics={basic:.4}ms blas={blas:.4}ms speedup={:.2}x",
            basic / blas
        );
    }
}
