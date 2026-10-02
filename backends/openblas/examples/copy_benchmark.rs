//! cargo run --release -p morflow-openblas --example copy_benchmark
use core_types::Tensor;
use std::{hint::black_box, time::Instant};
fn measure(mut run: impl FnMut(), iterations: usize) -> f64 {
    for _ in 0..3 {
        run();
    }
    let begin = Instant::now();
    for _ in 0..iterations {
        run();
    }
    begin.elapsed().as_secs_f64() * 1000. / iterations as f64
}
fn main() {
    let blas = morflow_openblas::require().unwrap();
    for side in [64, 512, 2048] {
        let input = Tensor::from_f32_vec(vec![1.; side * side], vec![side, side]).unwrap();
        let basic_repeat = measure(
            || {
                let rows = Tensor::concat(&[input.clone(), input.clone()], 0).unwrap();
                black_box(Tensor::concat(&[rows.clone(), rows], 1).unwrap());
            },
            20,
        );
        let blas_repeat = measure(
            || {
                black_box(morflow_openblas::repeat(black_box(&input), &[2, 2]).unwrap());
            },
            20,
        );
        println!("repeat {side}x{side}: basics={basic_repeat:.4}ms blas={blas_repeat:.4}ms speedup={:.2}x",basic_repeat/blas_repeat);
        for axis in [0, 1] {
            let cut = side / 2;
            let parts = [
                input.slice_range(axis as usize, cut, side, 1).unwrap(),
                input.slice_range(axis as usize, 0, cut, 1).unwrap(),
            ];
            let basic_roll = measure(
                || {
                    black_box(Tensor::concat(black_box(&parts), axis).unwrap());
                },
                20,
            );
            let blas_roll = measure(
                || {
                    black_box(blas.concat(black_box(&parts), axis).unwrap());
                },
                20,
            );
            println!("roll {side}x{side} axis={axis}: basics={basic_roll:.4}ms blas={blas_roll:.4}ms speedup={:.2}x",basic_roll/blas_roll);
            let inputs = vec![input.clone(); 4];
            let basic = measure(
                || {
                    black_box(Tensor::concat(black_box(&inputs), axis).unwrap());
                },
                20,
            );
            let accelerated = measure(
                || {
                    black_box(blas.concat(black_box(&inputs), axis).unwrap());
                },
                20,
            );
            println!("concat {side}x{side} axis={axis}: basics={basic:.4}ms blas={accelerated:.4}ms speedup={:.2}x",basic/accelerated);
        }
    }
}
