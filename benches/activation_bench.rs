use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use nn_kernels::activation::{silu_naive, silu_simd};

const SIZE: usize = 1 << 20;

fn bench_activation(c: &mut Criterion) {
    let input: Vec<f32> = (0..SIZE)
        .map(|i| (i as f32 % 1000.0) / 100.0 - 5.0)
        .collect();

    c.bench_function("silu_naive", |b| {
        b.iter_batched(
            || input.clone(),
            |mut data| {
                silu_naive(&mut data);
                black_box(&data);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("silu_simd", |b| {
        b.iter_batched(
            || input.clone(),
            |mut data| {
                silu_simd(&mut data);
                black_box(&data);
            },
            BatchSize::LargeInput,
        )
    });
}

criterion_group!(benches, bench_activation);
criterion_main!(benches);
