use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use nn_kernels::activation::{silu_naive, silu_simd, silu_simd_rayon, silu_simd_rayon_blocked};
use rayon::ThreadPoolBuilder;

const SIZE: usize = 1 << 20;
const LARGE: usize = 1 << 24;
const DEFAULT_BLOCK: usize = 64 * 1024;

fn make_input(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 % 1000.0) / 100.0 - 5.0).collect()
}

fn bench_activation(c: &mut Criterion) {
    let input = make_input(SIZE);

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

    c.bench_function("silu_simd_rayon", |b| {
        b.iter_batched(
            || input.clone(),
            |mut data| {
                silu_simd_rayon(&mut data);
                black_box(&data);
            },
            BatchSize::LargeInput,
        )
    });
}

fn bench_rayon_experiments(c: &mut Criterion) {
    let small = make_input(SIZE);
    let large = make_input(LARGE);

    let mut g = c.benchmark_group("block_size_1M");
    for &block in &[8 * 1024usize, 64 * 1024, 256 * 1024] {
        g.bench_with_input(BenchmarkId::from_parameter(block), &block, |b, &block| {
            b.iter_batched(
                || small.clone(),
                |mut data| {
                    silu_simd_rayon_blocked(&mut data, block);
                    black_box(&data);
                },
                BatchSize::LargeInput,
            )
        });
    }
    g.finish();

    let mut g = c.benchmark_group("input_size");
    g.sample_size(20);
    g.bench_function("serial_1M", |b| {
        b.iter_batched(
            || small.clone(),
            |mut data| {
                silu_simd(&mut data);
                black_box(&data);
            },
            BatchSize::LargeInput,
        )
    });
    g.bench_function("rayon_1M", |b| {
        b.iter_batched(
            || small.clone(),
            |mut data| {
                silu_simd_rayon_blocked(&mut data, DEFAULT_BLOCK);
                black_box(&data);
            },
            BatchSize::LargeInput,
        )
    });
    g.bench_function("serial_16M", |b| {
        b.iter_batched(
            || large.clone(),
            |mut data| {
                silu_simd(&mut data);
                black_box(&data);
            },
            BatchSize::LargeInput,
        )
    });
    g.bench_function("rayon_16M", |b| {
        b.iter_batched(
            || large.clone(),
            |mut data| {
                silu_simd_rayon_blocked(&mut data, DEFAULT_BLOCK);
                black_box(&data);
            },
            BatchSize::LargeInput,
        )
    });
    g.finish();

    let mut g = c.benchmark_group("thread_sweep_16M");
    g.sample_size(20);
    for &threads in &[1usize, 2, 4, 8, 11, 22] {
        let pool = ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        g.bench_with_input(BenchmarkId::from_parameter(threads), &threads, |b, _| {
            b.iter_batched(
                || large.clone(),
                |mut data| {
                    pool.install(|| silu_simd_rayon_blocked(&mut data, DEFAULT_BLOCK));
                    black_box(&data);
                },
                BatchSize::LargeInput,
            )
        });
    }
    g.finish();
}

fn bench_cap_4_threads(c: &mut Criterion) {
    let large = make_input(LARGE);
    let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();

    let mut g = c.benchmark_group("cap4_16M");
    g.sample_size(20);
    g.bench_function("serial", |b| {
        b.iter_batched(
            || large.clone(),
            |mut data| {
                silu_simd(&mut data);
                black_box(&data);
            },
            BatchSize::LargeInput,
        )
    });
    for &block in &[64 * 1024usize, 256 * 1024] {
        g.bench_with_input(BenchmarkId::new("rayon_block", block), &block, |b, &block| {
            b.iter_batched(
                || large.clone(),
                |mut data| {
                    pool.install(|| silu_simd_rayon_blocked(&mut data, block));
                    black_box(&data);
                },
                BatchSize::LargeInput,
            )
        });
    }
    g.finish();
}

criterion_group!(benches, bench_activation, bench_rayon_experiments, bench_cap_4_threads);
criterion_main!(benches);
