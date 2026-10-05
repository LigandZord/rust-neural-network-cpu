use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use std::time::{Duration, Instant};
use nn_kernels::activation::{silu_naive, silu_simd, silu_simd_rayon, silu_simd_rayon_blocked};
use rayon::ThreadPoolBuilder;

const SMALL: usize = 1 << 20;
const LARGE: usize = 1 << 24;

fn make_input(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 % 1000.0) / 100.0 - 5.0).collect()
}

fn bench_matrix(c: &mut Criterion) {
    let small = make_input(SMALL);
    let large = make_input(LARGE);
    let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();

    c.bench_function("naive_1M", |b| {
        b.iter_batched(
            || small.clone(),
            |mut data| {
                silu_naive(&mut data);
                black_box(&data);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("naive_16M", |b| {
        b.iter_batched(
            || large.clone(),
            |mut data| {
                silu_naive(&mut data);
                black_box(&data);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("simd_1M", |b| {
        b.iter_batched(
            || small.clone(),
            |mut data| {
                silu_simd(&mut data);
                black_box(&data);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("simd_16M", |b| {
        b.iter_batched(
            || large.clone(),
            |mut data| {
                silu_simd(&mut data);
                black_box(&data);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("rayon4_1M", |b| {
        b.iter_batched(
            || small.clone(),
            |mut data| {
                pool.install(|| silu_simd_rayon(&mut data));
                black_box(&data);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("rayon4_16M", |b| {
        b.iter_batched(
            || large.clone(),
            |mut data| {
                pool.install(|| silu_simd_rayon(&mut data));
                black_box(&data);
            },
            BatchSize::LargeInput,
        )
    });
}

fn warm_loop<F: Fn(&mut [f32])>(c: &mut Criterion, name: &str, src: &[f32], batch: usize, kernel: F) {
    c.bench_function(name, |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            let mut done = 0u64;
            while done < iters {
                let n = batch.min((iters - done) as usize);
                let mut bufs: Vec<Vec<f32>> = (0..n).map(|_| src.to_vec()).collect();
                let start = Instant::now();
                for buf in bufs.iter_mut() {
                    kernel(buf.as_mut_slice());
                }
                total += start.elapsed();
                black_box(&bufs);
                done += n as u64;
            }
            total
        })
    });
}

fn bench_warm(c: &mut Criterion) {
    let small = make_input(SMALL);
    let large = make_input(LARGE);
    let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();

    warm_loop(c, "warm_serial_1M", &small, 100, |x: &mut [f32]| silu_simd(x));
    warm_loop(c, "warm_serial_16M", &large, 20, |x: &mut [f32]| silu_simd(x));

    warm_loop(c, "warm_rayon64K_1M", &small, 100, |x: &mut [f32]| {
        pool.install(|| silu_simd_rayon_blocked(x, 64 * 1024))
    });
    warm_loop(c, "warm_rayon64K_16M", &large, 20, |x: &mut [f32]| {
        pool.install(|| silu_simd_rayon_blocked(x, 64 * 1024))
    });

    warm_loop(c, "warm_split4_1M", &small, 100, |x: &mut [f32]| {
        let block = x.len() / 4;
        pool.install(|| silu_simd_rayon_blocked(x, block))
    });
    warm_loop(c, "warm_split4_16M", &large, 20, |x: &mut [f32]| {
        let block = x.len() / 4;
        pool.install(|| silu_simd_rayon_blocked(x, block))
    });
}

criterion_group!(benches, bench_matrix, bench_warm);
criterion_main!(benches);
