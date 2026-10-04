use criterion::{criterion_group, criterion_main, Criterion};

fn bench_linear(_c: &mut Criterion) {
    // TODO: benchmark linear/GEMM kernels here
}

criterion_group!(benches, bench_linear);
criterion_main!(benches);
