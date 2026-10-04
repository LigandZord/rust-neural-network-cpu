"""Benchmark PyTorch's CPU SiLU as the baseline for the naive Rust kernel.

Matches benches/activation_bench.rs: same element count, in-place op, and
setup (tensor clone) excluded from the timed region.
"""
import time
import torch
import torch.nn.functional as F

SIZE = 1 << 20
WARMUP = 10
ITERS = 100


def bench(label: str) -> None:
    x = torch.empty(SIZE, dtype=torch.float32).uniform_(-5, 5)

    for _ in range(WARMUP):
        F.silu(x.clone(), inplace=True)

    clones = [x.clone() for _ in range(ITERS)]
    start = time.perf_counter()
    for t in clones:
        F.silu(t, inplace=True)
    elapsed = time.perf_counter() - start

    per_iter = elapsed / ITERS
    print(
        f"{label:24s} {per_iter * 1e6:9.2f} us/iter  "
        f"{SIZE / per_iter / 1e9:6.3f} G-elem/s"
    )


if __name__ == "__main__":
    print(f"SiLU on {SIZE:,} f32 elements, {ITERS} iters (after {WARMUP} warmup)\n")

    default_threads = torch.get_num_threads()

    torch.set_num_threads(1)
    bench("1 thread (vs naive)")

    torch.set_num_threads(default_threads)
    bench(f"{default_threads} threads (default)")
