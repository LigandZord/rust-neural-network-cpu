"""Benchmark PyTorch's CPU SiLU across thread counts.

Matches benches/activation_bench.rs: same element count, in-place op, and
setup (tensor clone) excluded from the timed region.
"""
import os
import time
import torch
import torch.nn.functional as F

SIZE = int(os.environ.get("SIZE", 1 << 20))
WARMUP = 10
ITERS = 100 if SIZE <= (1 << 20) else 20
THREADS = [int(os.environ["THREADS"])] if "THREADS" in os.environ else [1, 2, 4, 8, 11, 22]


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

    for n in THREADS:
        torch.set_num_threads(n)
        bench(f"{n} threads")
