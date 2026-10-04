# rust-neural-network-cpu

Hand-optimized CPU inference kernels for basic neural network layers, written
in Rust from scratch and benchmarked against PyTorch's CPU backend at each
stage of optimization (naive -> cache/layout -> SIMD -> multithreaded).

First stage of a larger project on optimizing CPU inference for the Mamba
(selective state space model) architecture; a separate repo will build on the
techniques developed here.

## Structure
- `src/` - kernel implementations, one module per layer type
- `benches/` - Criterion benchmarks, one per layer type
- `python/` - PyTorch CPU baselines and fixture generation for correctness validation
- `testdata/` - generated fixtures (inputs + expected outputs), not committed

## Layer order
Activation -> Linear/GEMM -> Convolution. GEMM gets the most attention since
it is reused by im2col-based convolution and, later, Mamba's chunked scan.

## Status
Environment set up (WSL2 Ubuntu, Rust stable, no AVX-512 on this CPU - AVX2/FMA/AVX-VNNI only).
Kernel work starting with activation functions.
