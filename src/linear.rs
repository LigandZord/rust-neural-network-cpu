// Linear/GEMM kernels. Progression: naive -> cache/layout -> SIMD (AVX2+FMA) -> threaded (rayon).
// This is the highest-leverage kernel in the repo: convolution (im2col) and, later,
// Mamba's chunked selective scan both reuse whatever GEMM is built here.
