// Activation kernels. Progression: naive -> cache/layout -> SIMD (AVX2+FMA) -> threaded (rayon).
use std::arch::x86_64::*;
use rayon::prelude::*;

pub fn silu_naive(input: &mut [f32]) {
    for v in input.iter_mut() {
        *v = *v/ (1.0 + (-*v).exp());
    }
}

pub fn silu_simd(input: &mut [f32]) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe {
            silu_avx2(input)
        }
    } else {
        silu_naive(input)
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn silu_avx2(input :&mut [f32]){
    let chunks = input.len() / 8;
    for i in 0..chunks {
        //load certain chunks into the mem
        let c = _mm256_loadu_ps(input.as_ptr().add(i * 8)); //only loads 8 f32 instances
        _mm256_storeu_ps(input.as_mut_ptr().add(i * 8), silu_calc(c));
    }

    //remaining ones, that could not be vectorized
    for i in (chunks*8)..input.len() {
        input[i] = input[i]/ (1.0 + (-input[i]).exp());
    }

}

#[target_feature(enable = "avx2,fma")]
unsafe fn exp_avx2_f32(x: __m256) -> __m256 {
    let log2_e = _mm256_set1_ps(std::f32::consts::LOG2_E);
    // clamp x
    let hi =  _mm256_set1_ps(87.0);
    let lo = _mm256_set1_ps(-87.0);
    let x = _mm256_max_ps(_mm256_min_ps(x, hi), lo);

    let t = _mm256_mul_ps(x, log2_e);
    let n = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    let r = _mm256_sub_ps(t, n);

    // 2^n
    let n_int: __m256i = _mm256_cvtps_epi32(n);
    let exponent_bias = _mm256_set1_epi32(127); //f32 specific
    let twopow_n = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n_int, exponent_bias), 23)); // 2^n

    // 2^r
    let c0 = _mm256_set1_ps(1.0);
    let c1 = _mm256_set1_ps(0.69314718);  // ln(2)
    let c2 = _mm256_set1_ps(0.24022650);  // ln(2)^2 / 2
    let c3 = _mm256_set1_ps(0.05550411);  // ln(2)^3 / 6
    let c4 = _mm256_set1_ps(0.00961812);  // ln(2)^4 / 24
    let c5 = _mm256_set1_ps(0.00133336); // ln(2)^5 / 120
    // using Horner method for calculating the taylor expansion

    let mut p = c5;
    p = _mm256_fmadd_ps(p, r, c4);
    p = _mm256_fmadd_ps(p, r, c3);
    p = _mm256_fmadd_ps(p, r, c2);
    p = _mm256_fmadd_ps(p, r, c1);
    p = _mm256_fmadd_ps(p, r, c0);

    _mm256_mul_ps(p, twopow_n) //return the approximate answer for e^x

}

#[target_feature(enable = "avx2,fma")]
unsafe fn silu_calc(x : __m256) -> __m256 {
    let minus_one_vec = _mm256_set1_ps(-1.0);
    _mm256_div_ps(x,_mm256_add_ps(_mm256_set1_ps(1.0),exp_avx2_f32(_mm256_mul_ps(minus_one_vec,x))))

}

const BLOCK: usize = 64 * 1024;

pub fn silu_simd_rayon(input: &mut [f32]) {
    silu_simd_rayon_blocked(input, BLOCK)
}

pub fn silu_simd_rayon_blocked(input: &mut [f32], block: usize) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        input.par_chunks_mut(block)
        .for_each(|x| unsafe {
            silu_avx2(x)
        })
    } else {
        silu_naive(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Broadcasts x into all 8 lanes, runs exp_avx2_f32, and reads back one lane.
    fn exp_avx2_scalar(x: f32) -> f32 {
        unsafe {
            let v = _mm256_set1_ps(x);
            let result = exp_avx2_f32(v);
            let mut out = [0.0f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), result);
            out[0]
        }
    }

    #[test]
    fn exp_avx2_matches_std_exp() {
        // Stay well inside the +-87 clamp so we're testing the polynomial's
        // own accuracy, not the intentional saturation behavior at the edges.
        let mut max_rel_err = 0.0f32;
        let mut worst_x = 0.0f32;
        let mut x = -80.0f32;
        while x <= 80.0 {
            let expected = x.exp();
            let actual = exp_avx2_scalar(x);
            let rel_err = ((actual - expected) / expected).abs();
            if rel_err > max_rel_err {
                max_rel_err = rel_err;
                worst_x = x;
            }
            x += 0.01;
        }
        println!("exp_avx2_f32: max relative error = {:e} at x = {}", max_rel_err, worst_x);
        assert!(
            max_rel_err < 1e-4,
            "exp_avx2_f32 relative error too high: {:e} at x = {}",
            max_rel_err,
            worst_x
        );
    }

    #[test]
    fn silu_simd_matches_silu_naive() {
        let input: Vec<f32> = (0..1001).map(|i| (i as f32 - 500.0) * 0.1).collect();

        let mut naive = input.clone();
        silu_naive(&mut naive);

        let mut simd = input.clone();
        silu_simd(&mut simd);

        let mut max_abs_err = 0.0f32;
        let mut worst_x = 0.0f32;
        for ((x, a), b) in input.iter().zip(naive.iter()).zip(simd.iter()) {
            let err = (a - b).abs();
            if err > max_abs_err {
                max_abs_err = err;
                worst_x = *x;
            }
        }
        println!("silu_simd vs silu_naive: max abs error = {:e} at x = {}", max_abs_err, worst_x);
        assert!(
            max_abs_err < 1e-3,
            "silu_simd diverges from silu_naive by {:e} at x = {}",
            max_abs_err,
            worst_x
        );
    }

    fn load_npy_f32(path: &str) -> Vec<f32> {
        let bytes = std::fs::read(path)
            .unwrap_or_else(|e| panic!("failed to read {path} ({e}) - run `python gen_activation_fixtures.py` from python/ first"));
        npyz::NpyFile::new(&bytes[..])
            .unwrap_or_else(|e| panic!("failed to parse {path}: {e}"))
            .into_vec::<f32>()
            .unwrap_or_else(|e| panic!("failed to read f32 data from {path}: {e}"))
    }

    // allclose-style tolerance: |actual - expected| <= atol + rtol * |expected|
    fn within_tolerance(actual: f32, expected: f32, atol: f32, rtol: f32) -> bool {
        (actual - expected).abs() <= atol + rtol * expected.abs()
    }

    fn assert_matches_pytorch(kernel: fn(&mut [f32]), name: &str) {
        let input = load_npy_f32("testdata/silu_input.npy");
        let expected = load_npy_f32("testdata/silu_expected.npy");
        assert_eq!(input.len(), expected.len());

        let mut actual = input.clone();
        kernel(&mut actual);

        let atol = 1e-4;
        let rtol = 1e-3;
        let mut max_abs_err = 0.0f32;
        let mut worst_x = 0.0f32;
        let mut num_failures = 0;
        for ((x, a), e) in input.iter().zip(actual.iter()).zip(expected.iter()) {
            let err = (a - e).abs();
            if err > max_abs_err {
                max_abs_err = err;
                worst_x = *x;
            }
            if !within_tolerance(*a, *e, atol, rtol) {
                num_failures += 1;
            }
        }
        println!(
            "{name} vs PyTorch ({} fixtures): max abs error = {:e} at x = {}, {} outside tolerance",
            input.len(),
            max_abs_err,
            worst_x,
            num_failures
        );
        assert_eq!(
            num_failures, 0,
            "{num_failures} of {} {name} outputs fell outside atol={atol}, rtol={rtol} of PyTorch",
            input.len()
        );
    }

    #[test]
    fn silu_simd_matches_pytorch_fixtures() {
        assert_matches_pytorch(silu_simd, "silu_simd");
    }

    #[test]
    fn silu_simd_rayon_matches_pytorch_fixtures() {
        assert_matches_pytorch(silu_simd_rayon, "silu_simd_rayon");
    }
}