"""Generate SiLU input/expected-output fixtures from PyTorch's CPU kernel,
for cross-language validation against the Rust implementation."""
import numpy as np
import torch
import torch.nn.functional as F

from gen_fixtures import save_fixture

np.random.seed(0)

N = 10_000
# Wide range on purpose: covers typical pre-activation magnitudes and also
# samples well past the +-87 clamp boundary used in the AVX2 exp kernel.
x = np.random.uniform(-100.0, 100.0, size=N).astype(np.float32)

y = F.silu(torch.from_numpy(x)).numpy()

save_fixture("silu", input=x, expected=y)
print(f"Wrote {N} SiLU fixtures to ../testdata/silu_input.npy and ../testdata/silu_expected.npy")
