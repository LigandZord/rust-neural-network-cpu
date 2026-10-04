"""Shared helpers for cross-language validation: save random inputs and
PyTorch's expected outputs so Rust-side tests can load and compare against them."""
import numpy as np

def save_fixture(name: str, **arrays):
    for key, arr in arrays.items():
        np.save(f"../testdata/{name}_{key}.npy", np.asarray(arr, dtype=np.float32))
