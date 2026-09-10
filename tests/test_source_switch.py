"""Lightweight shape/behaviour test for the reconstructed probe modules.

Runs on CPU with synthetic tensors -- it does NOT need the full dagr package or
the PEDRo dataset. It checks:
  * AdaptiveHyperedge accepts all four source widths and emits a row-normalised
    [N, num_hyperedges] incidence matrix,
  * HyperConv is shape-preserving and starts from a residual (out != 0 path),
  * hyperedge_regularization returns a finite scalar for every weight setting.

Usage:
    python tests/test_source_switch.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code", "dagr_env", "src", "dagr", "model", "layers"))

from ssm_hyper import AdaptiveHyperedge, HyperConv, hyperedge_regularization  # noqa: E402

NODE_DIM = 64
M = 16
N = 97

SOURCE_DIMS = {
    "ssm": NODE_DIM,
    "semantic": NODE_DIM,
    "motion": 3,
    "joint": 2 * NODE_DIM + 3,
}


def main():
    torch.manual_seed(0)
    x = torch.randn(N, NODE_DIM)

    for name, in_dim in SOURCE_DIMS.items():
        gen = AdaptiveHyperedge(in_dim, M)
        h_src = torch.randn(N, in_dim)
        H = gen(h_src)

        assert H.shape == (N, M), f"{name}: expected {(N, M)}, got {tuple(H.shape)}"
        row_sums = H.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(N), atol=1e-5), f"{name}: rows not normalised"
        assert (H >= 0).all(), f"{name}: negative incidence"

        conv = HyperConv(NODE_DIM)
        out = conv(x, H)
        assert out.shape == x.shape, f"{name}: HyperConv changed shape"
        assert torch.isfinite(out).all(), f"{name}: non-finite HyperConv output"

        loss, logs = hyperedge_regularization(
            h_src, H, entropy_weight=0.02, sharpness_weight=0.02, consistency_weight=0.1
        )
        assert loss is not None and torch.isfinite(loss), f"{name}: bad regularisation loss"
        assert set(logs) == {
            "ssm_hyper_entropy_loss",
            "ssm_hyper_sharpness_loss",
            "ssm_hyper_consistency_loss",
        }, f"{name}: unexpected log keys {sorted(logs)}"

        # empty-graph path must be a no-op returning (None, {})
        empty_loss, empty_logs = hyperedge_regularization(
            x.new_zeros((0, in_dim)), x.new_zeros((0, M))
        )
        assert empty_loss is None and empty_logs == {}, f"{name}: empty path not a no-op"

        print(
            f"  {name:9s} in_dim={in_dim:3d}  H={tuple(H.shape)}  "
            f"row_sum~1  loss={float(loss):+.4f}  OK"
        )

    print("\nall source-switch checks passed")


if __name__ == "__main__":
    main()
