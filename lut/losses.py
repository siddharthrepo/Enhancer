"""LUT regularizers from the paper: 3D total variation (smoothness) and monotonicity.

Follows the reference implementation: squared neighbour differences along each
colour axis with the boundary cells weighted 2x, and a ReLU penalty on any
output value that decreases as an input colour increases.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LUTRegularizer(nn.Module):
    def __init__(self, dim: int = 33):
        super().__init__()
        w = torch.ones(dim - 1)
        w[[0, -1]] *= 2.0
        self.register_buffer("edge_w", w)

    def forward(self, luts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """luts: (N, 3, D, D, D) indexed [n, c, blue, green, red]. Returns (tv, monotonicity)."""
        tv = luts.new_zeros(())
        mn = luts.new_zeros(())
        w = self.edge_w
        for axis, shape in ((4, (1, 1, 1, 1, -1)), (3, (1, 1, 1, -1, 1)), (2, (1, 1, -1, 1, 1))):
            # LUT value at lattice i minus value at i + 1 along this colour axis
            diff = luts.narrow(axis, 0, luts.shape[axis] - 1) - luts.narrow(axis, 1, luts.shape[axis] - 1)
            tv = tv + (diff.pow(2) * w.view(shape)).mean()
            mn = mn + F.relu(diff).mean()
        return tv, mn


def lut_loss(pred, target, weights, luts, reg: LUTRegularizer, lambda_s=1e-4, lambda_m=10.0):
    mse = F.mse_loss(pred, target)
    tv, mn = reg(luts)
    total = mse + lambda_s * (tv + weights.pow(2).mean()) + lambda_m * mn
    return total, {"mse": mse.item(), "tv": tv.item(), "mn": mn.item()}
