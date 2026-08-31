"""Vector addition."""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel() -> None:
    """GPU addition kernel."""
    pass
