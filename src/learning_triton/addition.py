"""Vector addition."""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(
    x_ptr: tl.pointer_type,
    y_ptr: tl.pointer_type,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """GPU vector addition kernel.

    Args:
        x_ptr: Pointer to first element of ``x`` vector.
        y_ptr: Pointer to first element of ``y`` vector.
        n_elements: Number of elements in the vectors.
        BLOCK_SIZE: _description_
    """
    pass


def add(x: torch.Tensor, y: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    """Add tensors on GPU device.

    Args:
        x: First tensor.
        y: Second tensor.
        BLOCK_SIZE: _description_. Defaults to 1024.

    Returns:
        A tensor with the result of the addition operation. 

    Raises:
        ValueError: If ``x`` or ``y`` are not on a CUDA device.
    """
    if not (x.is_cuda and y.is_cuda):
        msg = f"Both x and y must be on CUDA device: {x.is_cuda=}; {y.is_cuda=}"
        raise ValueError(msg)

    return torch.tensor([1.0])
