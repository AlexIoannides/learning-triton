"""Vector addition."""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(
    x_ptr: tl.pointer_type,
    y_ptr: tl.pointer_type,
    z_ptr: tl.pointer_type,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """GPU vector addition kernel.

    Args:
        x_ptr: Pointer to first element of the ``x`` input vector.
        y_ptr: Pointer to first element of the ``y`` input vector.
        y_ptr: Pointer to first element of the ``y`` output vector.
        n_elements: Number of elements in the vectors.
        BLOCK_SIZE: The number of data elements operated on within a single block.
    """
    program_id = tl.program_id(axis=0)  # Retreive the program ID for this block
    block_start = program_id * BLOCK_SIZE  # Starting array index for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # All array indices for this block
    mask = offsets < n_elements  # If n_elements % BLOCK_SIZE != 0 mask out-of-range


def add(x: torch.Tensor, y: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    """Add tensors on GPU device.

    Args:
        x: First tensor.
        y: Second tensor.
        BLOCK_SIZE: The number of data elements operated on within a single block.
            Defaults to 1024.

    Returns:
        A tensor with the result of the addition operation.

    Raises:
        ValueError: If ``x`` or ``y`` are not on a CUDA device.
    """
    if not (x.is_cuda and y.is_cuda):
        msg = f"Both x and y must be on CUDA device: {x.is_cuda=}; {y.is_cuda=}"
        raise ValueError(msg)

    z = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: triton.cdiv(n_elements, meta["BLOCK_SIZE"])
    _add_kernel[grid](x, y, z, n_elements)

    return z
