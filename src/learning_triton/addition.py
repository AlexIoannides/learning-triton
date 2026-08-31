"""Vector addition.

This module demonstrates a simple Triton kernel that adds two CUDA tensors
element-wise. The first time this script is run, Triton will compile the kernel
and cache the artifacts in the ``.triton`` directory. For example:

    ll ../.triton/cache/B2JIAJYUKPCXPPTRDFJVO4BWZEWLZTW3XVC4E6AULWZEXFNYMGIA/
    total 56
    drwxr-xr-x 2 root root 4096 Aug 31 15:48 ./
    drwxr-xr-x 8 root root 4096 Aug 31 15:56 ../
    -rw-r--r-- 1 root root  814 Aug 31 15:48 __grp___add_kernel.json
    -rw-r--r-- 1 root root 6376 Aug 31 15:48 _add_kernel.cubin
    -rw-r--r-- 1 root root 1100 Aug 31 15:48 _add_kernel.json
    -rw-r--r-- 1 root root 6906 Aug 31 15:48 _add_kernel.llir
    -rw-r--r-- 1 root root 5560 Aug 31 15:48 _add_kernel.ptx
    -rw-r--r-- 1 root root 4759 Aug 31 15:48 _add_kernel.source
    -rw-r--r-- 1 root root 3670 Aug 31 15:48 _add_kernel.ttgir
    -rw-r--r-- 1 root root 3264 Aug 31 15:48 _add_kernel.ttir

Run this script with:

    uv run python -m learning_triton.addition
"""

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
        z_ptr: Pointer to first element of the ``y`` output vector.
        n_elements: Number of elements in the vectors.
        BLOCK_SIZE: The number of data elements operated on within a single block.
    """
    program_id = tl.program_id(axis=0)  # Retrieve the program ID for this block
    block_start = program_id * BLOCK_SIZE  # Starting array index for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # All array indices for this block
    mask = offsets < n_elements  # If n_elements % BLOCK_SIZE != 0 mask out-of-range

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = x + y

    tl.store(z_ptr + offsets, z, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    """Add tensors on GPU device.

    Args:
        x: First tensor.
        y: Second tensor.
        BLOCK_SIZE: The number of data elements operated on within a single block. Note,
            Triton will compile one kernel per-block size. Thus, the block size needs to
            be defined in advance. Defaults to 1024.

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
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # 1D grid
    _add_kernel[grid](x, y, z, n_elements, BLOCK_SIZE=BLOCK_SIZE)  # type: ignore

    return z


if __name__ == "__main__":
    # Basic kernel tests
    n_elements = 1024 * 100  # 4 * 1024 * 100 / (1024 ^ 2) = 0.4 MB
    x = torch.ones(n_elements, device="cuda", dtype=torch.float32)
    y = torch.ones(n_elements, device="cuda", dtype=torch.float32)
    z = add(x, y)
    print(f"{x=}")
    print(f"{y=}")
    print(f"z = x + y\n{z=}")

    # Testing with Triton
    triton.testing.assert_close(z, x + y)
    triton_benchmark_time = triton.testing.do_bench(
        lambda: add(x, y),
        warmup=25,
        rep=1000,
    )
    pytorch_benchmark_time = triton.testing.do_bench(
        lambda: x + y,
        warmup=25,
        rep=1000,
    )
    print(f"{triton_benchmark_time=}")
    print(f"{pytorch_benchmark_time=}")
