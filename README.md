# Learning Triton

Experiments with custom kernel development using Triton.

## Developer Setup

Install uv:

```text
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies:

```text
uv sync
```

## Vector Addition

```text
uv run python -m learning_triton.addition
```

Which should output something like:

```text
x=tensor([1., 1., 1.,  ..., 1., 1., 1.], device='cuda:0')
y=tensor([1., 1., 1.,  ..., 1., 1., 1.], device='cuda:0')
z = x + y
z=tensor([2., 2., 2.,  ..., 2., 2., 2.], device='cuda:0')
triton_benchmark_time=0.006812520052724322
pytorch_benchmark_time=0.007304278626697691
```
