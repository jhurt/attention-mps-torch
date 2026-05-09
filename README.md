# attention_mps_torch
[![PyPI version](https://badge.fury.io/py/attention-mps-torch.svg)](https://badge.fury.io/py/attention-mps-torch)

attention_mps_torch provides custom PyTorch operators for invoking high performance Apple Silicon SDPA backends during inference.

Supported backends include:
 * Metal Performance Shaders Graph [scaledDotProductAttentionWithQueryTensor:keyTensor:valueTensor:maskTensor:scale:name:](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph/scaleddotproductattention(query:key:value:mask:scale:name:)?language=objc)
 * MLX [mlx.core.fast.scaled_dot_product_attention](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html). Note this project builds against my fork of [MLX](https://github.com/jhurt/mlx) that has a patch for allowing MLX arrays to wrap memory owned by PyTorch-created Metal buffers to avoid unnecessary data copying.

## Motivation
As of PyTorch 2.11.0, calling PyTorch's [torch.nn.functional.scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 
function using the MPS backend will either invoke custom Metal kernels or an implementation of 
attention that uses MPSGraph's gemm, transpose and softmax operations.

Howerver, for some shapes of Q, K, and V, MPSGraph's SDPA and/or MLX's SDPA are more performant.
Refer to the [benchmark](#m3-max-benchmark-results) for the difference in performance for various Q, K, and V shapes.

## Install
```
pip install attention-mps-torch
```

## Install from source
```
xcode-select --install 
pip install .
```

## Usage
```python
import attention_mps
import torch

# define q, k, v
# optionally define attention_mask

# call attention_mps.attention_mps_graph
attention_output = torch.ops.custom_ops.attention_mps_graph(q, k, v, attention_mask=attention_mask)

# call attention_mps.attention_mlx
attention_output = torch.ops.custom_ops.attention_mlx(q, k, v, attention_mask=attention_mask)
```

## Run Tests
```
pip install -e ".[test]"
python3 -m pytest -v tests/test_operator.py
```

## Run Benchmark
```
pip install -e ".[test]"
python3 tests/benchmark_performance.py
```

## M3 Max 128 GB RAM Benchmark Results
| Data Type   | Shape (B,H,S,D)     |   Native (ms) |   MPS Graph (ms) | MPS Graph Speedup   |   MLX (ms) | MLX Speedup   |
|-------------|---------------------|---------------|------------------|---------------------|------------|---------------|
| float32     | (1, 1, 64, 32)      |        0.0743 |           0.2322 | 0.32x               |     0.4591 | 0.16x         |
| float32     | (2, 4, 128, 64)     |        0.1332 |           0.3067 | 0.43x               |     0.4755 | 0.28x         |
| float32     | (4, 8, 256, 128)    |        0.305  |           0.3375 | 0.90x               |     0.6162 | 0.50x         |
| float32     | (1, 12, 512, 64)    |        0.4608 |           0.3902 | 1.18x               |     0.6128 | 0.75x         |
| float32     | (2, 16, 1024, 32)   |        3.7303 |           0.7917 | 4.71x               |     2.3611 | 1.58x         |
| float32     | (8, 1, 32, 128)     |        0.0826 |           0.2251 | 0.37x               |     0.4142 | 0.20x         |
| float32     | (1, 24, 4096, 256)  |       70.5377 |          79.2986 | 0.89x               |    48.6256 | 1.45x         |
| float32     | (1, 128, 4096, 512) |      614.069  |        1228.02   | 0.50x               |   499.373  | 1.23x         |
| float32     | (1, 24, 8192, 128)  |      236.209  |         154.243  | 1.53x               |   108.217  | 2.18x         |
| float16     | (1, 1, 64, 32)      |        0.0644 |           0.3099 | 0.21x               |     0.5135 | 0.13x         |
| float16     | (2, 4, 128, 64)     |        0.0685 |           0.2317 | 0.30x               |     0.408  | 0.17x         |
| float16     | (4, 8, 256, 128)    |        0.2733 |           0.421  | 0.65x               |     0.5509 | 0.50x         |
| float16     | (1, 12, 512, 64)    |        0.2754 |           0.2962 | 0.93x               |     0.516  | 0.53x         |
| float16     | (2, 16, 1024, 32)   |        4.0599 |           0.7068 | 5.74x               |     1.8057 | 2.25x         |
| float16     | (8, 1, 32, 128)     |        0.0763 |           0.206  | 0.37x               |     0.4205 | 0.18x         |
| float16     | (1, 24, 4096, 256)  |       76.9161 |          42.2653 | 1.82x               |    37.9382 | 2.03x         |
| float16     | (1, 128, 4096, 512) |      600.736  |         539.977  | 1.11x               |   373.658  | 1.61x         |
| float16     | (1, 24, 8192, 128)  |      261.41   |          88.5494 | 2.95x               |    75.1817 | 3.48x         |
| bfloat16    | (1, 1, 64, 32)      |        0.0631 |           0.2641 | 0.24x               |     0.4223 | 0.15x         |
| bfloat16    | (2, 4, 128, 64)     |        0.08   |           0.2807 | 0.29x               |     0.429  | 0.19x         |
| bfloat16    | (4, 8, 256, 128)    |        0.2764 |           0.3301 | 0.84x               |     0.5858 | 0.47x         |
| bfloat16    | (1, 12, 512, 64)    |        0.2524 |           0.3144 | 0.80x               |     0.5175 | 0.49x         |
| bfloat16    | (2, 16, 1024, 32)   |        4.0296 |           0.6933 | 5.81x               |     1.8074 | 2.23x         |
| bfloat16    | (8, 1, 32, 128)     |        0.0621 |           0.1971 | 0.32x               |     0.377  | 0.16x         |
| bfloat16    | (1, 24, 4096, 256)  |       80.6665 |          44.7403 | 1.80x               |    38.046  | 2.12x         |
| bfloat16    | (1, 128, 4096, 512) |      638.942  |         558.299  | 1.14x               |   371.982  | 1.72x         |
| bfloat16    | (1, 24, 8192, 128)  |      255.026  |          96.8845 | 2.63x               |    79.3998 | 3.21x         |

## M4 16 GB RAM Benchmark Results
| Data Type   | Shape (B,H,S,D)    |   Native (ms) |   MPS Graph (ms) | MPS Graph Speedup   |   MLX (ms) | MLX Speedup   |
|-------------|--------------------|---------------|------------------|---------------------|------------|---------------|
| float32     | (1, 1, 64, 32)     |        0.0747 |           0.2725 | 0.27x               |     0.573  | 0.13x         |
| float32     | (2, 4, 128, 64)    |        0.2674 |           0.3232 | 0.83x               |     0.6704 | 0.40x         |
| float32     | (4, 8, 256, 128)   |        1.4591 |           0.7413 | 1.97x               |     1.3142 | 1.11x         |
| float32     | (1, 12, 512, 64)   |        1.3835 |           0.7667 | 1.80x               |     0.9661 | 1.43x         |
| float32     | (2, 16, 1024, 32)  |       14.096  |           2.1518 | 6.55x               |     6.8876 | 2.05x         |
| float32     | (8, 1, 32, 128)    |        0.0518 |           0.1858 | 0.28x               |     0.3661 | 0.14x         |
| float32     | (1, 24, 4096, 256) |      286.783  |         191.849  | 1.49x               |   219.435  | 1.31x         |
| float16     | (1, 1, 64, 32)     |        0.0545 |           0.3076 | 0.18x               |     0.4185 | 0.13x         |
| float16     | (2, 4, 128, 64)    |        0.0772 |           0.2239 | 0.34x               |     0.3895 | 0.20x         |
| float16     | (4, 8, 256, 128)   |        1.46   |           0.6745 | 2.16x               |     1.0732 | 1.36x         |
| float16     | (1, 12, 512, 64)   |        1.7188 |           0.5546 | 3.10x               |     0.8472 | 2.03x         |
| float16     | (2, 16, 1024, 32)  |       14.7816 |           2.0584 | 7.18x               |     4.5086 | 3.28x         |
| float16     | (8, 1, 32, 128)    |        0.0544 |           0.1647 | 0.33x               |     0.3504 | 0.16x         |
| float16     | (1, 24, 4096, 256) |      302.955  |         186.339  | 1.63x               |   175.23   | 1.73x         |
| bfloat16    | (1, 1, 64, 32)     |        0.051  |           0.1852 | 0.28x               |     0.3946 | 0.13x         |
| bfloat16    | (2, 4, 128, 64)    |        0.0838 |           0.2241 | 0.37x               |     0.4145 | 0.20x         |
| bfloat16    | (4, 8, 256, 128)   |        1.4498 |           0.719  | 2.02x               |     1.0984 | 1.32x         |
| bfloat16    | (1, 12, 512, 64)   |        1.5777 |           0.5704 | 2.77x               |     0.8491 | 1.86x         |
| bfloat16    | (2, 16, 1024, 32)  |       14.6453 |           2.073  | 7.06x               |     5.7285 | 2.56x         |
| bfloat16    | (8, 1, 32, 128)    |        0.0688 |           0.2655 | 0.26x               |     0.3538 | 0.19x         |
| bfloat16    | (1, 24, 4096, 256) |      316.168  |         185.893  | 1.70x               |   175.203  | 1.80x         |


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
