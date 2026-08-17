# attention-mps-torch
[![PyPI version](https://badge.fury.io/py/attention-mps-torch.svg)](https://badge.fury.io/py/attention-mps-torch)

attention-mps-torch provides custom PyTorch operators for invoking high performance Apple Silicon SDPA implementations during inference.

Supported implementations include:
 * Metal Performance Shaders Graph [scaledDotProductAttentionWithQueryTensor:keyTensor:valueTensor:maskTensor:scale:name:](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph/scaleddotproductattention(query:key:value:mask:scale:name:)?language=objc)
 * MLX [mlx.core.fast.scaled_dot_product_attention](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html). Note this project builds against my fork of [MLX](https://github.com/jhurt/mlx) that has a patch for allowing MLX arrays to wrap memory owned by PyTorch-created Metal buffers to avoid unnecessary data copying.

## Motivation
As of PyTorch 2.11.0, calling PyTorch's [torch.nn.functional.scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 
function using the MPS backend will either invoke custom Metal kernels or an implementation of 
attention that uses MPSGraph's gemm, transpose and softmax operations.

Howerver, for some shapes of Q, K, and V, MPSGraph's SDPA and/or MLX's SDPA are more performant.
Refer to the [benchmarks](#benchmark-results) for the difference in performance for various Q, K, and V shapes.

## Install from source
```
xcode-select --install
xcodebuild -downloadComponent MetalToolchain
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

## Benchmark Results

### M1 Max 32 GB RAM
| Data Type   | Shape (B,H,S,D)    |   Native (ms) |   MPS Graph (ms) | MPS Graph Speedup   |   MLX (ms) | MLX Speedup   |
|-------------|--------------------|---------------|------------------|---------------------|------------|---------------|
| float32     | (1, 1, 64, 32)     |        0.1017 |           0.2526 | 0.40x               |     0.5442 | 0.19x         |
| float32     | (2, 4, 128, 64)    |        0.2162 |           0.3297 | 0.66x               |     0.5129 | 0.42x         |
| float32     | (4, 8, 256, 128)   |        0.6575 |           0.622  | 1.06x               |     0.9658 | 0.68x         |
| float32     | (1, 12, 512, 64)   |        0.6034 |           0.5757 | 1.05x               |     0.7653 | 0.79x         |
| float32     | (2, 16, 1024, 32)  |        4.0947 |           1.516  | 2.70x               |     2.8082 | 1.46x         |
| float32     | (8, 1, 32, 128)    |        0.0934 |           0.2728 | 0.34x               |     0.4915 | 0.19x         |
| float32     | (1, 24, 4096, 256) |       94.697  |         153.187  | 0.62x               |    75.3196 | 1.26x         |
| float16     | (1, 1, 64, 32)     |        0.1079 |           0.2796 | 0.39x               |     0.5614 | 0.19x         |
| float16     | (2, 4, 128, 64)    |        0.193  |           0.2988 | 0.65x               |     0.5281 | 0.37x         |
| float16     | (4, 8, 256, 128)   |        0.6161 |           0.5529 | 1.11x               |     0.8023 | 0.77x         |
| float16     | (1, 12, 512, 64)   |        0.7155 |           0.5097 | 1.40x               |     0.7061 | 1.01x         |
| float16     | (2, 16, 1024, 32)  |        4.5482 |           1.2235 | 3.72x               |     2.1448 | 2.12x         |
| float16     | (8, 1, 32, 128)    |        0.1125 |           0.3157 | 0.36x               |     0.5484 | 0.21x         |
| float16     | (1, 24, 4096, 256) |       98.3478 |         147.19   | 0.67x               |    64.8966 | 1.52x         |
| bfloat16    | (1, 1, 64, 32)     |        0.1137 |           0.2963 | 0.38x               |     0.5765 | 0.20x         |
| bfloat16    | (2, 4, 128, 64)    |        0.2066 |           0.3895 | 0.53x               |     0.5426 | 0.38x         |
| bfloat16    | (4, 8, 256, 128)   |        0.8171 |           1.0856 | 0.75x               |     0.9101 | 0.90x         |
| bfloat16    | (1, 12, 512, 64)   |        0.7237 |           0.9612 | 0.75x               |     0.7272 | 1.00x         |
| bfloat16    | (2, 16, 1024, 32)  |        4.8268 |           2.5261 | 1.91x               |     2.4599 | 1.96x         |
| bfloat16    | (8, 1, 32, 128)    |        0.1152 |           0.3801 | 0.30x               |     0.5634 | 0.20x         |
| bfloat16    | (1, 24, 4096, 256) |      133.621  |         185.198  | 0.72x               |    75.7356 | 1.76x         |

### M3 Max 128 GB RAM
| Data Type   | Shape (B,H,S,D)     | PyTorch 2.12.0 (ms) |   MPS Graph (ms) | MPS Graph Speedup   | MLX 0.32.1 (ms) | MLX Speedup   |
|-------------|---------------------|---------------------|------------------|---------------------|-----------------|---------------|
| float32     | (1, 1, 64, 32)      | 0.0825              |           0.2944 | 0.28x               | 0.5814          | 0.14x         |
| float32     | (2, 4, 128, 64)     | 0.0859              |           0.299  | 0.29x               | 0.5349          | 0.16x         |
| float32     | (4, 8, 256, 128)    | 0.3698              |           0.4198 | 0.88x               | 0.7318          | 0.51x         |
| float32     | (1, 12, 512, 64)    | 0.3069              |           0.3939 | 0.78x               | 0.6403          | 0.48x         |
| float32     | (2, 16, 1024, 32)   | 3.67                |           0.7844 | 4.68x               | 2.3859          | 1.54x         |
| float32     | (8, 1, 32, 128)     | 0.0687              |           0.3405 | 0.20x               | 0.4928          | 0.14x         |
| float32     | (1, 24, 4096, 256)  | 70.6913             |          76.5113 | 0.92x               | 47.7988         | 1.48x         |
| float32     | (1, 128, 4096, 512) | 586.52              |        1105.38   | 0.53x               | 508.661         | 1.15x         |
| float32     | (1, 24, 8192, 128)  | 258.666             |         163.363  | 1.58x               | 122.376         | 2.11x         |
| float16     | (1, 1, 64, 32)      | 0.0874              |           0.2988 | 0.29x               | 0.5686          | 0.15x         |
| float16     | (2, 4, 128, 64)     | 0.1573              |           0.3472 | 0.45x               | 0.5516          | 0.29x         |
| float16     | (4, 8, 256, 128)    | 0.3647              |           0.3803 | 0.96x               | 0.6861          | 0.53x         |
| float16     | (1, 12, 512, 64)    | 0.3544              |           0.3584 | 0.99x               | 0.5984          | 0.59x         |
| float16     | (2, 16, 1024, 32)   | 3.9643              |           0.7211 | 5.50x               | 2.0067          | 1.98x         |
| float16     | (8, 1, 32, 128)     | 0.0962              |           0.2868 | 0.34x               | 0.4808          | 0.20x         |
| float16     | (1, 24, 4096, 256)  | 98.1663             |          57.6599 | 1.70x               | 45.5339         | 2.16x         |
| float16     | (1, 128, 4096, 512) | 1030.64             |         969.766  | 1.06x               | 544.43          | 1.89x         |
| float16     | (1, 24, 8192, 128)  | 405.578             |         113.65   | 3.57x               | 97.861          | 4.14x         |
| bfloat16    | (1, 1, 64, 32)      | 0.0881              |           0.3069 | 0.29x               | 0.5607          | 0.16x         |
| bfloat16    | (2, 4, 128, 64)     | 0.1609              |           0.3185 | 0.51x               | 0.4675          | 0.34x         |
| bfloat16    | (4, 8, 256, 128)    | 0.4823              |           0.375  | 1.29x               | 0.6952          | 0.69x         |
| bfloat16    | (1, 12, 512, 64)    | 0.3514              |           0.3519 | 1.00x               | 0.6028          | 0.58x         |
| bfloat16    | (2, 16, 1024, 32)   | 3.9333              |           0.7718 | 5.10x               | 2.0587          | 1.91x         |
| bfloat16    | (8, 1, 32, 128)     | 0.1007              |           0.3019 | 0.33x               | 0.4827          | 0.21x         |
| bfloat16    | (1, 24, 4096, 256)  | 111.371             |          63.9731 | 1.74x               | 48.7031         | 2.29x         |
| bfloat16    | (1, 128, 4096, 512) | 831.07              |         646.757  | 1.28x               | 440.921         | 1.88x         |
| bfloat16    | (1, 24, 8192, 128)  | 322.043             |          98.0884 | 3.28x               | 89.2385         | 3.61x         |

### M4 16 GB RAM
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
