# attention-mps-torch

attention-mps-torch provides custom PyTorch operators for invoking high performance Apple Silicon SDPA implementations during inference.

Supported implementations include:
 * Metal Performance Shaders Graph [scaledDotProductAttentionWithQueryTensor:keyTensor:valueTensor:maskTensor:scale:name:](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph/scaleddotproductattention(query:key:value:mask:scale:name:)?language=objc)
 * MLX [mlx.core.fast.scaled_dot_product_attention](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html). Note this project builds against my fork of [MLX](https://github.com/jhurt/mlx) that has a patch for allowing MLX arrays to wrap memory owned by PyTorch-created Metal buffers to avoid unnecessary data copying.

## Motivation
Calling PyTorch's [torch.nn.functional.scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 
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
| Data Type   | Shape (B,H,S,D)    |   PyTorch 2.13.0 |   MPS Graph (ms) | MPS Graph Speedup   |   MLX 0.32.1 (ms) | MLX Speedup   |
|-------------|--------------------|------------------|------------------|---------------------|-------------------|---------------|
| float32     | (1, 1, 64, 32)     |           0.0271 |           0.2587 | 0.10x               |            0.5712 | 0.05x         |
| float32     | (2, 4, 128, 64)    |           0.0582 |           0.3342 | 0.17x               |            0.5761 | 0.10x         |
| float32     | (4, 8, 256, 128)   |           0.6143 |           0.6435 | 0.95x               |            1.004  | 0.61x         |
| float32     | (1, 12, 512, 64)   |           0.2945 |           0.594  | 0.50x               |            0.7607 | 0.39x         |
| float32     | (2, 16, 1024, 32)  |           1.0912 |           1.4993 | 0.73x               |            2.7011 | 0.40x         |
| float32     | (8, 1, 32, 128)    |           0.0365 |           0.3103 | 0.12x               |            0.5253 | 0.07x         |
| float32     | (1, 24, 4096, 256) |         282.25   |         153.305  | 1.84x               |           75.3863 | 3.74x         |
| float16     | (1, 1, 64, 32)     |           0.0284 |           0.2647 | 0.11x               |            0.569  | 0.05x         |
| float16     | (2, 4, 128, 64)    |           0.0536 |           0.3125 | 0.17x               |            0.5733 | 0.09x         |
| float16     | (4, 8, 256, 128)   |           0.4654 |           0.5535 | 0.84x               |            0.8059 | 0.58x         |
| float16     | (1, 12, 512, 64)   |           0.3631 |           0.5158 | 0.70x               |            0.7057 | 0.51x         |
| float16     | (2, 16, 1024, 32)  |           0.9469 |           1.2235 | 0.77x               |            2.1602 | 0.44x         |
| float16     | (8, 1, 32, 128)    |           0.0436 |           0.3066 | 0.14x               |            0.5455 | 0.08x         |
| float16     | (1, 24, 4096, 256) |         166.621  |         147.242  | 1.13x               |           64.9201 | 2.57x         |
| bfloat16    | (1, 1, 64, 32)     |           0.0322 |           0.3051 | 0.11x               |            0.5722 | 0.06x         |
| bfloat16    | (2, 4, 128, 64)    |           0.0627 |           0.4084 | 0.15x               |            0.5986 | 0.10x         |
| bfloat16    | (4, 8, 256, 128)   |           0.629  |           1.0868 | 0.58x               |            0.9183 | 0.68x         |
| bfloat16    | (1, 12, 512, 64)   |           0.4354 |           0.9536 | 0.46x               |            0.7717 | 0.56x         |
| bfloat16    | (2, 16, 1024, 32)  |           1.0331 |           2.5362 | 0.41x               |            2.4713 | 0.42x         |
| bfloat16    | (8, 1, 32, 128)    |           0.059  |           0.3748 | 0.16x               |            0.558  | 0.11x         |
| bfloat16    | (1, 24, 4096, 256) |         287.586  |         185.142  | 1.55x               |           75.7491 | 3.80x         |

### M3 Max 128 GB RAM
| Data Type   | Shape (B,H,S,D)     |   PyTorch 2.15.0 |   MPS Graph (ms) | MPS Graph Speedup   |   MLX 0.32.1 (ms) | MLX Speedup   |
|-------------|---------------------|------------------|------------------|---------------------|-------------------|---------------|
| float32     | (1, 1, 64, 32)      |           0.0346 |           0.3893 | 0.09x               |            0.5652 | 0.06x         |
| float32     | (2, 4, 128, 64)     |           0.0701 |           0.4533 | 0.15x               |            0.6094 | 0.12x         |
| float32     | (4, 8, 256, 128)    |           0.3912 |           0.442  | 0.88x               |            0.7939 | 0.49x         |
| float32     | (1, 12, 512, 64)    |           0.3884 |           0.4486 | 0.87x               |            0.7633 | 0.51x         |
| float32     | (2, 16, 1024, 32)   |           0.5361 |           0.9384 | 0.57x               |            2.4506 | 0.22x         |
| float32     | (8, 1, 32, 128)     |           0.0676 |           0.3778 | 0.18x               |            0.6953 | 0.10x         |
| float32     | (1, 24, 4096, 256)  |         154.46   |          67.3539 | 2.29x               |           58.071  | 2.66x         |
| float32     | (1, 128, 4096, 512) |         800.063  |         603.96   | 1.32x               |          528.821  | 1.51x         |
| float32     | (1, 24, 8192, 128)  |         136.793  |         149.08   | 0.92x               |          117.388  | 1.17x         |
| float16     | (1, 1, 64, 32)      |           0.0476 |           0.2477 | 0.19x               |            0.5019 | 0.09x         |
| float16     | (2, 4, 128, 64)     |           0.0374 |           0.2658 | 0.14x               |            0.4849 | 0.08x         |
| float16     | (4, 8, 256, 128)    |           0.1522 |           0.4067 | 0.37x               |            0.6129 | 0.25x         |
| float16     | (1, 12, 512, 64)    |           0.1224 |           0.3353 | 0.36x               |            0.5654 | 0.22x         |
| float16     | (2, 16, 1024, 32)   |           0.45   |           0.6841 | 0.66x               |            1.5073 | 0.30x         |
| float16     | (8, 1, 32, 128)     |           0.0303 |           0.2526 | 0.12x               |            0.4813 | 0.06x         |
| float16     | (1, 24, 4096, 256)  |          87.901  |          45.1683 | 1.95x               |           42.3316 | 2.08x         |
| float16     | (1, 128, 4096, 512) |         736.677  |         457.438  | 1.61x               |          420.352  | 1.75x         |
| float16     | (1, 24, 8192, 128)  |          80.6187 |          89.6483 | 0.90x               |           83.9063 | 0.96x         |
| bfloat16    | (1, 1, 64, 32)      |           0.0473 |           0.263  | 0.18x               |            0.492  | 0.10x         |
| bfloat16    | (2, 4, 128, 64)     |           0.079  |           0.272  | 0.29x               |            0.4708 | 0.17x         |
| bfloat16    | (4, 8, 256, 128)    |           0.2836 |           0.363  | 0.78x               |            0.6081 | 0.47x         |
| bfloat16    | (1, 12, 512, 64)    |           0.186  |           0.4144 | 0.45x               |            0.5671 | 0.33x         |
| bfloat16    | (2, 16, 1024, 32)   |           0.5071 |           0.7265 | 0.70x               |            1.4908 | 0.34x         |
| bfloat16    | (8, 1, 32, 128)     |           0.0537 |           0.2949 | 0.18x               |            0.4831 | 0.11x         |
| bfloat16    | (1, 24, 4096, 256)  |          91.1082 |          48.0427 | 1.90x               |           43.6296 | 2.09x         |
| bfloat16    | (1, 128, 4096, 512) |         779.702  |         465.043  | 1.68x               |          416.868  | 1.87x         |
| bfloat16    | (1, 24, 8192, 128)  |          79.98   |          88.5877 | 0.90x               |           85.0303 | 0.94x         |

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
