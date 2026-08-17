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
