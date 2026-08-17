# attention-mps-torch

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

| Data Type   | Shape (B,H,S,D)     |   PyTorch 2.13.0 |   MPS Graph (ms) | MPS Graph Speedup   |   MLX 0.32.1 (ms) | MLX Speedup   |
|-------------|---------------------|------------------|------------------|---------------------|-------------------|---------------|
| float32     | (1, 1, 64, 32)      |           0.0368 |           0.2802 | 0.13x               |            0.5683 | 0.06x         |
| float32     | (2, 4, 128, 64)     |           0.0371 |           0.2946 | 0.13x               |            0.4991 | 0.07x         |
| float32     | (4, 8, 256, 128)    |           0.3803 |           0.3711 | 1.02x               |            0.7239 | 0.53x         |
| float32     | (1, 12, 512, 64)    |           0.1767 |           0.446  | 0.40x               |            0.6178 | 0.29x         |
| float32     | (2, 16, 1024, 32)   |           0.459  |           0.7721 | 0.59x               |            2.3781 | 0.19x         |
| float32     | (8, 1, 32, 128)     |           0.0505 |           0.2649 | 0.19x               |            0.4888 | 0.10x         |
| float32     | (1, 24, 4096, 256)  |         154.501  |          76.5545 | 2.02x               |           47.604  | 3.25x         |
| float32     | (1, 128, 4096, 512) |         616.252  |        1084.56   | 0.57x               |          490.815  | 1.26x         |
| float32     | (1, 24, 8192, 128)  |         122.724  |         120.023  | 1.02x               |          104.998  | 1.17x         |
| float16     | (1, 1, 64, 32)      |           0.0385 |           0.2611 | 0.15x               |            0.6112 | 0.06x         |
| float16     | (2, 4, 128, 64)     |           0.0549 |           0.3567 | 0.15x               |            0.54   | 0.10x         |
| float16     | (4, 8, 256, 128)    |           0.3946 |           0.4184 | 0.94x               |            0.6824 | 0.58x         |
| float16     | (1, 12, 512, 64)    |           0.1497 |           0.3608 | 0.41x               |            0.6261 | 0.24x         |
| float16     | (2, 16, 1024, 32)   |           0.4679 |           0.7255 | 0.64x               |            2.0214 | 0.23x         |
| float16     | (8, 1, 32, 128)     |           0.0448 |           0.2664 | 0.17x               |            0.5235 | 0.09x         |
| float16     | (1, 24, 4096, 256)  |          78.4512 |          40.7157 | 1.93x               |           37.7336 | 2.08x         |
| float16     | (1, 128, 4096, 512) |         583.253  |         466.424  | 1.25x               |          382.221  | 1.53x         |
| float16     | (1, 24, 8192, 128)  |          76.4865 |          78.0709 | 0.98x               |           77.6958 | 0.98x         |
| bfloat16    | (1, 1, 64, 32)      |           0.037  |           0.3249 | 0.11x               |            0.5814 | 0.06x         |
| bfloat16    | (2, 4, 128, 64)     |           0.0587 |           0.3348 | 0.18x               |            0.5874 | 0.10x         |
| bfloat16    | (4, 8, 256, 128)    |           0.2486 |           0.452  | 0.55x               |            0.6812 | 0.37x         |
| bfloat16    | (1, 12, 512, 64)    |           0.1697 |           0.4282 | 0.40x               |            0.6164 | 0.28x         |
| bfloat16    | (2, 16, 1024, 32)   |           0.5611 |           0.7279 | 0.77x               |            1.9999 | 0.28x         |
| bfloat16    | (8, 1, 32, 128)     |           0.0214 |           0.2548 | 0.08x               |            0.5144 | 0.04x         |
| bfloat16    | (1, 24, 4096, 256)  |          79.5231 |          47.6671 | 1.67x               |           39.1355 | 2.03x         |
| bfloat16    | (1, 128, 4096, 512) |         694.526  |         594.115  | 1.17x               |          428.621  | 1.62x         |
| bfloat16    | (1, 24, 8192, 128)  |          84.5183 |          93.7658 | 0.90x               |           88.2005 | 0.96x         |

### M4 16 GB RAM
| Data Type   | Shape (B,H,S,D)    |   PyTorch 2.13.0 |   MPS Graph (ms) | MPS Graph Speedup   |   MLX 0.32.1 (ms) | MLX Speedup   |
|-------------|--------------------|------------------|------------------|---------------------|-------------------|---------------|
| float32     | (1, 1, 64, 32)     |           0.0541 |           0.6403 | 0.08x               |            0.8978 | 0.06x         |
| float32     | (2, 4, 128, 64)    |           0.0433 |           0.4391 | 0.10x               |            0.4373 | 0.10x         |
| float32     | (4, 8, 256, 128)   |           0.799  |           0.8324 | 0.96x               |            1.401  | 0.57x         |
| float32     | (1, 12, 512, 64)   |           0.4383 |           0.7696 | 0.57x               |            0.99   | 0.44x         |
| float32     | (2, 16, 1024, 32)  |           2.1346 |           2.1885 | 0.98x               |            6.871  | 0.31x         |
| float32     | (8, 1, 32, 128)    |           0.0211 |           0.2382 | 0.09x               |            0.424  | 0.05x         |
| float32     | (1, 24, 4096, 256) |         680.972  |         195.265  | 3.49x               |          219.355  | 3.10x         |
| float16     | (1, 1, 64, 32)     |           0.0381 |           0.6075 | 0.06x               |            0.7549 | 0.05x         |
| float16     | (2, 4, 128, 64)    |           0.1143 |           0.3982 | 0.29x               |            0.526  | 0.22x         |
| float16     | (4, 8, 256, 128)   |           0.4818 |           0.7896 | 0.61x               |            1.1582 | 0.42x         |
| float16     | (1, 12, 512, 64)   |           0.563  |           0.5894 | 0.96x               |            0.8539 | 0.66x         |
| float16     | (2, 16, 1024, 32)  |           1.9083 |           2.0822 | 0.92x               |            4.5711 | 0.42x         |
| float16     | (8, 1, 32, 128)    |           0.0393 |           0.5297 | 0.07x               |            0.4689 | 0.08x         |
| float16     | (1, 24, 4096, 256) |         318.926  |         173.17   | 1.84x               |          170.302  | 1.87x         |
| bfloat16    | (1, 1, 64, 32)     |           0.0374 |           0.5912 | 0.06x               |            0.8515 | 0.04x         |
| bfloat16    | (2, 4, 128, 64)    |           0.114  |           0.3635 | 0.31x               |            0.6078 | 0.19x         |
| bfloat16    | (4, 8, 256, 128)   |           0.7634 |           0.8037 | 0.95x               |            1.1528 | 0.66x         |
| bfloat16    | (1, 12, 512, 64)   |           0.5164 |           0.5627 | 0.92x               |            0.832  | 0.62x         |
| bfloat16    | (2, 16, 1024, 32)  |           1.8603 |           2.1031 | 0.88x               |            4.6118 | 0.40x         |
| bfloat16    | (8, 1, 32, 128)    |           0.0411 |           0.4677 | 0.09x               |            0.5036 | 0.08x         |
| bfloat16    | (1, 24, 4096, 256) |         318.005  |         177.42   | 1.79x               |          170.828  | 1.86x         |

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
