# attention_mps_torch
[![PyPI version](https://badge.fury.io/py/attention-mps-torch.svg)](https://badge.fury.io/py/attention-mps-torch)

attention_mps_torch is a custom PyTorch operator for invoking Metal Performance Shaders Graph [scaled dot product attention](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph/scaleddotproductattention(query:key:value:mask:scale:name:)?language=objc)
during inference.

As of PyTorch 2.11.0, calling PyTorch's [torch.nn.functional.scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 
function using the MPS backend will either invoke custom Metal kernels or an implementation of 
attention that uses MPSGraph's gemm, transpose and softmax operations
but does not use MPSGraph's `scaledDotProductAttentionWithQueryTensor:keyTensor:valueTensor:maskTensor:scale:name:` operation.
This operation is available since macOS 18.0 and is often faster for larger sequence lengths. 
See the [benchmark](#m3-max-benchmark-results) for the difference in performance for various Q, K, and V shapes.

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
attention_output = torch.ops.custom_ops.attention_mps(q, k, v, attention_mask=attention_mask)
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

## M3 Max Benchmark Results
| Data Type   | Q,K,V Shape (B,H,S,D) |   Native (ms) |   Custom (ms) | Speedup   |
|-------------|-----------------------|---------------|---------------|-----------|
| float32     | (1, 1, 64, 32)        |        0.1036 |        0.2845 | 0.36x     |
| float32     | (2, 4, 128, 64)       |        0.1406 |        0.3638 | 0.39x     |
| float32     | (4, 8, 256, 128)      |        0.3568 |        0.3582 | 1.00x     |
| float32     | (1, 12, 512, 64)      |        0.3007 |        0.3773 | 0.80x     |
| float32     | (2, 16, 1024, 32)     |        3.6885 |        0.7936 | 4.65x     |
| float32     | (8, 1, 32, 128)       |        0.076  |        0.2414 | 0.31x     |
| float32     | (1, 24, 4096, 256)    |       70.1794 |       70.9946 | 0.99x     |
| float32     | (1, 128, 4096, 512)   |      586.958  |     1083.19   | 0.54x     |
| float32     | (1, 24, 8192, 128)    |      220.881  |      125.949  | 1.75x     |
| float16     | (1, 1, 64, 32)        |        0.0704 |        0.3505 | 0.20x     |
| float16     | (2, 4, 128, 64)       |        0.0831 |        0.2872 | 0.29x     |
| float16     | (4, 8, 256, 128)      |        0.2774 |        0.3871 | 0.72x     |
| float16     | (1, 12, 512, 64)      |        0.2984 |        0.3438 | 0.87x     |
| float16     | (2, 16, 1024, 32)     |        3.9789 |        0.7366 | 5.40x     |
| float16     | (8, 1, 32, 128)       |        0.0696 |        0.2512 | 0.28x     |
| float16     | (1, 24, 4096, 256)    |       73.1595 |       38.8947 | 1.88x     |
| float16     | (1, 128, 4096, 512)   |      572.017  |      461.72   | 1.24x     |
| float16     | (1, 24, 8192, 128)    |      253.403  |       75.6061 | 3.35x     |
| bfloat16    | (1, 1, 64, 32)        |        0.0787 |        0.2378 | 0.33x     |
| bfloat16    | (2, 4, 128, 64)       |        0.0898 |        0.3306 | 0.27x     |
| bfloat16    | (4, 8, 256, 128)      |        0.27   |        0.4306 | 0.63x     |
| bfloat16    | (1, 12, 512, 64)      |        0.2971 |        0.3436 | 0.86x     |
| bfloat16    | (2, 16, 1024, 32)     |        3.9915 |        0.7494 | 5.33x     |
| bfloat16    | (8, 1, 32, 128)       |        0.0676 |        0.2374 | 0.28x     |
| bfloat16    | (1, 24, 4096, 256)    |       74.4551 |       40.4926 | 1.84x     |
| bfloat16    | (1, 128, 4096, 512)   |      579.861  |      500.145  | 1.16x     |
| bfloat16    | (1, 24, 8192, 128)    |      249.744  |       72.5937 | 3.44x     |

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.