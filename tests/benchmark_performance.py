# Copyright 2026 Jason Hurt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import pandas as pd
from tabulate import tabulate
import time
import torch
import torch.nn.functional as F

import attention_mps

def benchmark_operation(operation_func, args, iterations=20, warmup=10):
    """
    Measures the execution time of a GPU operation with proper synchronization.
    """
    for _ in range(warmup):
        operation_func(*args)

    torch.mps.synchronize()

    start_time = time.perf_counter()
    for _ in range(iterations):
        operation_func(*args)
    torch.mps.synchronize()
    end_time = time.perf_counter()

    average_ms = ((end_time - start_time) / iterations) * 1000
    return average_ms


def run_benchmarks():
    if not torch.backends.mps.is_available():
        print("MPS is not available on this system.")
        return

    device = "mps"
    dtypes = [torch.float32, torch.float16, torch.bfloat16]

    shapes = [
        (1, 1, 64, 32),
        (2, 4, 128, 64),
        (4, 8, 256, 128),
        (1, 12, 512, 64),
        (2, 16, 1024, 32),
        (8, 1, 32, 128),
        (1, 24, 4096, 256),
        (1, 128, 4096, 512),
        (1, 24, 8192, 128),
    ]
    all_results = []

    for dtype in dtypes:
        atol = 1e-3 if dtype == torch.float32 else 1e-2
        rtol = 1e-3 if dtype == torch.float32 else 1e-2

        print(f"\nEvaluating Dtype: {dtype}")

        for batch_size, head_count, seq_len, head_dim in shapes:
            print(f"\nEvaluating Shape: ({batch_size}, {head_count}, {seq_len}, {head_dim})")

            query_tensor = torch.randn(batch_size, head_count, seq_len, head_dim, device=device, dtype=dtype)
            key_tensor = torch.randn(batch_size, head_count, seq_len, head_dim, device=device, dtype=dtype)
            value_tensor = torch.randn(batch_size, head_count, seq_len, head_dim, device=device, dtype=dtype)

            with torch.no_grad():
                expected_output = F.scaled_dot_product_attention(
                    query_tensor, key_tensor, value_tensor, attn_mask=None
                )

                mps_graph_output = torch.ops.custom_ops.attention_mps_graph(
                    query_tensor, key_tensor, value_tensor, None
                )
                torch.testing.assert_close(mps_graph_output, expected_output, atol=atol, rtol=rtol)

                mlx_output = torch.ops.custom_ops.attention_mlx(
                    query_tensor, key_tensor, value_tensor, None
                )
                torch.testing.assert_close(mlx_output, expected_output, atol=atol, rtol=rtol)

            native_func = lambda q, k, v: F.scaled_dot_product_attention(q, k, v, attn_mask=None)
            native_ms = benchmark_operation(native_func, (query_tensor, key_tensor, value_tensor))

            mps_graph_func = lambda q, k, v: torch.ops.custom_ops.attention_mps_graph(q, k, v, None)
            mps_graph_ms = benchmark_operation(mps_graph_func, (query_tensor, key_tensor, value_tensor))
            mps_graph_speedup = native_ms / mps_graph_ms

            mlx_func = lambda q, k, v: torch.ops.custom_ops.attention_mlx(q, k, v, None)
            mlx_ms = benchmark_operation(mlx_func, (query_tensor, key_tensor, value_tensor))
            mlx_speedup = native_ms / mlx_ms

            all_results.append({
                "Data Type": str(dtype).replace("torch.", ""),
                "Shape (B,H,S,D)": f"({batch_size}, {head_count}, {seq_len}, {head_dim})",
                "Native (ms)": f"{native_ms:.4f}",
                "MPS Graph (ms)": f"{mps_graph_ms:.4f}",
                "MPS Graph Speedup": f"{mps_graph_speedup:.2f}x",
                "MLX (ms)": f"{mlx_ms:.4f}",
                "MLX Speedup": f"{mlx_speedup:.2f}x",
            })

            del query_tensor, key_tensor, value_tensor, expected_output, mps_graph_output, mlx_output
            gc.collect()
            torch.mps.empty_cache()

    df = pd.DataFrame(all_results)
    print("\nBenchmark Summary:")
    print(tabulate(df, headers='keys', tablefmt='github', showindex=False))


if __name__ == "__main__":
    run_benchmarks()
