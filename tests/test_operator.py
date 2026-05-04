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

import pytest
import torch
import torch.nn.functional as F

import attention_mps


class TestAttentionMPS:
    @pytest.mark.parametrize("batch_size, head_count, sequence_length, head_dim", [
        (1, 1, 64, 32),
        (2, 4, 128, 64),
        (4, 8, 256, 128),
        (1, 12, 512, 64),
        (2, 16, 1024, 32),
        (8, 1, 32, 128),
        (1, 24, 4192, 256),
        (1, 128, 4608, 512),
        (2, 32, 8192, 256),
    ])
    def test_functional_correctness(self, batch_size, head_count, sequence_length, head_dim):
        """
        Validates the custom attention_mps op against torch.nn.functional.scaled_dot_product_attention
        using the (batch_size, head_count, sequence_length, head_dim) shape format.
        """
        torch.manual_seed(42)
        dtype = torch.float32
        device = "mps"

        query_tensor = torch.randn(
            batch_size, head_count, sequence_length, head_dim,
            device=device, dtype=dtype
        )
        key_tensor = torch.randn(
            batch_size, head_count, sequence_length, head_dim,
            device=device, dtype=dtype
        )
        value_tensor = torch.randn(
            batch_size, head_count, sequence_length, head_dim,
            device=device, dtype=dtype
        )

        expected_output = F.scaled_dot_product_attention(
            query_tensor,
            key_tensor,
            value_tensor,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False
        )

        actual_output = torch.ops.custom_ops.attention_mps(
            query_tensor,
            key_tensor,
            value_tensor,
            None
        )

        torch.testing.assert_close(actual_output, expected_output, atol=1e-4, rtol=1e-4)

    def test_q_kv_different_shapes(self):
        torch.manual_seed(42)
        dtype = torch.float32
        device = "mps"

        q_shape = (1, 12, 512, 64)
        kv_shape = (1, 12, 1024, 64)
        query_tensor = torch.randn(
            q_shape,
            device=device, dtype=dtype
        )
        key_tensor = torch.randn(
            kv_shape,
            device=device, dtype=dtype
        )
        value_tensor = torch.randn(
            kv_shape,
            device=device, dtype=dtype
        )

        expected_output = F.scaled_dot_product_attention(
            query_tensor,
            key_tensor,
            value_tensor,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False
        )

        actual_output = torch.ops.custom_ops.attention_mps(
            query_tensor,
            key_tensor,
            value_tensor,
            None
        )

        torch.testing.assert_close(actual_output, expected_output, atol=1e-4, rtol=1e-4)

        q_shape = (1, 12, 1024, 64)
        kv_shape = (1, 12, 512, 64)
        query_tensor = torch.randn(
            q_shape,
            device=device, dtype=dtype
        )
        key_tensor = torch.randn(
            kv_shape,
            device=device, dtype=dtype
        )
        value_tensor = torch.randn(
            kv_shape,
            device=device, dtype=dtype
        )

        expected_output = F.scaled_dot_product_attention(
            query_tensor,
            key_tensor,
            value_tensor,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False
        )

        actual_output = torch.ops.custom_ops.attention_mps(
            query_tensor,
            key_tensor,
            value_tensor,
            None
        )

        torch.testing.assert_close(actual_output, expected_output, atol=1e-4, rtol=1e-4)
