/*
 * Copyright 2026 Jason Hurt
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "library.h"

#include <torch/library.h>
#include <ATen/native/mps/OperationUtils.h>

#include <mlx/mlx.h>
namespace mx = mlx::core;

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

TORCH_LIBRARY(custom_ops, m) {
    m.def("attention_mps_graph(Tensor q, Tensor k, Tensor v, Tensor? attention_mask) -> Tensor");
    m.impl("attention_mps_graph", c10::DispatchKey::MPS, TORCH_FN(attention_mps_graph));
    
    m.def("attention_mlx(Tensor q, Tensor k, Tensor v, Tensor? attention_mask) -> Tensor");
    m.impl("attention_mlx", c10::DispatchKey::MPS, TORCH_FN(attention_mlx));
}

at::Tensor attention_mps_graph(const at::Tensor& q_tensor,
                               const at::Tensor& k_tensor,
                               const at::Tensor& v_tensor,
                               const std::optional<at::Tensor>& attention_mask_tensor) {
    TORCH_CHECK(q_tensor.device().is_mps(), "q_tensor must be on MPS");
    TORCH_CHECK(q_tensor.is_contiguous(), "q_tensor must be contiguous");
    
    TORCH_CHECK(k_tensor.device().is_mps(), "k_tensor must be on MPS");
    TORCH_CHECK(k_tensor.is_contiguous(), "k_tensor must be contiguous");
    
    TORCH_CHECK(v_tensor.device().is_mps(), "v_tensor must be on MPS");
    TORCH_CHECK(v_tensor.is_contiguous(), "v_tensor must be contiguous");
    
    MPSDataType mpsDataTypeAttentionMask;
    if(attention_mask_tensor.has_value()) {
        TORCH_CHECK((*attention_mask_tensor).device().is_mps(), "attention_mask_tensor must be on MPS");
        TORCH_CHECK((*attention_mask_tensor).is_contiguous(), "attention_mask_tensor must be contiguous");
        mpsDataTypeAttentionMask = at::native::mps::getMPSScalarType((*attention_mask_tensor).scalar_type());
    }
    
    auto mpsDataType = at::native::mps::getMPSScalarType(q_tensor.scalar_type());
    
    auto batch_size = q_tensor.size(0);
    auto head_count = q_tensor.size(1);
    auto sequence_length_q = q_tensor.size(2);
    auto sequence_length_kv = k_tensor.size(2);
    auto head_dimension = q_tensor.size(3);
    
    at::Tensor output_tensor = at::empty({batch_size, head_count, sequence_length_q, head_dimension}, q_tensor.options());
    
    @autoreleasepool {
        static NSMutableDictionary<NSString *, MPSGraphExecutable *> *mpsGraphExecutableCache = nil;
        static dispatch_once_t onceToken;
        dispatch_once(&onceToken, ^{
            mpsGraphExecutableCache = [[NSMutableDictionary alloc] init];
        });
        
        id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
        
        dispatch_sync(torch::mps::get_dispatch_queue(), ^(){
            std::string mps_data_type_string = at::native::mps::getMPSTypeString(q_tensor.scalar_type());
            NSString *mpsGraphExecutableKey = [NSString stringWithFormat:@"%lld_%lld_%lld_%lld_%lld_%d_%s", batch_size, head_count, sequence_length_q, sequence_length_kv, head_dimension, attention_mask_tensor.has_value(), mps_data_type_string.c_str()];
            
            NSArray<NSNumber*> *qShape = @[@(batch_size), @(head_count), @(sequence_length_q), @(head_dimension)];
            NSArray<NSNumber*> *kvShape = @[@(batch_size), @(head_count), @(sequence_length_kv), @(head_dimension)];
            NSArray<NSNumber*> *attentionMaskShape;
            
            if(attention_mask_tensor.has_value()) {
                attentionMaskShape = @[
                    @((*attention_mask_tensor).size(0)),
                    @((*attention_mask_tensor).size(1)),
                    @((*attention_mask_tensor).size(2)),
                    @((*attention_mask_tensor).size(3)),
                ];
                mpsGraphExecutableKey = [NSString stringWithFormat:@"%@_%lld_%lld_%lld_%lld",
                                         mpsGraphExecutableKey,
                                         attentionMaskShape[0].longLongValue,
                                         attentionMaskShape[1].longLongValue,
                                         attentionMaskShape[2].longLongValue,
                                         attentionMaskShape[3].longLongValue];
            }
            
            MPSGraphExecutable *mpsGraphExecutable = mpsGraphExecutableCache[mpsGraphExecutableKey];
            
            if (!mpsGraphExecutable) {
                MPSGraphDevice *graphDevice = [MPSGraphDevice deviceWithMTLDevice:commandBuffer.device];
                MPSGraph *graph = [[MPSGraph alloc] init];
                
                MPSGraphTensor *mpsQ = [graph placeholderWithShape:qShape dataType:mpsDataType name:@"mpsQ"];
                MPSGraphTensor *mpsK = [graph placeholderWithShape:kvShape dataType:mpsDataType name:@"mpsK"];
                MPSGraphTensor *mpsV = [graph placeholderWithShape:kvShape dataType:mpsDataType name:@"mpsV"];
                MPSGraphTensor *mpsAttentionMask = nil;
                if(attention_mask_tensor.has_value()) {
                    mpsAttentionMask = [graph placeholderWithShape:attentionMaskShape dataType:mpsDataTypeAttentionMask name:@"mpsAttentionMask"];
                }
                
                float scale = 1.0f / sqrt(static_cast<float>(head_dimension));
                
                MPSGraphTensor *result = [graph scaledDotProductAttentionWithQueryTensor:mpsQ
                                                                               keyTensor:mpsK
                                                                             valueTensor:mpsV
                                                                              maskTensor:mpsAttentionMask
                                                                                   scale:scale
                                                                                    name:mpsGraphExecutableKey];
                
                NSMutableDictionary *feeds = [NSMutableDictionary dictionaryWithCapacity:4];
                feeds[mpsQ] = [[MPSGraphShapedType alloc] initWithShape:qShape dataType:mpsDataType];
                feeds[mpsK] = [[MPSGraphShapedType alloc] initWithShape:kvShape dataType:mpsDataType];
                feeds[mpsV] = [[MPSGraphShapedType alloc] initWithShape:kvShape dataType:mpsDataType];
                if(attention_mask_tensor.has_value()) {
                    feeds[mpsAttentionMask] = [[MPSGraphShapedType alloc] initWithShape:attentionMaskShape dataType:mpsDataTypeAttentionMask];
                }
                
                MPSGraphCompilationDescriptor *compilationDescriptor = [[MPSGraphCompilationDescriptor alloc] init];
                compilationDescriptor.optimizationLevel = MPSGraphOptimizationLevel1;
                compilationDescriptor.reducedPrecisionFastMath = MPSGraphReducedPrecisionFastMathAllowFP16Intermediates;
                mpsGraphExecutable = [graph compileWithDevice:graphDevice
                                                        feeds:feeds
                                                targetTensors:@[result]
                                             targetOperations:nil
                                        compilationDescriptor:compilationDescriptor];
                
                mpsGraphExecutableCache[mpsGraphExecutableKey] = mpsGraphExecutable;
            }
            
            NSMutableArray *inputsArray = [NSMutableArray arrayWithCapacity:4];
            [inputsArray addObject:[[MPSGraphTensorData alloc]
                                    initWithMTLBuffer:at::native::mps::getMTLBufferStorage(q_tensor)
                                    shape:qShape
                                    dataType:mpsDataType]];
            [inputsArray addObject:[[MPSGraphTensorData alloc]
                                    initWithMTLBuffer:at::native::mps::getMTLBufferStorage(k_tensor)
                                    shape:kvShape
                                    dataType:mpsDataType]];
            [inputsArray addObject:[[MPSGraphTensorData alloc]
                                    initWithMTLBuffer:at::native::mps::getMTLBufferStorage(v_tensor)
                                    shape:kvShape
                                    dataType:mpsDataType]];
            if(attention_mask_tensor.has_value()) {
                [inputsArray addObject:[[MPSGraphTensorData alloc]
                                        initWithMTLBuffer:at::native::mps::getMTLBufferStorage(*attention_mask_tensor)
                                        shape:attentionMaskShape
                                        dataType:mpsDataTypeAttentionMask]];
            }
            
            MPSGraphTensorData *output = [[MPSGraphTensorData alloc]
                                          initWithMTLBuffer:at::native::mps::getMTLBufferStorage(output_tensor)
                                          shape:qShape
                                          dataType:mpsDataType];
            
            [mpsGraphExecutable encodeToCommandBuffer:commandBuffer
                                          inputsArray:inputsArray
                                         resultsArray:@[output]
                                  executionDescriptor:nil];
            
            torch::mps::commit();
        });
    }
    
    torch::mps::synchronize();
    
    return output_tensor;
}

mx::Dtype torch_dtype_to_mlx_dtype(at::ScalarType torch_dtype) {
    switch (torch_dtype) {
        case at::kFloat: return mx::float32;
        case at::kHalf: return mx::float16;
        case at::kBFloat16: return mx::bfloat16;
        case at::kDouble: return mx::float64;
        case at::kBool: return mx::bool_;
        default:
            throw std::runtime_error("unsupported torch tensor type");
    }
}

at::Tensor attention_mlx(const at::Tensor& q_tensor,
                         const at::Tensor& k_tensor,
                         const at::Tensor& v_tensor,
                         const std::optional<at::Tensor>& attention_mask_tensor) {
    torch::mps::synchronize();
    
    TORCH_CHECK(q_tensor.device().is_mps(), "q_tensor must be on MPS");
    TORCH_CHECK(q_tensor.is_contiguous(), "q_tensor must be contiguous");
    
    TORCH_CHECK(k_tensor.device().is_mps(), "k_tensor must be on MPS");
    TORCH_CHECK(k_tensor.is_contiguous(), "k_tensor must be contiguous");
    
    TORCH_CHECK(v_tensor.device().is_mps(), "v_tensor must be on MPS");
    TORCH_CHECK(v_tensor.is_contiguous(), "v_tensor must be contiguous");
    
    mx::SmallVector<int> q_shape;
    for (auto s : q_tensor.sizes()) {
        q_shape.push_back(static_cast<int>(s));
    }
    
    mx::SmallVector<int> kv_shape;
    for (auto s : k_tensor.sizes()) {
        kv_shape.push_back(static_cast<int>(s));
    }
    
    // mlx Buffers wrap the torch tensors' MTLBuffer*
    auto q_buffer = mx::allocator::Buffer{q_tensor.storage().mutable_data()};
    mx::array q_mlx(q_buffer, q_shape, torch_dtype_to_mlx_dtype(q_tensor.scalar_type()));
    
    auto k_buffer = mx::allocator::Buffer{k_tensor.storage().mutable_data()};
    mx::array k_mlx(k_buffer, kv_shape, torch_dtype_to_mlx_dtype(k_tensor.scalar_type()));
    
    auto v_buffer = mx::allocator::Buffer{v_tensor.storage().mutable_data()};
    mx::array v_mlx(v_buffer, kv_shape, torch_dtype_to_mlx_dtype(v_tensor.scalar_type()));
    
    float scale = 1.0f / sqrt(static_cast<float>(q_tensor.size(3)));
    
    auto mask_mlx_opt = std::optional<mx::array>();
    if(attention_mask_tensor.has_value()) {
        auto mask_tensor = *attention_mask_tensor;
        mx::SmallVector<int> mask_shape;
        for (auto s : mask_tensor.sizes()) {
            mask_shape.push_back(static_cast<int>(s));
        }
        
        auto mask_buffer = mx::allocator::Buffer{mask_tensor.storage().mutable_data()};
        mx::array mask_mlx(mask_buffer, mask_shape, torch_dtype_to_mlx_dtype(mask_tensor.scalar_type()));
        mask_mlx_opt = mask_mlx;
    }
    
    mx::array output_mlx = mx::fast::scaled_dot_product_attention(q_mlx, k_mlx, v_mlx, scale, "", mask_mlx_opt);
    output_mlx = mx::contiguous(output_mlx);
    mx::eval({output_mlx});
    
    auto output_mlx_dtype = output_mlx.dtype();
    at::ScalarType output_torch_dype;
    void* data = nullptr;
    if (output_mlx_dtype == mx::float32) {
        output_torch_dype = at::kFloat;
        data = output_mlx.data<float32_t>();
    } else if (output_mlx_dtype == mx::float16) {
        output_torch_dype = at::kHalf;
        data = output_mlx.data<float16_t>();
    } else if (output_mlx_dtype == mx::bfloat16) {
        output_torch_dype = at::kBFloat16;
        data = output_mlx.data<bfloat16_t>();
    } else if (output_mlx_dtype == mx::float64) {
        output_torch_dype = at::kDouble;
        data = output_mlx.data<float64_t>();
    } else {
        throw std::runtime_error("unsupported mlx array output type");
    }
    
    std::vector<int64_t> output_shape_torch(output_mlx.shape().begin(), output_mlx.shape().end());
    
    // pass ownership to torch
    auto tensor_options = torch::TensorOptions()
        .dtype(output_torch_dype)
        .device(torch::kCPU);
    at::Tensor out_torch = at::from_blob(data,
                                         output_shape_torch,
                                         [](void* pointer) mutable { },
                                         tensor_options);
    
    mx::synchronize();
    
    return out_torch.to(torch::kMPS);
}
