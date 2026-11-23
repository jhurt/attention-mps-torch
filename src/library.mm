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

        dispatch_sync(torch::mps::get_dispatch_queue(), ^(){
            @autoreleasepool {
                id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();

                std::string mps_data_type_string = at::native::mps::getMPSTypeString(q_tensor.scalar_type());
                NSString *mpsGraphExecutableKey = [NSString stringWithFormat:@"%lld_%lld_%lld_%lld_%lld_%d_%s", batch_size, head_count, sequence_length_q, sequence_length_kv, head_dimension, attention_mask_tensor.has_value(), mps_data_type_string.c_str()];

                NSArray<NSNumber*> *qShape = @[@(batch_size), @(head_count), @(sequence_length_q), @(head_dimension)];
                NSArray<NSNumber*> *kvShape = @[@(batch_size), @(head_count), @(sequence_length_kv), @(head_dimension)];
                NSArray<NSNumber*> *attentionMaskShape = nil;

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
                    MPSGraph *graph = [[[MPSGraph alloc] init] autorelease];

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
                    feeds[mpsQ] = [[[MPSGraphShapedType alloc] initWithShape:qShape dataType:mpsDataType] autorelease];
                    feeds[mpsK] = [[[MPSGraphShapedType alloc] initWithShape:kvShape dataType:mpsDataType] autorelease];
                    feeds[mpsV] = [[[MPSGraphShapedType alloc] initWithShape:kvShape dataType:mpsDataType] autorelease];
                    if(attention_mask_tensor.has_value()) {
                        feeds[mpsAttentionMask] = [[[MPSGraphShapedType alloc] initWithShape:attentionMaskShape dataType:mpsDataTypeAttentionMask] autorelease];
                    }

                    MPSGraphCompilationDescriptor *compilationDescriptor = [[[MPSGraphCompilationDescriptor alloc] init] autorelease];
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
                [inputsArray addObject:[[[MPSGraphTensorData alloc]
                                         initWithMTLBuffer:at::native::mps::getMTLBufferStorage(q_tensor)
                                         shape:qShape
                                         dataType:mpsDataType] autorelease]];
                [inputsArray addObject:[[[MPSGraphTensorData alloc]
                                         initWithMTLBuffer:at::native::mps::getMTLBufferStorage(k_tensor)
                                         shape:kvShape
                                         dataType:mpsDataType] autorelease]];
                [inputsArray addObject:[[[MPSGraphTensorData alloc]
                                         initWithMTLBuffer:at::native::mps::getMTLBufferStorage(v_tensor)
                                         shape:kvShape
                                         dataType:mpsDataType] autorelease]];
                if(attention_mask_tensor.has_value()) {
                    [inputsArray addObject:[[[MPSGraphTensorData alloc]
                                             initWithMTLBuffer:at::native::mps::getMTLBufferStorage(*attention_mask_tensor)
                                             shape:attentionMaskShape
                                             dataType:mpsDataTypeAttentionMask] autorelease]];
                }

                MPSGraphTensorData *output = [[[MPSGraphTensorData alloc]
                                               initWithMTLBuffer:at::native::mps::getMTLBufferStorage(output_tensor)
                                               shape:qShape
                                               dataType:mpsDataType] autorelease];

                [mpsGraphExecutable encodeToCommandBuffer:commandBuffer
                                              inputsArray:inputsArray
                                             resultsArray:@[output]
                                      executionDescriptor:nil];

                torch::mps::commit();
            }
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

mx::array torch_mps_tensor_to_mlx_array(const at::Tensor& tensor) {
    TORCH_CHECK(tensor.device().is_mps(), "tensor must be on MPS");
    TORCH_CHECK(tensor.is_contiguous(), "tensor must be contiguous");
    TORCH_CHECK(tensor.storage_offset() == 0, "tensor must have zero storage offset");

    mx::SmallVector<int> shape;
    for (auto s : tensor.sizes()) {
        shape.push_back(static_cast<int>(s));
    }

    id<MTLBuffer> mtl_buffer = at::native::mps::getMTLBufferStorage(tensor);
    mx::allocator::Buffer buffer{(__bridge void*)mtl_buffer};
    mx::allocator::Data data{buffer, [](mx::allocator::Buffer) {}};

    return mx::array(std::move(data), std::move(shape), torch_dtype_to_mlx_dtype(tensor.scalar_type()));
}

at::Tensor attention_mlx(const at::Tensor& q_tensor,
                         const at::Tensor& k_tensor,
                         const at::Tensor& v_tensor,
                         const std::optional<at::Tensor>& attention_mask_tensor) {
    torch::mps::synchronize();

    mx::array q_mlx = torch_mps_tensor_to_mlx_array(q_tensor);
    mx::array k_mlx = torch_mps_tensor_to_mlx_array(k_tensor);
    mx::array v_mlx = torch_mps_tensor_to_mlx_array(v_tensor);

    float scale = 1.0f / sqrt(static_cast<float>(q_tensor.size(3)));

    auto mask_mlx_opt = std::optional<mx::array>();
    if (attention_mask_tensor.has_value()) {
        mask_mlx_opt = torch_mps_tensor_to_mlx_array(*attention_mask_tensor);
    }

    mx::array output_mlx = mx::fast::scaled_dot_product_attention(q_mlx, k_mlx, v_mlx, scale, -1, "", mask_mlx_opt);
    output_mlx = mx::contiguous(output_mlx);
    mx::eval({output_mlx});
    mx::synchronize();

    std::vector<int64_t> output_shape_torch(output_mlx.shape().begin(), output_mlx.shape().end());
    at::Tensor output_tensor = at::empty(output_shape_torch, q_tensor.options());

    id<MTLBuffer> sourceBuffer = (__bridge id<MTLBuffer>)output_mlx.buffer().ptr();
    id<MTLBuffer> destinationBuffer = at::native::mps::getMTLBufferStorage(output_tensor);
    size_t sourceOffset = output_mlx.offset();
    size_t byteCount = output_mlx.nbytes();

    @autoreleasepool {
        dispatch_sync(torch::mps::get_dispatch_queue(), ^() {
            id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
            id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
            [blitEncoder copyFromBuffer:sourceBuffer
                           sourceOffset:sourceOffset
                               toBuffer:destinationBuffer
                      destinationOffset:0
                                   size:byteCount];
            [blitEncoder endEncoding];
            torch::mps::commit();
        });
    }

    torch::mps::synchronize();

    return output_tensor;
}
