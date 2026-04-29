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

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

TORCH_LIBRARY(custom_ops, m) {
    m.def("attention_mps(Tensor q, Tensor k, Tensor v, Tensor? attention_mask) -> Tensor");
    m.impl("attention_mps", c10::DispatchKey::MPS, TORCH_FN(attention_mps));
}

at::Tensor attention_mps(const at::Tensor& q_tensor,
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
    auto sequence_length = q_tensor.size(2);
    auto head_dimension = q_tensor.size(3);
    
    at::Tensor output_tensor = at::empty({batch_size, head_count, sequence_length, head_dimension}, q_tensor.options());
    
    @autoreleasepool {
        static NSMutableDictionary<NSString *, MPSGraphExecutable *> *mpsGraphExecutableCache = nil;
        static dispatch_once_t onceToken;
        dispatch_once(&onceToken, ^{
            mpsGraphExecutableCache = [[NSMutableDictionary alloc] init];
        });
        
        id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
        
        dispatch_sync(torch::mps::get_dispatch_queue(), ^(){
            std::string mps_data_type_string = at::native::mps::getMPSTypeString(q_tensor.scalar_type());
            NSString *mpsGraphExecutableKey = [NSString stringWithFormat:@"%lld_%lld_%lld_%lld_%d_%s", batch_size, head_count, sequence_length, head_dimension, attention_mask_tensor.has_value(), mps_data_type_string.c_str()];
            MPSGraphExecutable *mpsGraphExecutable = mpsGraphExecutableCache[mpsGraphExecutableKey];
            NSArray<NSNumber*> *shape = @[@(batch_size), @(head_count), @(sequence_length), @(head_dimension)];
            
            if (!mpsGraphExecutable) {
                MPSGraphDevice *graphDevice = [MPSGraphDevice deviceWithMTLDevice:commandBuffer.device];
                MPSGraph *graph = [[MPSGraph alloc] init];
                
                MPSGraphTensor *mpsQ = [graph placeholderWithShape:shape dataType:mpsDataType name:@"mpsQ"];
                MPSGraphTensor *mpsK = [graph placeholderWithShape:shape dataType:mpsDataType name:@"mpsK"];
                MPSGraphTensor *mpsV = [graph placeholderWithShape:shape dataType:mpsDataType name:@"mpsV"];
                MPSGraphTensor *mpsAttentionMask = nil;
                if(attention_mask_tensor.has_value()) {
                    mpsAttentionMask = [graph placeholderWithShape:shape dataType:mpsDataTypeAttentionMask name:@"mpsAttentionMask"];
                }
                
                float scale = 1.0f / sqrt(static_cast<float>(head_dimension));
                
                MPSGraphTensor *result = [graph scaledDotProductAttentionWithQueryTensor:mpsQ
                                                                               keyTensor:mpsK
                                                                             valueTensor:mpsV
                                                                              maskTensor:mpsAttentionMask
                                                                                   scale:scale
                                                                                    name:mpsGraphExecutableKey];
                
                NSMutableDictionary *feeds = [NSMutableDictionary dictionaryWithCapacity:4];
                feeds[mpsQ] = [[MPSGraphShapedType alloc] initWithShape:shape dataType:mpsDataType];
                feeds[mpsK] = [[MPSGraphShapedType alloc] initWithShape:shape dataType:mpsDataType];
                feeds[mpsV] = [[MPSGraphShapedType alloc] initWithShape:shape dataType:mpsDataType];
                if(attention_mask_tensor.has_value()) {
                    feeds[mpsAttentionMask] = [[MPSGraphShapedType alloc] initWithShape:shape dataType:mpsDataTypeAttentionMask];
                }
                
                MPSGraphCompilationDescriptor *compilationDescriptor = [[MPSGraphCompilationDescriptor alloc] init];
                compilationDescriptor.optimizationLevel = MPSGraphOptimizationLevel1;
                compilationDescriptor.reducedPrecisionFastMath = MPSGraphReducedPrecisionFastMathAllowFP16Intermediates;
                [compilationDescriptor convertLayoutToNHWC];
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
                                    shape:shape
                                    dataType:mpsDataType]];
            [inputsArray addObject:[[MPSGraphTensorData alloc]
                                    initWithMTLBuffer:at::native::mps::getMTLBufferStorage(k_tensor)
                                    shape:shape
                                    dataType:mpsDataType]];
            [inputsArray addObject:[[MPSGraphTensorData alloc]
                                    initWithMTLBuffer:at::native::mps::getMTLBufferStorage(v_tensor)
                                    shape:shape
                                    dataType:mpsDataType]];
            if(attention_mask_tensor.has_value()) {
                [inputsArray addObject:[[MPSGraphTensorData alloc]
                                        initWithMTLBuffer:at::native::mps::getMTLBufferStorage(*attention_mask_tensor)
                                        shape:shape
                                        dataType:mpsDataTypeAttentionMask]];
            }
            
            MPSGraphTensorData *output = [[MPSGraphTensorData alloc]
                                          initWithMTLBuffer:at::native::mps::getMTLBufferStorage(output_tensor)
                                          shape:shape
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
