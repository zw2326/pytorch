#pragma once

#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#if !defined(__MAC_15_0) && \
    (!defined(MAC_OS_X_VERSION_15_0) || (MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_15_0))

@interface MPSNDArrayIdentity : MPSNDArrayUnaryKernel
-(MPSNDArray * __nullable) reshapeWithCommandBuffer: (__nullable id <MTLCommandBuffer>) cmdBuf
                                        sourceArray: (MPSNDArray * __nonnull) sourceArray
                                              shape: (MPSShape * __nonnull) shape
                                   destinationArray: (MPSNDArray * __nullable) destinationArray;
@end

@interface MPSNDArrayDescriptor()
@property (readwrite, nonatomic) BOOL preferPackedRows;
@end

@interface MPSNDArray()
-(nonnull instancetype) initWithBuffer:(id<MTLBuffer> _Nonnull) buffer
                                offset:(NSUInteger) offset
                            descriptor:(MPSNDArrayDescriptor * _Nonnull) descriptor;
-(MPSNDArray * __nullable) arrayViewWithShape:(MPSShape * _Nullable) shape
                                      strides:(MPSShape * _Nonnull)  strides;
@end

@interface MPSNDArrayQuantizedMatrixMultiplication : MPSNDArrayMatrixMultiplication
- (nonnull instancetype) initWithDevice: (nonnull id<MTLDevice>) device
             leftQuantizationDescriptor: (MPSNDArrayQuantizationDescriptor* _Nullable) leftQuantizationDescriptor
            rightQuantizationDescriptor: (MPSNDArrayQuantizationDescriptor* _Nullable) rightQuantizationDescriptor;
@end

@interface MPSNDArrayAffineQuantizationDescriptor()
- (nonnull instancetype) initWithDataType: (MPSDataType) quantizationDataType
                             hasZeroPoint: (BOOL) hasZeroPoint
                              hasMinValue: (BOOL) hasMinValue;
@end

#endif
