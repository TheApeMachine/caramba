#include "bridge_darwin_private.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include "_cgo_export.h"
#include <stdio.h>

static void metal_elementwise_status_clear(MetalStatus* status) {
    if (status == NULL) {
        return;
    }

    status->code = 0;
    status->message[0] = '\0';
}

static void metal_elementwise_status_set(MetalStatus* status, int code, const char* message) {
    if (status == NULL) {
        return;
    }

    status->code = code;

    if (message == NULL) {
        status->message[0] = '\0';
        return;
    }

    snprintf(status->message, METAL_STATUS_MESSAGE_BYTES, "%s", message);
}

static const char* metal_binary_operation_name(int operation) {
    switch (operation) {
    case MetalBinaryFloat32Add: return "add";
    case MetalBinaryFloat32Sub: return "sub";
    case MetalBinaryFloat32Mul: return "mul";
    case MetalBinaryFloat32Div: return "div";
    case MetalBinaryFloat32Max: return "max";
    case MetalBinaryFloat32Min: return "min";
    case MetalBinaryFloat32Eq: return "eq";
    case MetalBinaryFloat32Ne: return "ne";
    case MetalBinaryFloat32Lt: return "lt";
    case MetalBinaryFloat32Le: return "le";
    case MetalBinaryFloat32Gt: return "gt";
    case MetalBinaryFloat32Ge: return "ge";
    case MetalBinaryFloat32Pow: return "pow";
    case MetalBinaryFloat32Atan2: return "atan2";
    case MetalBinaryFloat32Mod: return "mod";
    default: return NULL;
    }
}

static const char* metal_unary_operation_name(int operation) {
    switch (operation) {
    case MetalUnaryFloat32Relu: return "relu";
    case MetalUnaryFloat32Abs: return "abs";
    case MetalUnaryFloat32Neg: return "neg";
    case MetalUnaryFloat32Square: return "square";
    case MetalUnaryFloat32Recip: return "recip";
    case MetalUnaryFloat32Sqrt: return "sqrt";
    case MetalUnaryFloat32Sign: return "sign";
    case MetalUnaryFloat32Rsqrt: return "rsqrt";
    case MetalUnaryFloat32Exp: return "exp";
    case MetalUnaryFloat32Log: return "log";
    case MetalUnaryFloat32Sin: return "sin";
    case MetalUnaryFloat32Cos: return "cos";
    case MetalUnaryFloat32Tanh: return "tanh";
    case MetalUnaryFloat32Gelu: return "gelu";
    case MetalUnaryFloat32Sigmoid: return "sigmoid";
    case MetalUnaryFloat32Silu: return "silu";
    case MetalUnaryFloat32Swish: return "swish";
    case MetalUnaryFloat32Softsign: return "softsign";
    case MetalUnaryFloat32ELU: return "elu";
    case MetalUnaryFloat32SELU: return "selu";
    case MetalUnaryFloat32LeakyReLU: return "leaky_relu";
    case MetalUnaryFloat32HardSigmoid: return "hardsigmoid";
    case MetalUnaryFloat32HardSwish: return "hardswish";
    default: return NULL;
    }
}

static const char* metal_element_dtype_suffix(int elementDType) {
    switch (elementDType) {
    case MetalElementDTypeFloat32: return "float32";
    case MetalElementDTypeFloat16: return "float16";
    case MetalElementDTypeBFloat16: return "bfloat16";
    default: return NULL;
    }
}

static int metal_elementwise_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    const char* dtypeSuffix,
    MetalStatus* status
) {
    if (operationName == NULL || dtypeSuffix == NULL) {
        metal_elementwise_status_set(status, -6, "unknown Metal elementwise kernel");
        return -6;
    }

    int written = snprintf(out, outBytes, "%s_%s", operationName, dtypeSuffix);

    if (written <= 0 || (size_t)written >= outBytes) {
        metal_elementwise_status_set(status, -6, "Metal elementwise kernel name overflow");
        return -6;
    }

    return 0;
}

static void metal_elementwise_complete(
    uint64_t completionToken,
    id<MTLCommandBuffer> completedBuffer
) {
    @autoreleasepool {
        if ([completedBuffer status] == MTLCommandBufferStatusCompleted) {
            metalCommandCompleted(completionToken, 0, "");
            return;
        }

        NSError* error = [completedBuffer error];
        NSString* message = @"Metal command buffer failed";

        if (error != nil) {
            message = [NSString
                stringWithFormat:@"%@: %@",
                message,
                [error localizedDescription]
            ];
        }

        metalCommandCompleted(completionToken, -5, (char*)[message UTF8String]);
    }
}

static int metal_elementwise_prepare(
    MetalContext* context,
    const char* kernelName,
    MetalStatus* status,
    id<MTLCommandQueue>* queue,
    id<MTLComputePipelineState>* pipeline
) {
    if (context == NULL || context->queue == NULL) {
        metal_elementwise_status_set(status, -1, "invalid Metal context");
        return -1;
    }

    *queue = (__bridge id<MTLCommandQueue>)context->queue;
    *pipeline = metal_get_pipeline(context, kernelName, status);

    if (*pipeline == nil) {
        return status != NULL && status->code != 0 ? status->code : -7;
    }

    return 0;
}

int metal_dispatch_binary_elementwise(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_elementwise_status_clear(status);

        if (count == 0) {
            return 0;
        }

        if (leftRef == NULL || rightRef == NULL || outRef == NULL) {
            metal_elementwise_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char kernelName[128];
        int nameCode = metal_elementwise_kernel_name(
            kernelName,
            sizeof(kernelName),
            metal_binary_operation_name(operation),
            metal_element_dtype_suffix(elementDType),
            status
        );

        if (nameCode != 0) {
            return nameCode;
        }

        id<MTLCommandQueue> queue = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_elementwise_prepare(
            (MetalContext*)contextRef,
            kernelName,
            status,
            &queue,
            &pipeline
        );

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

        if (commandBuffer == nil) {
            metal_elementwise_status_set(status, -3, "commandBuffer returned nil");
            return -3;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        if (encoder == nil) {
            metal_elementwise_status_set(status, -4, "computeCommandEncoder returned nil");
            return -4;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)leftRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)rightRef offset:0 atIndex:1];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:2];
        [encoder setBytes:&count length:sizeof(count) atIndex:3];

        NSUInteger threadWidth = [pipeline threadExecutionWidth];

        if (threadWidth == 0) {
            threadWidth = 1;
        }

        NSUInteger vectorCount = (NSUInteger)((count + 3) / 4);
        [encoder
            dispatchThreads:MTLSizeMake(vectorCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)
        ];
        [encoder endEncoding];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_elementwise_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}

int metal_dispatch_unary_elementwise(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
) {
    @autoreleasepool {
        metal_elementwise_status_clear(status);

        if (count == 0) {
            return 0;
        }

        if (inputRef == NULL || outRef == NULL) {
            metal_elementwise_status_set(status, -2, "nil Metal buffer");
            return -2;
        }

        char kernelName[128];
        int nameCode = metal_elementwise_kernel_name(
            kernelName,
            sizeof(kernelName),
            metal_unary_operation_name(operation),
            metal_element_dtype_suffix(elementDType),
            status
        );

        if (nameCode != 0) {
            return nameCode;
        }

        id<MTLCommandQueue> queue = nil;
        id<MTLComputePipelineState> pipeline = nil;
        int prepareCode = metal_elementwise_prepare(
            (MetalContext*)contextRef,
            kernelName,
            status,
            &queue,
            &pipeline
        );

        if (prepareCode != 0) {
            return prepareCode;
        }

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

        if (commandBuffer == nil) {
            metal_elementwise_status_set(status, -3, "commandBuffer returned nil");
            return -3;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        if (encoder == nil) {
            metal_elementwise_status_set(status, -4, "computeCommandEncoder returned nil");
            return -4;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:(__bridge id<MTLBuffer>)inputRef offset:0 atIndex:0];
        [encoder setBuffer:(__bridge id<MTLBuffer>)outRef offset:0 atIndex:1];
        [encoder setBytes:&count length:sizeof(count) atIndex:2];

        NSUInteger threadWidth = [pipeline threadExecutionWidth];

        if (threadWidth == 0) {
            threadWidth = 1;
        }

        NSUInteger vectorCount = (NSUInteger)((count + 3) / 4);
        [encoder
            dispatchThreads:MTLSizeMake(vectorCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)
        ];
        [encoder endEncoding];
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
            metal_elementwise_complete(completionToken, completedBuffer);
        }];
        [commandBuffer commit];

        return 0;
    }
}
