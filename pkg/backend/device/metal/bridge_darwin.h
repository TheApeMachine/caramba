#ifndef CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_DARWIN_H
#define CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_DARWIN_H

#include <stdint.h>

#define METAL_STATUS_MESSAGE_BYTES 1024

#ifdef __cplusplus
extern "C" {
#endif

typedef void* MetalDeviceRef;
typedef void* MetalBufferRef;

typedef struct MetalStatus {
    int code;
    char message[METAL_STATUS_MESSAGE_BYTES];
} MetalStatus;

typedef enum MetalBinaryFloat32Op {
    MetalBinaryFloat32Add = 0,
    MetalBinaryFloat32Sub = 1,
    MetalBinaryFloat32Mul = 2,
    MetalBinaryFloat32Div = 3,
    MetalBinaryFloat32Max = 4,
    MetalBinaryFloat32Min = 5,
    MetalBinaryFloat32Eq = 6,
    MetalBinaryFloat32Ne = 7,
    MetalBinaryFloat32Lt = 8,
    MetalBinaryFloat32Le = 9,
    MetalBinaryFloat32Gt = 10,
    MetalBinaryFloat32Ge = 11,
} MetalBinaryFloat32Op;

typedef enum MetalUnaryFloat32Op {
    MetalUnaryFloat32Relu = 0,
    MetalUnaryFloat32Abs = 1,
    MetalUnaryFloat32Neg = 2,
    MetalUnaryFloat32Square = 3,
    MetalUnaryFloat32Recip = 4,
    MetalUnaryFloat32Sqrt = 5,
    MetalUnaryFloat32Sign = 6,
} MetalUnaryFloat32Op;

typedef enum MetalElementDType {
    MetalElementDTypeFloat32 = 0,
    MetalElementDTypeFloat16 = 1,
    MetalElementDTypeBFloat16 = 2,
} MetalElementDType;

MetalDeviceRef metal_open_default_device(
    const uint8_t* libraryBytes,
    long long libraryLength,
    MetalStatus* status
);
long long metal_recommended_max_working_set(MetalDeviceRef contextRef);
MetalBufferRef metal_buffer_new_shared(MetalDeviceRef contextRef, long long bytes);
void metal_buffer_release(MetalBufferRef bufferRef);
void* metal_buffer_contents(MetalBufferRef bufferRef);
int metal_dispatch_binary_float32(
    MetalDeviceRef contextRef,
    int operation,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_unary_float32(
    MetalDeviceRef contextRef,
    int operation,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
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
);
int metal_dispatch_unary_elementwise(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_copy_bytes(
    MetalDeviceRef contextRef,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t byteCount,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_concat_bytes(
    MetalDeviceRef contextRef,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef outRef,
    uint32_t leftBytes,
    uint32_t rightBytes,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_split2_bytes(
    MetalDeviceRef contextRef,
    MetalBufferRef inputRef,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    uint32_t leftBytes,
    uint32_t rightBytes,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_last_token_bytes(
    MetalDeviceRef contextRef,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t seq,
    uint32_t hiddenBytes,
    uint32_t outBytes,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_transpose2d_bytes(
    MetalDeviceRef contextRef,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint32_t elementBytes,
    uint64_t completionToken,
    MetalStatus* status
);
int metal_dispatch_upsample_nearest2d_bytes(
    MetalDeviceRef contextRef,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t channels,
    uint32_t inHeight,
    uint32_t inWidth,
    uint32_t outHeight,
    uint32_t outWidth,
    uint32_t elementBytes,
    uint32_t outElements,
    uint64_t completionToken,
    MetalStatus* status
);
void metal_device_release(MetalDeviceRef contextRef);

#ifdef __cplusplus
}
#endif

#endif
