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
} MetalBinaryFloat32Op;

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
void metal_device_release(MetalDeviceRef contextRef);

#ifdef __cplusplus
}
#endif

#endif
