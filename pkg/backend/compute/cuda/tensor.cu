#include <cuda_runtime.h>
#include <stdint.h>
#include "tensor.h"

extern "C" {

static int cuda_tensor_memcpy_async_and_sync(
    void* dst,
    const void* src,
    size_t bytes,
    cudaMemcpyKind kind)
{
    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, kind, 0);

    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return -1;
    }

    err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return -1;
    }

    return 0;
}

void* cuda_tensor_alloc(size_t bytes) {
    void* device = NULL;

    if (bytes == 0) {
        return NULL;
    }

    if (cudaMalloc(&device, bytes) != cudaSuccess) {
        return NULL;
    }

    return device;
}

int cuda_tensor_upload_double(void* device, const double* host, size_t n) {
    const size_t elem_size = sizeof(double);

    if (n > SIZE_MAX / elem_size) {
        return -1;
    }

    size_t bytes = n * elem_size;

    if (bytes == 0) {
        return 0;
    }

    if (!device || !host) {
        return -1;
    }

    return cuda_tensor_memcpy_async_and_sync(device, host, bytes, cudaMemcpyHostToDevice);
}

int cuda_tensor_download_double(const void* device, double* host, size_t n) {
    const size_t elem_size = sizeof(double);

    if (n > SIZE_MAX / elem_size) {
        return -1;
    }

    size_t bytes = n * elem_size;

    if (bytes == 0) {
        return 0;
    }

    if (!device || !host) {
        return -1;
    }

    return cuda_tensor_memcpy_async_and_sync((void*)host, device, bytes, cudaMemcpyDeviceToHost);
}

int cuda_tensor_free(void* device) {
    if (!device) {
        return 0;
    }

    return cudaFree(device) == cudaSuccess ? 0 : -1;
}

}
