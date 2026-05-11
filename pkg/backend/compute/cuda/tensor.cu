#include <cuda_runtime.h>
#include "tensor.h"

extern "C" {

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

int cuda_tensor_upload_double(void* device, const double* host, int n) {
    size_t bytes = (size_t)n * sizeof(double);

    if (bytes == 0) {
        return 0;
    }

    if (!device || !host) {
        return -1;
    }

    return cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice) == cudaSuccess ? 0 : -1;
}

int cuda_tensor_download_double(const void* device, double* host, int n) {
    size_t bytes = (size_t)n * sizeof(double);

    if (bytes == 0) {
        return 0;
    }

    if (!device || !host) {
        return -1;
    }

    return cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -1;
}

int cuda_tensor_free(void* device) {
    if (!device) {
        return 0;
    }

    return cudaFree(device) == cudaSuccess ? 0 : -1;
}

}
