#ifndef CUDA_TENSOR_H
#define CUDA_TENSOR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void* cuda_tensor_alloc(size_t bytes);
int cuda_tensor_upload_double(void* device, const double* host, int n);
int cuda_tensor_download_double(const void* device, double* host, int n);
int cuda_tensor_free(void* device);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_TENSOR_H */
