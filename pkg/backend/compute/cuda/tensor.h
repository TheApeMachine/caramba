#ifndef PKG_BACKEND_COMPUTE_CUDA_TENSOR_H
#define PKG_BACKEND_COMPUTE_CUDA_TENSOR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
cuda_tensor_alloc -- allocates bytes on the current CUDA device.

Returns NULL when bytes==0 or cudaMalloc fails (allocation failure).
On success the pointer is owned by the caller and must be released exactly once
with cuda_tensor_free. Not synchronized across host threads beyond CUDA runtime
defaults (follow ordinary CUDA device-context rules).

Thread-safety: matches cudaMalloc for the active device context.
*/
void* cuda_tensor_alloc(size_t bytes);

/*
cuda_tensor_upload_double / cuda_tensor_download_double -- host <-> device transfers.

Parameters:
  device -- CUDA device pointer (cuda_tensor_alloc or compatible).
  host   -- pinned or pageable host pointer with room for n doubles.
  n      -- number of double elements. n==0 is a no-op success.
           If n * sizeof(double) would overflow size_t, returns -1.

Return convention:
  0 on success, -1 on invalid arguments or CUDA API failure (overflowing element
  count, NULL pointers where data transfer required, memcpy/sync failure).

Ownership: device memory is not freed by these routines. They perform an async
copy on the default per-device stream (stream 0) followed by device-wide
synchronize so the transfer is complete before return; no partial writes are
visible on success.

They do not set errno; use only the integer return code.
*/
int cuda_tensor_upload_double(void* device, const double* host, size_t n);
int cuda_tensor_download_double(const void* device, double* host, size_t n);

/*
cuda_tensor_free -- frees device memory from cuda_tensor_alloc.

Returns 0 on success or when device is NULL; -1 if cudaFree fails.
Safe to call with NULL.
*/
int cuda_tensor_free(void* device);

#ifdef __cplusplus
}
#endif

#endif /* PKG_BACKEND_COMPUTE_CUDA_TENSOR_H */
