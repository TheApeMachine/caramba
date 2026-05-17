#ifndef METAL_TENSOR_H
#define METAL_TENSOR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Error codes for Metal tensor C APIs (metal_tensor_*).
 * Success is always METAL_TENSOR_OK (0). Negative values indicate failure.
 */
#define METAL_TENSOR_OK             0
#define METAL_TENSOR_ERR_INIT      -1
#define METAL_TENSOR_ERR_NULL_PTR  -2
#define METAL_TENSOR_ERR_OVERFLOW  -3
#define METAL_TENSOR_ERR_BOUNDS    -4

/*
 * metal_tensor_init -- ensures a default Metal device is available for tensor allocation.
 *
 * Return values:
 *   METAL_TENSOR_OK -- already initialized or device created successfully.
 *   METAL_TENSOR_ERR_INIT -- no Metal device (or runtime failure).
 *
 * Idempotent: safe to call multiple times; first successful initialization is retained.
 * Thread-safety: safe to call concurrently; callers should treat failure as permanent
 * for the process until hardware/runtime conditions change.
 * Ordering: no prior Metal tensor API is required; other Metal subsystems may initialize separately.
 */
int metal_tensor_init(void);

/*
 * metal_tensor_empty_float32 -- allocates a shared-storage float32 buffer of n elements.
 *
 * Parameters:
 *   n -- element count; must be > 0. Returns NULL if n == 0 or on overflow/allocation failure.
 *
 * Returns:
 *   Non-NULL owned handle on success (caller must pass exactly once to metal_tensor_free).
 *   NULL on invalid n, SIZE_MAX overflow, allocation failure, or failed metal_tensor_init.
 *
 * Ownership: returned pointer is an MTLBuffer cast to void*; release via metal_tensor_free only.
 */
void* metal_tensor_empty_float32(size_t n);

/*
 * metal_tensor_empty_float32_mode -- allocates a float32 buffer with explicit Metal storage mode.
 *
 * storage_mode values:
 *   0 -- MTLResourceStorageModeShared
 *   1 -- MTLResourceStorageModePrivate
 */
void* metal_tensor_empty_float32_mode(size_t n, int storage_mode);

/*
 * metal_tensor_upload_float32 -- copies host float data into a new device buffer.
 *
 * Parameters:
 *   src -- host pointer; must be non-NULL when n > 0.
 *   n -- element count; must be > 0 (returns NULL if n == 0).
 *
 * Returns NULL on NULL src (when n > 0), overflow, allocation failure, or init failure.
 * Ownership: same as metal_tensor_empty_float32.
 */
void* metal_tensor_upload_float32(const float* src, size_t n);

/*
 * metal_tensor_upload_float32_mode -- uploads host float data to explicit Metal storage mode.
 *
 * Private storage uses a shared staging buffer and a blit pass. The returned handle is still an
 * owned MTLBuffer released through metal_tensor_free.
 */
void* metal_tensor_upload_float32_mode(const float* src, size_t n, int storage_mode);

/*
 * metal_tensor_download_float32 -- copies up to n floats from handle into dst.
 *
 * Requires handle and dst non-NULL when n > 0.
 * Verifies n * sizeof(float) <= MTLBuffer length to prevent overrun.
 *
 * Returns METAL_TENSOR_OK on success, METAL_TENSOR_ERR_NULL_PTR, METAL_TENSOR_ERR_OVERFLOW,
 * or METAL_TENSOR_ERR_BOUNDS as appropriate. n == 0 is a no-op success.
 */
int metal_tensor_download_float32(const void* handle, float* dst, size_t n);

/*
 * metal_tensor_free -- releases buffer ownership obtained from empty/upload.
 *
 * NULL handle: no-op, returns METAL_TENSOR_OK.
 * Double-free: undefined behavior — each non-NULL handle must be freed at most once.
 * Thread-safety: not synchronized; callers must not free the same handle concurrently
 * or concurrently with GPU use of the buffer unless externally synchronized.
 *
 * Returns METAL_TENSOR_OK on success; future implementations may return non-zero on failure.
 */
int metal_tensor_free(void* handle);
void* metal_tensor_retain(const void* handle);

/*
 * metal_tensor_get_size -- byte length of the buffer backing handle.
 *
 * Returns 0 if handle is NULL. Does not validate handle type; invalid pointers yield undefined behavior.
 */
size_t metal_tensor_get_size(const void* handle);

/*
 * metal_tensor_get_storage_mode -- returns the MTLResourceStorageMode integer for handle.
 *
 * Returns -1 when handle is NULL.
 */
int metal_tensor_get_storage_mode(const void* handle);

#ifdef __cplusplus
}
#endif

#endif /* METAL_TENSOR_H */
