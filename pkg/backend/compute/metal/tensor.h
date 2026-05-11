#ifndef METAL_TENSOR_H
#define METAL_TENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

int metal_tensor_init(void);
void* metal_tensor_empty_float32(int n);
void* metal_tensor_upload_float32(const float* src, int n);
int metal_tensor_download_float32(const void* handle, float* dst, int n);
int metal_tensor_free(void* handle);

#ifdef __cplusplus
}
#endif

#endif /* METAL_TENSOR_H */
