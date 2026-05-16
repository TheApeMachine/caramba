#ifndef METAL_MASKING_H
#define METAL_MASKING_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize Metal device, command queue, and masking pipelines.
// metallib_path: path to the compiled masking.metallib file.
// Returns 0 on success, -1 on failure.
int metal_masking_init(const char* metallib_path);

// Generate a causal (lower-triangular) attention mask.
// out: caller-allocated float array of seq_len*seq_len elements.
// out[row*seq_len+col] = 0.0 if col<=row else -FLT_MAX
int metal_causal_mask(float* out, int seq_len);

// Apply additive mask to attention scores: out[i] = scores[i] + mask[i]
// n: total number of elements
int metal_apply_mask(const float* scores, const float* mask, float* out, int n);

int metal_causal_mask_tensor(void* out, int seq_len);

int metal_apply_mask_tensor(const void* scores, const void* mask, void* out, int n);

#ifdef __cplusplus
}
#endif

#endif /* METAL_MASKING_H */
