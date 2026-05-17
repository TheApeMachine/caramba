#ifndef METAL_PHYSICS_H
#define METAL_PHYSICS_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize Metal pipelines for physics-stencil kernels.
// metallib_path: path to compiled physics.metallib.
// Returns 0 on success, -1 on failure.
int metal_physics_init(const char* metallib_path);

// 2nd-order central-difference Laplacian on a uniform 1D grid with
// periodic boundary conditions:
//   dst[i] = (src[(i-1+n)%n] + src[(i+1)%n] - 2*src[i]) * inv_h2
// src/dst are host float arrays of length n.
int metal_laplacian_1d(
    const float* src,
    float*       dst,
    int          n,
    float        inv_h2);

// Same operation against resident Metal buffers (zero-copy chained-op path).
int metal_laplacian_1d_tensor(
    const void*  src_buffer,
    void*        dst_buffer,
    int          n,
    float        inv_h2);

// 2nd-order central-difference Laplacian on a uniform 2D grid with
// periodic boundary conditions. Row-major [H, W]:
//   dst[i,j] = (src[i,(j-1+W)%W] + src[i,(j+1)%W]
//             + src[(i-1+H)%H,j] + src[(i+1)%H,j] - 4*src[i,j]) * inv_h2
int metal_laplacian_2d(
    const float* src,
    float*       dst,
    int          H,
    int          W,
    float        inv_h2);

int metal_laplacian_2d_tensor(
    const void*  src_buffer,
    void*        dst_buffer,
    int          H,
    int          W,
    float        inv_h2);

// 2nd-order central-difference Laplacian on a uniform 3D grid with
// periodic boundary conditions. Row-major [D, H, W]:
//   dst[k,i,j] = (left + right + up + down + front + back - 6*center) * inv_h2
int metal_laplacian_3d(
    const float* src,
    float*       dst,
    int          D,
    int          H,
    int          W,
    float        inv_h2);

int metal_laplacian_3d_tensor(
    const void*  src_buffer,
    void*        dst_buffer,
    int          D,
    int          H,
    int          W,
    float        inv_h2);

#ifdef __cplusplus
}
#endif

#endif /* METAL_PHYSICS_H */
