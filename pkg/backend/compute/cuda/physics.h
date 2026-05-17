#ifndef CUDA_PHYSICS_H
#define CUDA_PHYSICS_H

#ifdef __cplusplus
extern "C" {
#endif

// All double-precision physics kernels.
// src/dst are host pointers; device memory is managed internally.
// Returns 0 on success, -1 on CUDA error.

// 2nd-order central-difference Laplacian on a uniform 1D grid with
// periodic boundary conditions:
//   dst[i] = (src[(i-1+n)%n] + src[(i+1)%n] - 2*src[i]) * inv_h2
// n >= 1. For n == 1 the periodic wrap collapses and dst[0] == 0.
int cuda_laplacian_1d(const double* src, double* dst, int n, double inv_h2);

// 2nd-order central-difference Laplacian on a uniform 2D grid with
// periodic boundary conditions. Row-major layout, shape [H, W]:
//   dst[i,j] = (src[i,(j-1+W)%W] + src[i,(j+1)%W]
//             + src[(i-1+H)%H,j] + src[(i+1)%H,j]
//             - 4*src[i,j]) * inv_h2
int cuda_laplacian_2d(const double* src, double* dst, int H, int W, double inv_h2);

// 2nd-order central-difference Laplacian on a uniform 3D grid with
// periodic boundary conditions. Row-major layout, shape [D, H, W]:
//   dst[k,i,j] = (left + right + up + down + front + back - 6*center) * inv_h2
int cuda_laplacian_3d(const double* src, double* dst, int D, int H, int W, double inv_h2);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_PHYSICS_H */
