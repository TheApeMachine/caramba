// Package pool implements 2-D max/average pooling and adaptive pooling
// for float32 tensors in NCHW layout.
//
// Public entry points take unsafe.Pointer buffers and a dtype.DType,
// matching pkg/backend/device/cpu/activation. The arm64 f32 path uses
// row-wise NEON drivers selected at init via select_{arm64,amd64,386,generic}.go.
package pool
