// Package matmul implements dense and sparse matrix multiplication on CPU.
//
// Public entry points take unsafe.Pointer buffers and a dtype.DType,
// matching pkg/backend/device/cpu/activation. Float32/float64 kernels
// are selected at init via select_{arm64,amd64,386,generic}.go.
package matmul
