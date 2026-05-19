// Package dot implements dense vector dot products for float32,
// bfloat16, float16, and int8.
//
// Public entry points take unsafe.Pointer buffers and a dtype.DType
// (for float32), matching pkg/backend/device/cpu/activation. Float32
// kernels follow the pick-at-init model: select_{arm64,amd64,generic}.go
// register ISA candidates; f32_kernels.go binds the picked implementation.
package dot
