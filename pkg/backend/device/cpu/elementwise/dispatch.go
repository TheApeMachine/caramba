// Package elementwise implements dense element-wise binary and unary
// operations for float64, float32, float16, and bfloat16.
//
// Public entry points take unsafe.Pointer buffers and a dtype.DType,
// matching pkg/backend/device/cpu/activation. Float32 kernels follow
// the pick-at-init model: select_{arm64,amd64,386,generic}.go register
// ISA candidates; f32_kernels.go binds the picked implementation.
// Float16 and bfloat16 f32-backed paths widen in compute.go, execute
// the f32 kernel, then narrow on write-back.
package elementwise
