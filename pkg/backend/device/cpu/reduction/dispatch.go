// Package reduction implements dense reduction operations for float32,
// bfloat16, and float16.
//
// Public entry points take unsafe.Pointer buffers and a dtype.DType,
// matching pkg/backend/device/cpu/activation. Float32 kernels follow
// the pick-at-init model: select_{arm64,amd64,386,generic}.go register
// ISA candidates; f32_kernels.go binds the picked implementation.
package reduction
