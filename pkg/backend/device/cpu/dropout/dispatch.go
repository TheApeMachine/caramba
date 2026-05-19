// Package dropout implements inverted dropout for float32 tensors.
//
// Public entry points take unsafe.Pointer buffers and a dtype.DType,
// matching pkg/backend/device/cpu/activation. Float32 kernels follow
// the pick-at-init model via select_{arm64,amd64,386,generic}.go.
package dropout
