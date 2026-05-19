package math

import (
	"unsafe"
)

// Extremely fast BFloat16 -> Float32 cast
func bf16ToF32(b uint16) float32 {
	bits := uint32(b) << 16
	return *(*float32)(unsafe.Pointer(&bits))
}

// Extremely fast Float32 -> BFloat16 cast (with fast rounding)
func f32ToBf16(f float32) uint16 {
	bits := *(*uint32)(unsafe.Pointer(&f))
	// Add 0x7FFF for proper nearest-even rounding before shifting
	return uint16((bits + 0x7FFF) >> 16)
}
