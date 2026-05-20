package convolution

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func float32View(pointer unsafe.Pointer, length int) []float32 {
	if length == 0 {
		return nil
	}

	return unsafe.Slice((*float32)(pointer), length)
}

func loadF32(pointer unsafe.Pointer, index int) float32 {
	return *(*float32)(unsafe.Add(pointer, uintptr(index)*4))
}

func storeF32(pointer unsafe.Pointer, index int, value float32) {
	*(*float32)(unsafe.Add(pointer, uintptr(index)*4)) = value
}

func loadF16(pointer unsafe.Pointer, index int) float32 {
	bits := *(*uint16)(unsafe.Add(pointer, uintptr(index)*2))
	return dtype.Frombits(bits).Float32()
}

func storeF16(pointer unsafe.Pointer, index int, value float32) {
	bits := dtype.Fromfloat32(value).Bits()
	*(*uint16)(unsafe.Add(pointer, uintptr(index)*2)) = bits
}

func loadBF16(pointer unsafe.Pointer, index int) float32 {
	bits := *(*uint16)(unsafe.Add(pointer, uintptr(index)*2))
	bf16 := dtype.BF16(bits)
	return (&bf16).Float32()
}

func storeBF16(pointer unsafe.Pointer, index int, value float32) {
	encoded := dtype.NewBfloat16FromFloat32(value)
	*(*uint16)(unsafe.Add(pointer, uintptr(index)*2)) = uint16(encoded)
}

func widenToF32Buffer(dst, src unsafe.Pointer, count int, format dtype.DType) {
	if count == 0 {
		return
	}

	switch format {
	case dtype.Float16:
		for index := 0; index < count; index++ {
			storeF32(dst, index, loadF16(src, index))
		}
	case dtype.BFloat16:
		for index := 0; index < count; index++ {
			storeF32(dst, index, loadBF16(src, index))
		}
	}
}

func narrowFromF32Buffer(dst, src unsafe.Pointer, count int, format dtype.DType) {
	if count == 0 {
		return
	}

	switch format {
	case dtype.Float16:
		for index := 0; index < count; index++ {
			storeF16(dst, index, loadF32(src, index))
		}
	case dtype.BFloat16:
		for index := 0; index < count; index++ {
			storeBF16(dst, index, loadF32(src, index))
		}
	}
}
