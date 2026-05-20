package activation

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

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

func dispatchActivationLane(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	apply func(float32) float32,
) {
	if count == 0 {
		return
	}

	switch format {
	case dtype.Float32:
		for index := 0; index < count; index++ {
			storeF32(dst, index, apply(loadF32(src, index)))
		}
	case dtype.Float16:
		for index := 0; index < count; index++ {
			storeF16(dst, index, apply(loadF16(src, index)))
		}
	case dtype.BFloat16:
		for index := 0; index < count; index++ {
			storeBF16(dst, index, apply(loadBF16(src, index)))
		}
	}
}

func loadParamFloat32(slopes unsafe.Pointer, slopeCount, index int) float32 {
	if slopeCount == 1 {
		return loadF32(slopes, 0)
	}

	return loadF32(slopes, index)
}

func dispatchGatedPacked(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
	kernel func(dst, packed *float32, batch, halfCount int),
	combine func(gate, up float32) float32,
) {
	if batch == 0 || halfCount == 0 {
		return
	}

	rowStride := halfCount * 2

	switch format {
	case dtype.Float32:
		kernel(
			(*float32)(dst),
			(*float32)(packed),
			batch,
			halfCount,
		)
	case dtype.Float16:
		for batchIndex := 0; batchIndex < batch; batchIndex++ {
			rowBase := batchIndex * rowStride
			dstBase := batchIndex * halfCount

			for index := 0; index < halfCount; index++ {
				gate := loadF16(packed, rowBase+index)
				up := loadF16(packed, rowBase+halfCount+index)
				storeF16(dst, dstBase+index, combine(gate, up))
			}
		}
	case dtype.BFloat16:
		for batchIndex := 0; batchIndex < batch; batchIndex++ {
			rowBase := batchIndex * rowStride
			dstBase := batchIndex * halfCount

			for index := 0; index < halfCount; index++ {
				gate := loadBF16(packed, rowBase+index)
				up := loadBF16(packed, rowBase+halfCount+index)
				storeBF16(dst, dstBase+index, combine(gate, up))
			}
		}
	}
}

func dispatchActivationLaneIndexed(
	dst, src, slopes unsafe.Pointer,
	count int,
	format dtype.DType,
	slopeCount int,
	apply func(value, slope float32) float32,
) {
	if count == 0 {
		return
	}

	switch format {
	case dtype.Float32:
		for index := 0; index < count; index++ {
			slope := loadParamFloat32(slopes, slopeCount, index)
			storeF32(dst, index, apply(loadF32(src, index), slope))
		}
	case dtype.Float16:
		for index := 0; index < count; index++ {
			slope := loadParamFloat32(slopes, slopeCount, index)
			storeF16(dst, index, apply(loadF16(src, index), slope))
		}
	case dtype.BFloat16:
		for index := 0; index < count; index++ {
			slope := loadParamFloat32(slopes, slopeCount, index)
			storeBF16(dst, index, apply(loadBF16(src, index), slope))
		}
	}
}

func dispatchGatedTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
	kernel func(dst, gate, up *float32, count int),
	combine func(gate, up float32) float32,
) {
	if count == 0 {
		return
	}

	switch format {
	case dtype.Float32:
		kernel(
			(*float32)(dst),
			(*float32)(gate),
			(*float32)(up),
			count,
		)
	case dtype.Float16:
		for index := 0; index < count; index++ {
			storeF16(dst, index, combine(loadF16(gate, index), loadF16(up, index)))
		}
	case dtype.BFloat16:
		for index := 0; index < count; index++ {
			storeBF16(dst, index, combine(loadBF16(gate, index), loadBF16(up, index)))
		}
	}
}
