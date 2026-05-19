package losses

import (
	"math"
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func loadF32(pointer unsafe.Pointer, index int) float32 {
	return *(*float32)(unsafe.Add(pointer, uintptr(index)*4))
}

func loadF16(pointer unsafe.Pointer, index int) float32 {
	bits := *(*uint16)(unsafe.Add(pointer, uintptr(index)*2))
	return dtype.Frombits(bits).Float32()
}

func loadBF16(pointer unsafe.Pointer, index int) float32 {
	bits := *(*uint16)(unsafe.Add(pointer, uintptr(index)*2))
	bf16 := dtype.BF16(bits)
	return (&bf16).Float32()
}

func loadInt32(pointer unsafe.Pointer, index int) int32 {
	return *(*int32)(unsafe.Add(pointer, uintptr(index)*4))
}

func widenToFloat32Buffer(
	source unsafe.Pointer,
	count int,
	format dtype.DType,
) []float32 {
	buffer := BorrowFloat32Buffer(count)

	switch format {
	case dtype.Float32:
		sourceView := unsafe.Slice((*float32)(source), count)
		copy(buffer, sourceView)
	case dtype.Float16:
		for index := 0; index < count; index++ {
			buffer[index] = loadF16(source, index)
		}
	case dtype.BFloat16:
		for index := 0; index < count; index++ {
			buffer[index] = loadBF16(source, index)
		}
	default:
		panic("losses: unsupported dtype for widen")
	}

	return buffer
}

func dispatchPairLoss(
	predictions, targets unsafe.Pointer,
	count int,
	format dtype.DType,
	f32 func(predictions, targets unsafe.Pointer, count int) float32,
) float32 {
	if count == 0 {
		return 0
	}

	switch format {
	case dtype.Float32:
		return f32(predictions, targets, count)
	case dtype.Float16, dtype.BFloat16:
		predBuffer := widenToFloat32Buffer(predictions, count, format)
		targetBuffer := widenToFloat32Buffer(targets, count, format)

		defer ReleaseFloat32Buffer(predBuffer)
		defer ReleaseFloat32Buffer(targetBuffer)

		return f32(
			unsafe.Pointer(&predBuffer[0]),
			unsafe.Pointer(&targetBuffer[0]),
			count,
		)
	default:
		panic("losses: unsupported dtype")
	}
}

func dispatchCrossEntropy(
	logits unsafe.Pointer,
	targets unsafe.Pointer,
	batchSize, classes int,
	format dtype.DType,
) float32 {
	if batchSize == 0 || classes == 0 {
		return 0
	}

	logitCount := batchSize * classes

	switch format {
	case dtype.Float32:
		return crossEntropyF32(logits, targets, batchSize, classes)
	case dtype.Float16, dtype.BFloat16:
		logitBuffer := widenToFloat32Buffer(logits, logitCount, format)

		defer ReleaseFloat32Buffer(logitBuffer)

		return crossEntropyF32(
			unsafe.Pointer(&logitBuffer[0]),
			targets,
			batchSize,
			classes,
		)
	default:
		panic("losses: unsupported dtype")
	}
}

func crossEntropyF32(
	logits unsafe.Pointer,
	targets unsafe.Pointer,
	batchSize, classes int,
) float32 {
	var sum float64

	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		rowBase := batchIndex * classes
		maxLogit := loadF32(logits, rowBase)

		for classIndex := 1; classIndex < classes; classIndex++ {
			candidate := loadF32(logits, rowBase+classIndex)

			if candidate > maxLogit {
				maxLogit = candidate
			}
		}

		var denominator float64

		for classIndex := 0; classIndex < classes; classIndex++ {
			candidate := loadF32(logits, rowBase+classIndex)
			denominator += math.Exp(float64(candidate - maxLogit))
		}

		targetClass := int(loadInt32(targets, batchIndex))

		if targetClass < 0 || targetClass >= classes {
			panic("losses: cross entropy target out of range")
		}

		logProb := float64(loadF32(logits, rowBase+targetClass)-maxLogit) - math.Log(denominator)
		sum += -logProb
	}

	return float32(sum / float64(batchSize))
}

func huberMeanF32(predictions, targets unsafe.Pointer, count int) float32 {
	const delta = float32(1.0)
	var sum float64

	for index := 0; index < count; index++ {
		diff := loadF32(predictions, index) - loadF32(targets, index)
		absDiff := float32(math.Abs(float64(diff)))

		switch {
		case absDiff <= delta:
			sum += 0.5 * float64(diff) * float64(diff)
		default:
			sum += float64(delta) * (float64(absDiff) - 0.5*float64(delta))
		}
	}

	return float32(sum / float64(count))
}

func binaryCrossEntropyMeanF32(predictions, targets unsafe.Pointer, count int) float32 {
	var sum float64
	const eps = 1e-7

	for index := 0; index < count; index++ {
		value := loadF32(predictions, index)
		target := loadF32(targets, index)
		clamped := math.Max(eps, math.Min(1-eps, float64(value)))
		sum += -float64(target)*math.Log(clamped) -
			(1-float64(target))*math.Log(1-clamped)
	}

	return float32(sum / float64(count))
}

func klDivergenceMeanF32(predictions, targets unsafe.Pointer, count int) float32 {
	var sum float64
	const eps = 1e-12

	for index := 0; index < count; index++ {
		predicted := math.Max(eps, float64(loadF32(predictions, index)))
		target := math.Max(eps, float64(loadF32(targets, index)))
		sum += target * math.Log(target/predicted)
	}

	return float32(sum / float64(count))
}
