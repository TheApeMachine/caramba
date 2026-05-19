package losses

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func MSE(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	return dispatchPairLoss(predictions, targets, count, format, runMSEF32)
}

func MAE(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	return dispatchPairLoss(predictions, targets, count, format, runMAEF32)
}

func Huber(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	return dispatchPairLoss(predictions, targets, count, format, runHuberF32)
}

func BinaryCrossEntropy(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	return dispatchPairLoss(predictions, targets, count, format, runBinaryCrossEntropyF32)
}

func KLDivergence(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	return dispatchPairLoss(predictions, targets, count, format, runKLDivergenceF32)
}

func CrossEntropy(
	logits unsafe.Pointer,
	targets unsafe.Pointer,
	batchSize, classes int,
	format dtype.DType,
) float32 {
	return dispatchCrossEntropy(logits, targets, batchSize, classes, format)
}
