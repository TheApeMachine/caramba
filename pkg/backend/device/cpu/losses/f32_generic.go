package losses

import "unsafe"

func MseSumF32Generic(predictions, targets *float32, count int) float32 {
	predictionView := unsafe.Slice(predictions, count)
	targetView := unsafe.Slice(targets, count)

	return MseSumFloat32Scalar(predictionView, targetView)
}

func MaeSumF32Generic(predictions, targets *float32, count int) float32 {
	predictionView := unsafe.Slice(predictions, count)
	targetView := unsafe.Slice(targets, count)

	return MaeSumFloat32Scalar(predictionView, targetView)
}
