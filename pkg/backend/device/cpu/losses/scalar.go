package losses

import "math"

func MseSumFloat32Scalar(predictions, targets []float32) float32 {
	var sum float32

	for index, value := range predictions {
		diff := value - targets[index]
		sum += diff * diff
	}

	return sum
}

func MaeSumFloat32Scalar(predictions, targets []float32) float32 {
	var sum float32

	for index, value := range predictions {
		sum += float32(math.Abs(float64(value - targets[index])))
	}

	return sum
}
