package activation

import (
	"testing"
)

func TestPReLU(t *testing.T) {
	dst := make([]float32, 16)
	src := []float32{1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8}
	slopes := []float32{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}
	
	PReLUVF32Generic(&dst[0], &src[0], &slopes[0], 16)
	
	for i := range dst {
		t.Logf("src=%f, dst=%f", src[i], dst[i])
	}
}
