package main

import (
	"fmt"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/activation"
)

func main() {
	src := []float32{1.0, -1.0, 2.0, -2.0}
	dst := make([]float32, 4)
	activation.SigmoidF32SSE2(&dst[0], &src[0], 4)
	fmt.Printf("SigmoidF32SSE2(1.0, -1.0, 2.0, -2.0) = %v\n", dst)
}
