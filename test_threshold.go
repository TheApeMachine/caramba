package main

import (
	"fmt"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/activation"
)

func main() {
	src := []float32{1.0, 2.0, 3.0, 4.0}
	dst := make([]float32, 4)
	activation.ThresholdF32NEON(&dst[0], &src[0], 4, 2.5)
	fmt.Printf("dst: %v\n", dst)
}
