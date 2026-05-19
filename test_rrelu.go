package main

import (
	"fmt"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/activation"
)

func main() {
	src := []float32{-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0}
	dst := make([]float32, 8)
	activation.RReLUF32NEON(&dst[0], &src[0], 8, 0.1, 0.3)
	fmt.Printf("dst: %v\n", dst)
}
