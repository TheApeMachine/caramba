package main

import (
	"fmt"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/activation"
)

func main() {
	src := []float32{-1.0}
	dst := make([]float32, 1)
	activation.RReLUF32NEON(&dst[0], &src[0], 1, 0.1, 0.3)
	fmt.Printf("dst: %v\n", dst)
}
