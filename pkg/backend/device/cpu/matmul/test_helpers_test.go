package matmul

import (
	"github.com/theapemachine/manifesto/tensor"
)

var parityNs = []int{1, 7, 64, 1024, 8192}

func mustShape(dims []int) tensor.Shape {
	shape, err := tensor.NewShape(dims)

	if err != nil {
		panic(err)
	}

	return shape
}
