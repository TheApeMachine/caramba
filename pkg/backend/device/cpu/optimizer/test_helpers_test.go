package optimizer

import "github.com/theapemachine/caramba/pkg/backend/compute/tensor"

func mustShape(dims []int) tensor.Shape {
	shape, err := tensor.NewShape(dims)

	if err != nil {
		panic(err)
	}

	return shape
}
