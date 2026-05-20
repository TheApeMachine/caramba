package pool

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestMaxPool2DFloat32(t *testing.T) {
	inputShape, err := tensor.NewShape([]int{1, 1, 4, 4})

	if err != nil {
		t.Fatal(err)
	}

	input, err := tensor.NewZeroed(inputShape, dtype.Float32)

	if err != nil {
		t.Fatal(err)
	}

	outputShape, err := tensor.NewShape([]int{1, 1, 2, 2})

	if err != nil {
		t.Fatal(err)
	}

	output, err := tensor.NewZeroed(outputShape, dtype.Float32)

	if err != nil {
		t.Fatal(err)
	}

	inputView, err := input.Float32Native()

	if err != nil {
		t.Fatal(err)
	}

	for index := range inputView {
		inputView[index] = float32(index)
	}

	config := PoolConfig{KernelH: 2, KernelW: 2, StrideH: 2, StrideW: 2}

	if err := MaxPool2DFloat32(config, input, output); err != nil {
		t.Fatal(err)
	}

	outputView, err := output.Float32Native()

	if err != nil {
		t.Fatal(err)
	}

	if outputView[0] != 5 {
		t.Fatalf("output[0]=%g want 5", outputView[0])
	}

	if outputView[3] != 15 {
		t.Fatalf("output[3]=%g want 15", outputView[3])
	}
}

func TestPoolWindowMaxFloat32NativeParity(t *testing.T) {
	channel := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	got := PoolWindowMaxFloat32Native(channel, 3, 0, 2, 0, 2)
	want := PoolWindowMaxScalar(channel, 3, 0, 2, 0, 2)

	if got != want {
		t.Fatalf("got=%g want=%g", got, want)
	}
}
