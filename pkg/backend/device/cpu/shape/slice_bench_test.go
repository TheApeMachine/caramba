package shape

import (
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func BenchmarkRunSlice(benchmark *testing.B) {
	const inner = 4

	const tail = 7

	seqLen := 8192
	inputShape, err := tensor.NewShape([]int{1, seqLen + tail, inner})
	if err != nil {
		benchmark.Fatal(err)
	}

	outShape, err := tensor.NewShape([]int{1, seqLen, inner})
	if err != nil {
		benchmark.Fatal(err)
	}

	hostInput, err := tensor.NewZeroed(inputShape, dtype.Float32)
	if err != nil {
		benchmark.Fatal(err)
	}

	hostOutput, err := tensor.NewZeroed(outShape, dtype.Float32)
	if err != nil {
		benchmark.Fatal(err)
	}

	inputView, err := hostInput.Float32Native()
	if err != nil {
		benchmark.Fatal(err)
	}

	for index := range inputView {
		inputView[index] = float32(index) * 0.01
	}

	dimTensor, err := newInt32ScalarTensor(1)
	if err != nil {
		benchmark.Fatal(err)
	}

	startTensor, err := newInt32ScalarTensor(0)
	if err != nil {
		benchmark.Fatal(err)
	}

	endTensor, err := newInt32ScalarTensor(int32(seqLen))
	if err != nil {
		benchmark.Fatal(err)
	}

	benchmark.ResetTimer()

	for benchmark.Loop() {
		if err := RunSlice(hostInput, dimTensor, startTensor, endTensor, hostOutput); err != nil {
			benchmark.Fatal(err)
		}
	}
}
