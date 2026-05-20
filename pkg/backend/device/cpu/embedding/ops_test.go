package embedding

import (
	"testing"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func TestLookup(t *testing.T) {
	table := []float32{1, 2, 3, 4, 5, 6}
	indices := []int32{2, 0}
	output := make([]float32, 4)

	Lookup(
		unsafe.Pointer(&table[0]),
		unsafe.Pointer(&indices[0]),
		unsafe.Pointer(&output[0]),
		3, 2, 2,
		dtype.Float32,
	)

	if output[0] != 5 || output[1] != 6 || output[2] != 1 || output[3] != 2 {
		t.Fatalf("unexpected lookup output: %v", output)
	}
}
