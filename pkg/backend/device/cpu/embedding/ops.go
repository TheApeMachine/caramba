package embedding

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func Lookup(
	table, indices, output unsafe.Pointer,
	vocab, hidden, indexCount int,
	format dtype.DType,
) {
	dispatchLookup(table, indices, output, vocab, hidden, indexCount, format)
}

func Bag(
	table, indices, offsets, output unsafe.Pointer,
	vocab, hidden, bagCount, indexCount int,
	format dtype.DType,
) {
	dispatchBag(table, indices, offsets, output, vocab, hidden, bagCount, indexCount, format)
}
