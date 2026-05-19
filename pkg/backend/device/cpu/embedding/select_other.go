//go:build !amd64

package embedding

import "unsafe"

func runLookupF32Native(
	table, indices, output unsafe.Pointer,
	vocab, hidden, indexCount int,
) {
	runLookupF32Generic(table, indices, output, vocab, hidden, indexCount)
}

func runBagF32Native(
	table, indices, offsets, output unsafe.Pointer,
	vocab, hidden, bagCount, indexCount int,
) {
	runBagF32Generic(table, indices, offsets, output, vocab, hidden, bagCount, indexCount)
}
