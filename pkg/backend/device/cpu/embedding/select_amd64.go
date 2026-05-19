//go:build amd64

package embedding

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

func runLookupF32Native(
	table, indices, output unsafe.Pointer,
	vocab, hidden, indexCount int,
) {
	if cpu.X86.HasAVX512F {
		runLookupF32AVX512(table, indices, output, vocab, hidden, indexCount)

		return
	}

	runLookupF32Generic(table, indices, output, vocab, hidden, indexCount)
}

func runBagF32Native(
	table, indices, offsets, output unsafe.Pointer,
	vocab, hidden, bagCount, indexCount int,
) {
	if cpu.X86.HasAVX512F {
		runBagF32AVX512(table, indices, offsets, output, vocab, hidden, bagCount, indexCount)

		return
	}

	runBagF32Generic(table, indices, offsets, output, vocab, hidden, bagCount, indexCount)
}

func runLookupF32AVX512(
	table, indices, output unsafe.Pointer,
	vocab, hidden, indexCount int,
) {
	tableView := unsafe.Slice((*float32)(table), vocab*hidden)
	outputView := unsafe.Slice((*float32)(output), indexCount*hidden)

	for resultIndex := 0; resultIndex < indexCount; resultIndex++ {
		tokenID := int(loadInt32(indices, resultIndex))

		if tokenID < 0 || tokenID >= vocab {
			panic("embedding: index out of range")
		}

		copyRowF32AVX512(
			&outputView[resultIndex*hidden],
			&tableView[tokenID*hidden],
			hidden,
		)
	}
}

func runBagF32AVX512(
	table, indices, offsets, output unsafe.Pointer,
	vocab, hidden, bagCount, indexCount int,
) {
	tableView := unsafe.Slice((*float32)(table), vocab*hidden)
	outputView := unsafe.Slice((*float32)(output), bagCount*hidden)

	for bagIndex := 0; bagIndex < bagCount; bagIndex++ {
		startIdx := int(loadInt32(offsets, bagIndex))
		endIdx := indexCount

		if bagIndex+1 < bagCount {
			endIdx = int(loadInt32(offsets, bagIndex+1))
		}

		outRow := &outputView[bagIndex*hidden]

		for elementIndex := startIdx; elementIndex < endIdx; elementIndex++ {
			tokenID := int(loadInt32(indices, elementIndex))

			if tokenID < 0 || tokenID >= vocab {
				panic("embedding: index out of range")
			}

			addRowF32AVX512(
				outRow,
				&tableView[tokenID*hidden],
				hidden,
			)
		}
	}
}
