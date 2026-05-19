package embedding

import "unsafe"

func runLookupF32Generic(
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

		copy(
			outputView[resultIndex*hidden:(resultIndex+1)*hidden],
			tableView[tokenID*hidden:(tokenID+1)*hidden],
		)
	}
}

func runBagF32Generic(
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

		outOffset := bagIndex * hidden

		for elementIndex := startIdx; elementIndex < endIdx; elementIndex++ {
			tokenID := int(loadInt32(indices, elementIndex))

			if tokenID < 0 || tokenID >= vocab {
				panic("embedding: index out of range")
			}

			for dimIndex := 0; dimIndex < hidden; dimIndex++ {
				outputView[outOffset+dimIndex] += tableView[tokenID*hidden+dimIndex]
			}
		}
	}
}
