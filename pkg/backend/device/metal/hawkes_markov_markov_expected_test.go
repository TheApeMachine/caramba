package metal

func markovJointValues(rows int, cols int) []float32 {
	values := make([]float32, rows*cols)
	var sum float32

	for index := range values {
		values[index] = 0.01 + 0.99*float32((index*17+5)%101)/100
		sum += values[index]
	}

	for index := range values {
		values[index] /= sum
	}

	return values
}

func markovAdjacencyValues(nodeCount int) []float32 {
	values := make([]float32, nodeCount*nodeCount)

	for rowIndex := range nodeCount {
		for colIndex := range nodeCount {
			if rowIndex != colIndex && (rowIndex*7+colIndex*11)%5 == 0 {
				values[rowIndex*nodeCount+colIndex] = 1
			}
		}
	}

	return values
}

func markovMutualInformationMatrix(nodeCount int) []float32 {
	values := make([]float32, nodeCount*nodeCount)

	for rowIndex := range nodeCount {
		for colIndex := range nodeCount {
			values[rowIndex*nodeCount+colIndex] = float32((rowIndex+1)*(colIndex+3)%23) / 31
		}
	}

	return values
}

func markovInternalNodes(nodeCount int) []int32 {
	if nodeCount == 1 {
		return []int32{0}
	}

	return []int32{0, int32(nodeCount / 2)}
}

func markovPartitionLabels(nodeCount int) []int32 {
	labels := make([]int32, nodeCount)

	for index := range labels {
		labels[index] = int32(index % 4)
	}

	return labels
}

func markovContainsNode(labels []int32, nodeIndex int, nodeCount int) bool {
	for _, label := range labels {
		if label >= 0 && int(label) < nodeCount && int(label) == nodeIndex {
			return true
		}
	}

	return false
}
