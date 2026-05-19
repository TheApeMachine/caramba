package neon

func PoolWindowMaxScalar(
	channel []float32,
	inWidth, startRow, endRow, startCol, endCol int,
) float32 {
	maximum := float32(-1e30)

	for rowIndex := startRow; rowIndex < endRow; rowIndex++ {
		rowOffset := rowIndex * inWidth

		for colIndex := startCol; colIndex < endCol; colIndex++ {
			value := channel[rowOffset+colIndex]

			if value > maximum {
				maximum = value
			}
		}
	}

	return maximum
}

func PoolWindowAvgScalar(
	channel []float32,
	inWidth, startRow, endRow, startCol, endCol int,
) float32 {
	var sum float32
	count := 0

	for rowIndex := startRow; rowIndex < endRow; rowIndex++ {
		rowOffset := rowIndex * inWidth

		for colIndex := startCol; colIndex < endCol; colIndex++ {
			sum += channel[rowOffset+colIndex]
			count++
		}
	}

	if count == 0 {
		return 0
	}

	return sum / float32(count)
}

func PoolWindowGather(
	channel, scratch []float32,
	inWidth, startRow, endRow, startCol, endCol int,
) {
	scratchIndex := 0

	for rowIndex := startRow; rowIndex < endRow; rowIndex++ {
		rowOffset := rowIndex * inWidth

		for colIndex := startCol; colIndex < endCol; colIndex++ {
			scratch[scratchIndex] = channel[rowOffset+colIndex]
			scratchIndex++
		}
	}
}
