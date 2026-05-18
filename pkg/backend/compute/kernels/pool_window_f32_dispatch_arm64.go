//go:build arm64

package kernels

func poolWindowMaxFloat32Native(
	channel []float32,
	inWidth, startRow, endRow, startCol, endCol int,
) float32 {
	elementCount := (endRow - startRow) * (endCol - startCol)

	if elementCount <= 0 {
		return poolWindowMaxScalar(channel, inWidth, startRow, endRow, startCol, endCol)
	}

	scratch := borrowFloat32Buffer(elementCount)
	defer releaseFloat32Buffer(scratch)

	poolWindowGather(channel, scratch, inWidth, startRow, endRow, startCol, endCol)

	return reduceMaxFloat32Native(scratch[:elementCount])
}

func poolWindowAvgFloat32Native(
	channel []float32,
	inWidth, startRow, endRow, startCol, endCol int,
) float32 {
	elementCount := (endRow - startRow) * (endCol - startCol)

	if elementCount <= 0 {
		return poolWindowAvgScalar(channel, inWidth, startRow, endRow, startCol, endCol)
	}

	scratch := borrowFloat32Buffer(elementCount)
	defer releaseFloat32Buffer(scratch)

	poolWindowGather(channel, scratch, inWidth, startRow, endRow, startCol, endCol)

	return sumFloat32Native(scratch[:elementCount]) / float32(elementCount)
}
