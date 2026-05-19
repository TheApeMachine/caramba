//go:build arm64

package cpu

func PoolWindowMaxFloat32Native(
	channel []float32,
	inWidth, startRow, endRow, startCol, endCol int,
) float32 {
	elementCount := (endRow - startRow) * (endCol - startCol)

	if elementCount <= 0 {
		return PoolWindowMaxScalar(channel, inWidth, startRow, endRow, startCol, endCol)
	}

	scratch := BorrowFloat32Buffer(elementCount)
	defer ReleaseFloat32Buffer(scratch)

	PoolWindowGather(channel, scratch, inWidth, startRow, endRow, startCol, endCol)

	return ReduceMaxFloat32Native(scratch[:elementCount])
}

func PoolWindowAvgFloat32Native(
	channel []float32,
	inWidth, startRow, endRow, startCol, endCol int,
) float32 {
	elementCount := (endRow - startRow) * (endCol - startCol)

	if elementCount <= 0 {
		return PoolWindowAvgScalar(channel, inWidth, startRow, endRow, startCol, endCol)
	}

	scratch := BorrowFloat32Buffer(elementCount)
	defer ReleaseFloat32Buffer(scratch)

	PoolWindowGather(channel, scratch, inWidth, startRow, endRow, startCol, endCol)

	return SumFloat32Native(scratch[:elementCount]) / float32(elementCount)
}
