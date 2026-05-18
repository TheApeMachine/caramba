//go:build !arm64

package kernels

func poolWindowMaxFloat32Native(
	channel []float32,
	inWidth, startRow, endRow, startCol, endCol int,
) float32 {
	return poolWindowMaxScalar(channel, inWidth, startRow, endRow, startCol, endCol)
}

func poolWindowAvgFloat32Native(
	channel []float32,
	inWidth, startRow, endRow, startCol, endCol int,
) float32 {
	return poolWindowAvgScalar(channel, inWidth, startRow, endRow, startCol, endCol)
}
