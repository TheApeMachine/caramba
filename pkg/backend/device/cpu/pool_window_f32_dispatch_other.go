//go:build !arm64

package cpu

func PoolWindowMaxFloat32Native(
	channel []float32,
	inWidth, startRow, endRow, startCol, endCol int,
) float32 {
	return PoolWindowMaxScalar(channel, inWidth, startRow, endRow, startCol, endCol)
}

func PoolWindowAvgFloat32Native(
	channel []float32,
	inWidth, startRow, endRow, startCol, endCol int,
) float32 {
	return PoolWindowAvgScalar(channel, inWidth, startRow, endRow, startCol, endCol)
}
