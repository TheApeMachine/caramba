//go:build arm64

package neon

//go:noescape
func Conv2dStride1RowNEONAsm(
	outRow, input, weight *float32,
	biasValue float32,
	outCols, inChannels, kH, kW int,
	inHStride, inCStride int,
	wHStride, wCStride int,
	ihStart, iwStart int,
)
