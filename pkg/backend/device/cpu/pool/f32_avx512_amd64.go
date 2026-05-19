//go:build amd64

package pool

//go:noescape
func MaxPool2DStride1RowAVX512Asm(
	outRow, input *float32,
	outCols, kH, kW, inHStride, ihStart int,
)

//go:noescape
func AvgPool2DStride1RowAVX512Asm(
	outRow, input *float32,
	outCols, kH, kW, inHStride, ihStart int,
)

//go:noescape
func MaxPool2x2Stride2RowAVX512Asm(
	outRow, input *float32,
	outCols, inWidth, ihStart int,
)

//go:noescape
func AvgPool2x2Stride2RowAVX512Asm(
	outRow, input *float32,
	outCols, inWidth, ihStart int,
)
