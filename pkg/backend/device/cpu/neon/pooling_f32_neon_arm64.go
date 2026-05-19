//go:build arm64

package neon

//go:noescape
func MaxPool2DStride1RowNEONAsm(outRow, input *float32, outCols, kH, kW, inHStride, ihStart int)

//go:noescape
func AvgPool2DStride1RowNEONAsm(outRow, input *float32, outCols, kH, kW, inHStride, ihStart int)

//go:noescape
func MaxPool2x2Stride2RowNEONAsm(outRow, input *float32, outCols, inWidth, ihStart int)

//go:noescape
func AvgPool2x2Stride2RowNEONAsm(outRow, input *float32, outCols, inWidth, ihStart int)
