//go:build arm64

package kernels

//go:noescape
func maxPool2DStride1RowNEONAsm(outRow, input *float32, outCols, kH, kW, inHStride, ihStart int)

//go:noescape
func avgPool2DStride1RowNEONAsm(outRow, input *float32, outCols, kH, kW, inHStride, ihStart int)

//go:noescape
func maxPool2x2Stride2RowNEONAsm(outRow, input *float32, outCols, inWidth, ihStart int)

//go:noescape
func avgPool2x2Stride2RowNEONAsm(outRow, input *float32, outCols, inWidth, ihStart int)
