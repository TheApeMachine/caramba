//go:build !amd64

package shape

var copyContiguousF32Funcs = []f32CopyContiguousKernelImpl{
	{copyContiguousF32Generic, "generic", true},
}

var whereF32Funcs = []f32WhereKernelImpl{
	{whereF32Generic, "generic", true},
}

var maskedFillF32Funcs = []f32MaskedFillKernelImpl{
	{maskedFillF32Generic, "generic", true},
}
