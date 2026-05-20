//go:build !amd64

package tokenizer

var packInt32Funcs = []int32PackKernelImpl{
	{packInt32Generic, "generic", true},
}
