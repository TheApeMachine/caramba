//go:build amd64

package tokenizer

import "golang.org/x/sys/cpu"

var packInt32Funcs = []int32PackKernelImpl{
	{TokenizerPackInt32AVX512, "avx512", cpu.X86.HasAVX512F},
	{packInt32Generic, "generic", true},
}
