package pool

/*
PoolConfig describes kernel size, stride, and padding for fixed-size
2-D pooling windows.
*/
type PoolConfig struct {
	KernelH  int
	KernelW  int
	StrideH  int
	StrideW  int
	PaddingH int
	PaddingW int
}

func DefaultPoolConfig() PoolConfig {
	return PoolConfig{KernelH: 2, KernelW: 2, StrideH: 2, StrideW: 2}
}
