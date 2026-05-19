package avx2

type Conv2DConfig struct {
	StrideH   int
	StrideW   int
	PaddingH  int
	PaddingW  int
	DilationH int
	DilationW int
}

func DefaultConv2DConfig() Conv2DConfig {
	return Conv2DConfig{
		StrideH: 1, StrideW: 1,
		PaddingH: 0, PaddingW: 0,
		DilationH: 1, DilationW: 1,
	}
}

func convTranspose2DConfigAVX2Eligible(config Conv2DConfig) bool {
	return config.StrideH == 1 &&
		config.StrideW == 1 &&
		config.PaddingH == 0 &&
		config.PaddingW == 0 &&
		config.DilationH == 1 &&
		config.DilationW == 1
}
