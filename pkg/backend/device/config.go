package device

/*
Shared configuration structs referenced by device interfaces. Domain
packages define matching types locally; the field layout must stay in
sync with these definitions.
*/

type DropoutConfig struct {
	Rate float32
	Seed uint64
}

type PoolConfig struct {
	KernelH  int
	KernelW  int
	StrideH  int
	StrideW  int
	PaddingH int
	PaddingW int
}

type Conv2DConfig struct {
	StrideH   int
	StrideW   int
	PaddingH  int
	PaddingW  int
	DilationH int
	DilationW int
}

type Conv1DConfig struct {
	Stride   int
	Padding  int
	Dilation int
}

type Conv3DConfig struct {
	StrideD, StrideH, StrideW       int
	PaddingD, PaddingH, PaddingW    int
	DilationD, DilationH, DilationW int
}

type SamplingConfig struct {
	Temperature float32
	TopK        int
	TopP        float32
	Seed        uint64
}

type GroupNormConfig struct {
	Groups int
}

type RoPEConfig struct {
	BaseFreq      float64
	StartPosition int
}

type VSAConfig struct {
	Shift int
}

type PredictiveCodingConfig struct {
	LearningRate float32
}

type DequantInt8Config struct {
	Scale     float32
	ZeroPoint int8
}

type DequantInt4Config struct {
	Scale     float32
	ZeroPoint int8
}

type FlashAttentionConfig struct {
	BlockSize int
	Causal    bool
}

type MultiHeadAttentionConfig struct {
	NumHeads    int
	HeadDim     int
	Causal      bool
	WindowSize  int
	ALiBiSlope  float32
	KVHeadCount int
}
