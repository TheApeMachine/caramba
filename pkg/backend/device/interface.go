package device

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

type PosPop interface {
	CountString(counts *[8]int, str string)
	Count8(counts *[8]int, buf []uint8)
	Count16(counts *[16]int, buf []uint16)
	Count32(counts *[32]int, buf []uint32)
	Count64(counts *[64]int, buf []uint64)
}

type Activation interface {
	Exp(dst, src unsafe.Pointer, count int, format dtype.DType)
	Log(dst, src unsafe.Pointer, count int, format dtype.DType)
	Log1p(dst, src unsafe.Pointer, count int, format dtype.DType)
	Expm1(dst, src unsafe.Pointer, count int, format dtype.DType)
	Sigmoid(dst, src unsafe.Pointer, count int, format dtype.DType)
	LogSigmoid(dst, src unsafe.Pointer, count int, format dtype.DType)
	Tanh(dst, src unsafe.Pointer, count int, format dtype.DType)
	Silu(dst, src unsafe.Pointer, count int, format dtype.DType)
	Swish(dst, src unsafe.Pointer, count int, format dtype.DType)
	GeluTanh(dst, src unsafe.Pointer, count int, format dtype.DType)
	Gelu(dst, src unsafe.Pointer, count int, format dtype.DType)
	ReLU(dst, src unsafe.Pointer, count int, format dtype.DType)
	LeakyReLU(dst, src unsafe.Pointer, count int, format dtype.DType)
	ELU(dst, src unsafe.Pointer, count int, format dtype.DType)
	CELU(dst, src unsafe.Pointer, count int, format dtype.DType)
	SELU(dst, src unsafe.Pointer, count int, format dtype.DType)
	Softplus(dst, src unsafe.Pointer, count int, format dtype.DType)
	Mish(dst, src unsafe.Pointer, count int, format dtype.DType)
	Softsign(dst, src unsafe.Pointer, count int, format dtype.DType)
	HardSigmoid(dst, src unsafe.Pointer, count int, format dtype.DType)
	HardSwish(dst, src unsafe.Pointer, count int, format dtype.DType)
	HardTanh(dst, src unsafe.Pointer, count int, format dtype.DType)
	HardGelu(dst, src unsafe.Pointer, count int, format dtype.DType)
	QuickGelu(dst, src unsafe.Pointer, count int, format dtype.DType)
	TanhShrink(dst, src unsafe.Pointer, count int, format dtype.DType)

	Softmax(dst, src unsafe.Pointer, count int, format dtype.DType)
	LogSoftmax(dst, src unsafe.Pointer, count int, format dtype.DType)

	PReLU(dst, src unsafe.Pointer, count int, format dtype.DType, negativeSlope float32)
	PReLUV(dst, src, slopes unsafe.Pointer, count int, format dtype.DType, slopeCount int)
	LeakyReLUSlope(dst, src unsafe.Pointer, count int, format dtype.DType, negativeSlope float32)
	ELUAlpha(dst, src unsafe.Pointer, count int, format dtype.DType, alpha float32)
	CELUAlpha(dst, src unsafe.Pointer, count int, format dtype.DType, alpha float32)
	Threshold(dst, src unsafe.Pointer, count int, format dtype.DType, threshold float32)
	HardTanhRange(dst, src unsafe.Pointer, count int, format dtype.DType, minVal, maxVal float32)
	Snake(dst, src unsafe.Pointer, count int, format dtype.DType, alpha float32)
	SnakeParametric(dst, src unsafe.Pointer, count int, format dtype.DType, alpha, beta float32)
	HardShrink(dst, src unsafe.Pointer, count int, format dtype.DType, lambda float32)
	SoftShrink(dst, src unsafe.Pointer, count int, format dtype.DType, lambda float32)
	RReLU(dst, src unsafe.Pointer, count int, format dtype.DType, lower, upper float32)

	GLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
	GeGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
	GeGLUTanh(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
	SwiGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
	ReGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
	SiGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
	GLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
	GeGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
	GeGLUTanhTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
	SwiGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
	ReGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
	SiGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
	LinGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
	SeGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
	LinGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
	SeGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
}

type Elementwise interface {
	Add(dst, left, right unsafe.Pointer, count int, format dtype.DType)
	Sub(dst, left, right unsafe.Pointer, count int, format dtype.DType)
	Mul(dst, left, right unsafe.Pointer, count int, format dtype.DType)
	Div(dst, left, right unsafe.Pointer, count int, format dtype.DType)
	Max(dst, left, right unsafe.Pointer, count int, format dtype.DType)
	Min(dst, left, right unsafe.Pointer, count int, format dtype.DType)
	Abs(dst, src unsafe.Pointer, count int, format dtype.DType)
	Neg(dst, src unsafe.Pointer, count int, format dtype.DType)
	Sqrt(dst, src unsafe.Pointer, count int, format dtype.DType)
	ReLU(dst, src unsafe.Pointer, count int, format dtype.DType)
	Axpy(y, x unsafe.Pointer, count int, alpha float32, format dtype.DType)
}

type Reduction interface {
	Sum(values unsafe.Pointer, count int, format dtype.DType) float32
	Prod(values unsafe.Pointer, count int, format dtype.DType) float32
	ReduceMin(values unsafe.Pointer, count int, format dtype.DType) float32
	ReduceMax(values unsafe.Pointer, count int, format dtype.DType) float32
	L1Norm(values unsafe.Pointer, count int, format dtype.DType) float32
}

type Dot interface {
	Dot(left, right unsafe.Pointer, count int, format dtype.DType) float32
}

type Matmul interface {
	Matmul(
		out, left, right unsafe.Pointer,
		rows, inner, cols int,
		format dtype.DType,
	)
}

type Pool interface {
	MaxPool2D(
		config PoolConfig,
		input, output unsafe.Pointer,
		batch, channels, inHeight, inWidth, outHeight, outWidth int,
		format dtype.DType,
	)
	AvgPool2D(
		config PoolConfig,
		input, output unsafe.Pointer,
		batch, channels, inHeight, inWidth, outHeight, outWidth int,
		format dtype.DType,
	)
	AdaptiveMaxPool2D(
		input, output unsafe.Pointer,
		batch, channels, inHeight, inWidth, outHeight, outWidth int,
		format dtype.DType,
	)
	AdaptiveAvgPool2D(
		input, output unsafe.Pointer,
		batch, channels, inHeight, inWidth, outHeight, outWidth int,
		format dtype.DType,
	)
}

type Convolution interface {
	Conv2D(
		config Conv2DConfig,
		input, weight, bias, output unsafe.Pointer,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth int,
		format dtype.DType,
	)
	Conv1D(
		config Conv1DConfig,
		input, weight, bias, output unsafe.Pointer,
		batch, inChannels, inLength, outChannels, kernelLength, outLength int,
		format dtype.DType,
	)
	Conv3D(
		config Conv3DConfig,
		input, weight, bias, output unsafe.Pointer,
		batch, inChannels, inD, inH, inW,
		outChannels, kD, kH, kW, outD, outH, outW int,
		format dtype.DType,
	)
	ConvTranspose2D(
		config Conv2DConfig,
		input, weight, bias, output unsafe.Pointer,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth int,
		format dtype.DType,
	)
}

type Dropout interface {
	Dropout(
		dst, src unsafe.Pointer,
		count int,
		config DropoutConfig,
		format dtype.DType,
	)
}

type Losses interface {
	MSE(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32
	MAE(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32
	Huber(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32
	BinaryCrossEntropy(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32
	KLDivergence(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32
	CrossEntropy(
		logits unsafe.Pointer,
		targets unsafe.Pointer,
		batchSize, classes int,
		format dtype.DType,
	) float32
}

type Sampling interface {
	GreedySample(logits unsafe.Pointer, vocabSize int, format dtype.DType) int32
	TopKSample(config SamplingConfig, logits unsafe.Pointer, vocabSize int, format dtype.DType) int32
	TopPSample(config SamplingConfig, logits unsafe.Pointer, vocabSize int, format dtype.DType) int32
}

type Embedding interface {
	Lookup(
		table, indices, output unsafe.Pointer,
		vocab, hidden, indexCount int,
		format dtype.DType,
	)
	Bag(
		table, indices, offsets, output unsafe.Pointer,
		vocab, hidden, bagCount, indexCount int,
		format dtype.DType,
	)
}

type Normalization interface {
	GroupNorm(
		config GroupNormConfig,
		input, scale, bias, output unsafe.Pointer,
		batch, channels, spatial int,
		format dtype.DType,
	)
	InstanceNorm(
		input, scale, bias, output unsafe.Pointer,
		batch, channels, spatial int,
		format dtype.DType,
	)
	BatchNormEval(
		input, scale, bias, mean, variance, output unsafe.Pointer,
		batch, channels, spatial int,
		format dtype.DType,
	)
}

type LayerNorm interface {
	LayerNorm(
		input, scale, bias, output unsafe.Pointer,
		rows, lastDim int,
		format dtype.DType,
	)
	RMSNorm(
		input, scale, output unsafe.Pointer,
		rows, lastDim int,
		format dtype.DType,
	)
}

type RoPE interface {
	RoPE(
		config RoPEConfig,
		input, output unsafe.Pointer,
		seqLen, numHeads, headDim int,
		format dtype.DType,
	)
	RoPEPairs(
		output, input, cosBuffer, sinBuffer unsafe.Pointer,
		halfDim int,
		format dtype.DType,
	)
}

type Hawkes interface {
	HawkesIntensity(
		eventTimes, queryTimes, output unsafe.Pointer,
		eventCount, queryCount int,
		mu, alpha, beta float32,
		format dtype.DType,
	)
	HawkesKernelMatrix(
		eventTimes, output unsafe.Pointer,
		eventCount int,
		alpha, beta float32,
		format dtype.DType,
	)
	HawkesLogLikelihood(
		eventTimes unsafe.Pointer,
		eventCount int,
		totalT, mu, alpha, beta float32,
		output unsafe.Pointer,
		format dtype.DType,
	)
	MarkovMutualInformation(
		joint, output unsafe.Pointer,
		xCount, yCount int,
		format dtype.DType,
	)
	MarkovBlanketPartition(
		adjacency, internal, output unsafe.Pointer,
		nodeCount, internalCount int,
		format dtype.DType,
	)
}

type Physics interface {
	Laplacian(input, output unsafe.Pointer, dims []int, spacing float32, format dtype.DType)
	Laplacian4(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType)
	Grad1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType)
	Divergence1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType)
	FFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType)
	IFFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType)
	QuantumPotential(density, output unsafe.Pointer, count int, spacing float32, format dtype.DType)
	BohmianVelocity(phase, output unsafe.Pointer, count int, spacing float32, format dtype.DType)
	MadelungContinuity(
		density, velocity, residual unsafe.Pointer,
		count int,
		spacing float32,
		format dtype.DType,
	)
}

type Causal interface {
	Cholesky(input, output unsafe.Pointer, matrixOrder int, format dtype.DType)
	BackdoorAdjustment(
		conditional, marginalZ, output unsafe.Pointer,
		xCount, zCount, yCount int,
		format dtype.DType,
	)
	FrontdoorAdjustment(
		mediatorGivenX, outcomeGivenXM, marginalX, output unsafe.Pointer,
		xCount, mediatorCount, yCount int,
		format dtype.DType,
	)
	DoIntervene(
		adjacency, intervened, output unsafe.Pointer,
		nodeCount, intervenedCount int,
		format dtype.DType,
	)
	CATE(treated, control, output unsafe.Pointer, count int, format dtype.DType)
	Counterfactual(
		observedY, observedX, counterfactualX, output unsafe.Pointer,
		count int,
		slope float32,
		format dtype.DType,
	)
	IVEstimate(
		instrument, treatment, outcome unsafe.Pointer,
		count int,
		output unsafe.Pointer,
		format dtype.DType,
	)
	DAGMarkovFactorization(
		conditionals unsafe.Pointer,
		conditionalCount int,
		output unsafe.Pointer,
		format dtype.DType,
	)
	MarkovFlowActive(
		mutualInformation, partition, output unsafe.Pointer,
		nodeCount int,
		format dtype.DType,
	)
	MarkovFlowInternal(
		mutualInformation, partition, output unsafe.Pointer,
		nodeCount int,
		format dtype.DType,
	)
}

type Masking interface {
	ApplyMask(input, mask, output unsafe.Pointer, count int, format dtype.DType)
	CausalMask(output unsafe.Pointer, seqQ, seqK int, format dtype.DType)
	ALiBiBias(scores, slope, output unsafe.Pointer, seqQ, seqK int, format dtype.DType)
}

type Attention interface {
	ScaledDotProductAttention(
		config FlashAttentionConfig,
		query, key, value, output unsafe.Pointer,
		seqQ, seqK, depth, valueDim int,
		format dtype.DType,
	)
	FlashAttention(
		config FlashAttentionConfig,
		query, key, value, output unsafe.Pointer,
		seqQ, seqK, depth, valueDim int,
		format dtype.DType,
	)
	MultiHeadAttention(
		config MultiHeadAttentionConfig,
		query, key, value, output unsafe.Pointer,
		seqQ, seqK int,
		format dtype.DType,
	)
}

type VSA interface {
	Bind(left, right, output unsafe.Pointer, count int, format dtype.DType)
	Bundle(left, right, output unsafe.Pointer, count int, format dtype.DType)
	Permute(config VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType)
	InversePermute(config VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType)
	Similarity(left, right unsafe.Pointer, count int, format dtype.DType) float32
}

type ActiveInference interface {
	FreeEnergy(
		likelihood, posterior, prior, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	ExpectedFreeEnergy(
		predictedObs, preferredObs, predictedState, output unsafe.Pointer,
		obsCount, stateCount int,
		format dtype.DType,
	)
	BeliefUpdate(likelihood, prior, output unsafe.Pointer, count int, format dtype.DType)
	PrecisionWeight(errors, precision, output unsafe.Pointer, count int, format dtype.DType)
}

type PredictiveCoding interface {
	Prediction(
		weights, representation, output unsafe.Pointer,
		outDim, inDim int,
		format dtype.DType,
	)
	PredictionError(
		observed, predicted, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	UpdateRepresentation(
		config PredictiveCodingConfig,
		weights, representation, predictionError, output unsafe.Pointer,
		outDim, inDim int,
		format dtype.DType,
	)
	UpdateWeights(
		config PredictiveCodingConfig,
		weights, representation, predictionError, output unsafe.Pointer,
		outDim, inDim int,
		format dtype.DType,
	)
}

type Dequant interface {
	Dequant(dst, src unsafe.Pointer, count int, config DequantInt8Config, dstFormat, srcFormat dtype.DType)
	Dequant4(dst, src unsafe.Pointer, pairCount int, config DequantInt4Config, dstFormat, srcFormat dtype.DType)
}

type Quant interface {
	Quant(dst, src unsafe.Pointer, count int, config DequantInt8Config, dstFormat, srcFormat dtype.DType)
}

type Backend interface {
	PosPop
	Activation
	Elementwise
	Reduction
	Dot
	Matmul
	Pool
	Convolution
	Dropout
	Losses
	Sampling
	Embedding
	Normalization
	LayerNorm
	RoPE
	Hawkes
	Physics
	Causal
	Masking
	Attention
	VSA
	ActiveInference
	PredictiveCoding
	Dequant
	Quant
}
