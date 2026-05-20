package neon

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/device/cpu/active_inference"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/attention"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/causal"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/checkpoint"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/convolution"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/dequant"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/dropout"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/hawkes"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/masking"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/matmul"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/optimizer"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/physics"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/pool"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/predictive_coding"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/quant"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/rope"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/sampling"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/shape"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/tokenizer"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/vsa"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func DequantInt8Native(dst []float32, src []int8, scale float32, zeroPoint int8) {
	dequant.DequantInt8Native(dst, src, scale, zeroPoint)
}

func DequantInt4Native(dst []float32, pairs tensor.Int4Vector, scale float32, zeroPoint int8) {
	dequant.DequantInt4Native(dst, pairs, scale, zeroPoint)
}

func QuantInt8Native(dst []int8, src []float32, scale float32, zeroPoint int8) {
	quant.QuantInt8Native(dst, src, scale, zeroPoint)
}

func RopePairsNative(out, in, cosBuf, sinBuf []float32) {
	rope.RopePairsNative(out, in, cosBuf, sinBuf)
}

func SparseCSRMatMulFloat32Native(
	outView, valuesView, rightView []float32,
	rowPtr, colIdx []int32,
	rows, cols int,
) {
	matmul.SparseCSRMatMulFloat32Native(outView, valuesView, rightView, rowPtr, colIdx, rows, cols)
}

func runGreedySample(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	logits, err := args[0].Float32Native()

	if err != nil {
		return err
	}

	out, err := args[1].Int32Native()

	if err != nil {
		return err
	}

	if len(logits) == 0 || len(out) < 1 {
		return tensor.ErrShapeMismatch
	}

	out[0] = sampling.GreedySample(unsafe.Pointer(&logits[0]), len(logits), dtype.Float32)

	return nil
}

func runTopKSampleDefault(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	logits, err := args[0].Float32Native()

	if err != nil {
		return err
	}

	out, err := args[1].Int32Native()

	if err != nil {
		return err
	}

	if len(out) < 1 {
		return tensor.ErrShapeMismatch
	}

	out[0] = sampling.TopKSample(
		sampling.DefaultSamplingConfig(),
		unsafe.Pointer(&logits[0]),
		len(logits),
		dtype.Float32,
	)

	return nil
}

func runTopPSampleDefault(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	logits, err := args[0].Float32Native()

	if err != nil {
		return err
	}

	out, err := args[1].Int32Native()

	if err != nil {
		return err
	}

	if len(out) < 1 {
		return tensor.ErrShapeMismatch
	}

	out[0] = sampling.TopPSample(
		sampling.DefaultSamplingConfig(),
		unsafe.Pointer(&logits[0]),
		len(logits),
		dtype.Float32,
	)

	return nil
}

func runDropoutDefault(args ...tensor.Tensor) error {
	return dropout.RunDropoutDefault(args...)
}

var (
	runWhereFloat32 = shape.RunWhereFloat32

	runMaskedFillFloat32 = shape.RunMaskedFillFloat32

	runTranspose = shape.RunTranspose

	runReshape = shape.RunReshape

	runUpsampleNearest2D = shape.RunUpsampleNearest2D

	runLastToken = shape.RunLastToken

	runMergeHeads = shape.RunMergeHeads

	runSplitHeads = shape.RunSplitHeads

	runSplit2 = shape.RunSplit2

	runViewAsHeads = shape.RunViewAsHeads

	runSlice = shape.RunSlice

	runVSABind = vsa.RunVSABind

	runVSABundle = vsa.RunVSABundle

	runVSAPermuteDefault = vsa.RunVSAPermuteDefault

	runVSAInversePermuteDefault = vsa.RunVSAInversePermuteDefault

	runFreeEnergy = active_inference.RunFreeEnergy

	runExpectedFreeEnergy = active_inference.RunExpectedFreeEnergy

	runBeliefUpdate = active_inference.RunBeliefUpdate

	runPrecisionWeight = active_inference.RunPrecisionWeight

	runPCPrediction = predictive_coding.RunPCPrediction

	runPCPredictionError = predictive_coding.RunPCPredictionError

	runPCUpdateRepresentationDefault = predictive_coding.RunPCUpdateRepresentationDefault

	runPCUpdateWeightsDefault = predictive_coding.RunPCUpdateWeightsDefault

	runCholesky = causal.RunCholesky

	runBackdoorAdjustment = causal.RunBackdoorAdjustment

	runFrontdoorAdjustment = causal.RunFrontdoorAdjustment

	runDoIntervene = causal.RunDoIntervene

	runCATE = causal.RunCATE

	runCounterfactual = causal.RunCounterfactual

	runIVEstimate = causal.RunIVEstimate

	runDAGMarkovFactorization = causal.RunDAGMarkovFactorization

	runMarkovFlowActive = causal.RunMarkovFlowActive

	runMarkovFlowInternal = causal.RunMarkovFlowInternal

	runApplyMask = masking.RunApplyMask

	runCausalMask = masking.RunCausalMask

	runALiBiBias = masking.RunALiBiBias

	runCausalMaskBFloat16 = masking.RunCausalMaskBFloat16

	runCausalMaskFloat16 = masking.RunCausalMaskFloat16

	runAttentionFloat32 = attention.RunAttentionFloat32

	runFlashAttentionFloat32Default = attention.RunFlashAttentionFloat32Default

	runMultiHeadAttentionDefault = attention.RunMultiHeadAttentionDefault

	runGroupedQueryAttentionDefault = attention.RunGroupedQueryAttentionDefault

	runSlidingWindowAttentionDefault = attention.RunSlidingWindowAttentionDefault

	runHawkesIntensity = hawkes.RunHawkesIntensity

	runHawkesKernelMatrix = hawkes.RunHawkesKernelMatrix

	runHawkesLogLikelihood = hawkes.RunHawkesLogLikelihood

	runMarkovMutualInformation = hawkes.RunMarkovMutualInformation

	runMarkovBlanketPartition = hawkes.RunMarkovBlanketPartition

	runLaplacian = physics.RunLaplacian

	runLaplacian4 = physics.RunLaplacian4

	runGrad1D = physics.RunGrad1D

	runDivergence1D = physics.RunDivergence1D

	runFFT1DDefault = physics.RunFFT1DDefault

	runIFFT1DDefault = physics.RunIFFT1DDefault

	runQuantumPotential = physics.RunQuantumPotential

	runBohmianVelocity = physics.RunBohmianVelocity

	runMadelungContinuity = physics.RunMadelungContinuity

	runConv2DDefault = convolution.RunConv2DDefault

	runConv1DDefault = convolution.RunConv1DDefault

	runConv3DDefault = convolution.RunConv3DDefault

	runConvTranspose2DDefault = convolution.RunConvTranspose2DDefault

	runMaxPool2DDefault = pool.RunMaxPool2DDefault

	runAvgPool2DDefault = pool.RunAvgPool2DDefault

	runAdaptiveAvgPool2DDefault = pool.RunAdaptiveAvgPool2DDefault

	runAdaptiveMaxPool2DDefault = pool.RunAdaptiveMaxPool2DDefault

	runAdamStepDefault = optimizer.RunAdamStepDefault

	runAdamWStepDefault = optimizer.RunAdamWStepDefault

	runLionStepDefault = optimizer.RunLionStepDefault

	runSGDStepDefault = optimizer.RunSGDStepDefault

	runAdamaxStepDefault = optimizer.RunAdamaxStepDefault

	runAdagradStepDefault = optimizer.RunAdagradStepDefault

	runRMSpropStepDefault = optimizer.RunRMSpropStepDefault

	runLARSStepDefault = optimizer.RunLARSStepDefault

	runHebbianStepDefault = optimizer.RunHebbianStepDefault

	runLBFGSStepDefault = optimizer.RunLBFGSStepDefault

	runTokenizerPackInt32 = tokenizer.RunTokenizerPackInt32

	runInt8DequantDefault = dequant.RunInt8DequantDefault

	runInt4DequantDefault = dequant.RunInt4DequantDefault

	runInt8QuantDefault = quant.RunInt8QuantDefault

	runCheckpointEncodeFloat32 = checkpoint.RunCheckpointEncodeFloat32

	runCheckpointDecodeFloat32 = checkpoint.RunCheckpointDecodeFloat32

	runMatMulInt8 = matmul.RunMatMulInt8
)
