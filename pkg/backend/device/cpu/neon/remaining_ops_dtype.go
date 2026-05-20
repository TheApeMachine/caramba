package neon

import (
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/attention"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/optimizer"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
Sweep file for the remaining production-critical and research-domain
op families that were registered f32-only. Each entry below specifies
the f32 runner, the input dtype mask (which input positions are
"params" rounded to the reduced dtype vs which are pass-through like
Int32 indices), and the output dtype mask.

The runner widens param-dtype inputs to f32 scratch tensors, calls the
f32 runner with those plus the pass-through tensors, and narrows the
f32 outputs back. fp32 accumulation per §5.5.

The pattern is mechanical, so most ops register through this table
rather than each requiring a hand-written wrapper.
*/

// opSpec describes a mixed-precision wrapper config.
type opSpec struct {
	name string
	// inputDTypes mirrors the F32 kernel's input dtypes. Positions of
	// dtype.Float32 become the paramDType in the new registration;
	// other dtypes (Int32, Bool) are pass-through.
	inputDTypes []dtype.DType
	// outputDTypes mirrors the F32 kernel's outputs. Positions of
	// dtype.Float32 become the paramDType in the new registration.
	outputDTypes []dtype.DType
	runF32       func(args ...tensor.Tensor) error
}

func (spec opSpec) registerF32() {
	Default.Register(Kernel{
		Name: spec.name,
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  spec.inputDTypes,
			Outputs: spec.outputDTypes,
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       spec.runF32,
	})
}

func (spec opSpec) registerMixed(paramDType dtype.DType) {
	inputs := make([]dtype.DType, len(spec.inputDTypes))

	for index, dt := range spec.inputDTypes {
		if dt == dtype.Float32 {
			inputs[index] = paramDType
		} else {
			inputs[index] = dt
		}
	}

	outputs := make([]dtype.DType, len(spec.outputDTypes))

	for index, dt := range spec.outputDTypes {
		if dt == dtype.Float32 {
			outputs[index] = paramDType
		} else {
			outputs[index] = dt
		}
	}

	totalArity := len(inputs) + len(outputs)

	Default.Register(Kernel{
		Name: spec.name,
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  inputs,
			Outputs: outputs,
		},
		Locations: []tensor.Location{tensor.Host},
		Run: func(args ...tensor.Tensor) error {
			return runGenericMixed(args, paramDType, spec, totalArity)
		},
	})
}

func runGenericMixed(
	args []tensor.Tensor,
	paramDType dtype.DType,
	spec opSpec,
	totalArity int,
) error {
	if len(args) != totalArity {
		return tensor.ErrShapeMismatch
	}

	// Build the f32 arg list. Inputs that are paramDType get widened into
	// fresh f32 temp tensors; pass-through inputs are forwarded as-is.
	// Outputs that are paramDType get fresh f32 temps; pass-through
	// outputs are forwarded as-is.
	f32Args := make([]tensor.Tensor, totalArity)
	temps := make(map[int]tensor.Tensor)

	for index, arg := range args {
		var isParam bool

		if index < len(spec.inputDTypes) {
			isParam = spec.inputDTypes[index] == dtype.Float32
		} else {
			outIndex := index - len(spec.inputDTypes)
			isParam = spec.outputDTypes[outIndex] == dtype.Float32
		}

		if !isParam {
			f32Args[index] = arg
			continue
		}

		temp, err := tensor.NewZeroed(arg.Shape(), dtype.Float32)

		if err != nil {
			return err
		}

		f32Args[index] = temp
		temps[index] = temp

		// Inputs need widening; outputs stay zero-init.
		if index < len(spec.inputDTypes) {
			tempView, _ := temp.Float32Native()

			if err := widenToF32(arg, paramDType, tempView); err != nil {
				return err
			}
		}
	}

	if err := spec.runF32(f32Args...); err != nil {
		return err
	}

	// Narrow paramDType outputs back to the caller's tensors.
	for index, temp := range temps {
		if index < len(spec.inputDTypes) {
			continue
		}

		tempView, _ := temp.Float32Native()

		if err := narrowFromF32(args[index], paramDType, tempView); err != nil {
			return err
		}
	}

	return nil
}

func init() {
	specs := []opSpec{
		// === math_extended.go (f32-only entries) ===
		// inv_sqrt_dim_scale: (Float32, Int32) → Float32
		{
			name:         "inv_sqrt_dim_scale",
			inputDTypes:  []dtype.DType{dtype.Float32, dtype.Int32},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runInvSqrtDimScale,
		},
		// logsumexp: (Float32) → Float32
		{
			name:         "logsumexp",
			inputDTypes:  []dtype.DType{dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runLogSumExp,
		},
		// outer: (Float32, Float32) → Float32
		{
			name:         "outer",
			inputDTypes:  []dtype.DType{dtype.Float32, dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runOuter,
		},
		// matmul_add: (A, B, bias) → out
		{
			name: "matmul_add",
			inputDTypes: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runMatMulAdd,
		},

		// === masking ===
		{
			name:         "apply_mask",
			inputDTypes:  []dtype.DType{dtype.Float32, dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runApplyMask,
		},
		{
			name:         "alibi_bias",
			inputDTypes:  []dtype.DType{dtype.Float32, dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runALiBiBias,
		},

		// === attention ===
		{
			name: "attention",
			inputDTypes: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runAttentionFloat32,
		},
		{
			name: "flash_attention",
			inputDTypes: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runFlashAttentionFloat32Default,
		},
		{
			name: "multi_head_attention",
			inputDTypes: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runMultiHeadAttentionDefault,
		},
		{
			name: "grouped_query_attention",
			inputDTypes: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runGroupedQueryAttentionDefault,
		},
		{
			name: "sliding_window_attention",
			inputDTypes: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runSlidingWindowAttentionDefault,
		},

		// === projection.go ===
		// linear: (x, W, b) → y
		{
			name:         "linear",
			inputDTypes:  []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runLinear,
		},
		// fused_qkv: (x, Wqkv, bqkv) → (Q, K, V)
		{
			name:         "fused_qkv",
			inputDTypes:  []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32},
			runF32:       runFusedQKV,
		},

		// === dropout.go ===
		{
			name:         "dropout",
			inputDTypes:  []dtype.DType{dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runDropoutDefault,
		},

		// === pool ===
		{
			name:         "max_pool2d",
			inputDTypes:  []dtype.DType{dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runMaxPool2DDefault,
		},
		{
			name:         "avg_pool2d",
			inputDTypes:  []dtype.DType{dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runAvgPool2DDefault,
		},
		{
			name:         "adaptive_avg_pool2d",
			inputDTypes:  []dtype.DType{dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runAdaptiveAvgPool2DDefault,
		},
		{
			name:         "adaptive_max_pool2d",
			inputDTypes:  []dtype.DType{dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runAdaptiveMaxPool2DDefault,
		},

		// === convolution ===
		{
			name: "conv1d",
			inputDTypes: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runConv1DDefault,
		},
		{
			name: "conv2d",
			inputDTypes: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runConv2DDefault,
		},
		{
			name: "conv3d",
			inputDTypes: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runConv3DDefault,
		},
		{
			name: "conv_transpose2d",
			inputDTypes: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runConvTranspose2DDefault,
		},

		// === sampling.go ===
		// greedy_sample: (logits Float32) → (token_id Int32)
		{
			name:         "greedy_sample",
			inputDTypes:  []dtype.DType{dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Int32},
			runF32:       runGreedySample,
		},
		// topk_sample: (logits Float32) → (token_id Int32)
		{
			name:         "topk_sample",
			inputDTypes:  []dtype.DType{dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Int32},
			runF32:       runTopKSampleDefault,
		},
		// topp_sample: (logits Float32) → (token_id Int32)
		{
			name:         "topp_sample",
			inputDTypes:  []dtype.DType{dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Int32},
			runF32:       runTopPSampleDefault,
		},

		// === model_ops.go ===
		// lora_apply: (base, A, B, output)
		{
			name:         "lora_apply",
			inputDTypes:  []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runLoRAApplyDefault,
		},
		// lora_merge: (base, A, B, output)
		{
			name:         "lora_merge",
			inputDTypes:  []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runLoRAMergeDefault,
		},
		// weight_freeze_mask: (weights, mask, output)
		{
			name:         "weight_freeze_mask",
			inputDTypes:  []dtype.DType{dtype.Float32, dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runWeightFreezeMask,
		},

		// === shape_ops.go (remaining: where, masked_fill) ===
		// where: (cond Bool, a, b) → output
		{
			name:         "where",
			inputDTypes:  []dtype.DType{dtype.Bool, dtype.Float32, dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runWhereFloat32,
		},
		// masked_fill: (input, mask Bool, value) → output
		{
			name:         "masked_fill",
			inputDTypes:  []dtype.DType{dtype.Float32, dtype.Bool, dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runMaskedFillFloat32,
		},

		// === shape_more.go ===
		// transpose: (input) → output
		{
			name:         "transpose",
			inputDTypes:  []dtype.DType{dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runTranspose,
		},
		// reshape: (input) → output (logical reshape; copy)
		{
			name:         "reshape",
			inputDTypes:  []dtype.DType{dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runReshape,
		},
		// upsample_nearest2d: (input) → output
		{
			name:         "upsample_nearest2d",
			inputDTypes:  []dtype.DType{dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runUpsampleNearest2D,
		},

		// === shape_extended.go ===
		// last_token: (input) → output
		{
			name:         "last_token",
			inputDTypes:  []dtype.DType{dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runLastToken,
		},
		// merge_heads: (input) → output
		{
			name:         "merge_heads",
			inputDTypes:  []dtype.DType{dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runMergeHeads,
		},
		// split_heads: (input) → output
		{
			name:         "split_heads",
			inputDTypes:  []dtype.DType{dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runSplitHeads,
		},
		// split2: (input) → (a, b)
		{
			name:         "split2",
			inputDTypes:  []dtype.DType{dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Float32, dtype.Float32},
			runF32:       runSplit2,
		},
		// slice: (input, dim, start, end) → output
		{
			name:         "slice",
			inputDTypes:  []dtype.DType{dtype.Float32, dtype.Int32, dtype.Int32, dtype.Int32},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runSlice,
		},
		// view_as_heads: (input) → output (reshape-style)
		{
			name:         "view_as_heads",
			inputDTypes:  []dtype.DType{dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runViewAsHeads,
		},

		// === vsa.go ===
		{name: "vsa_bind", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runVSABind},
		{name: "vsa_bundle", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runVSABundle},
		{name: "vsa_permute", inputDTypes: []dtype.DType{dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runVSAPermuteDefault},
		{name: "vsa_inverse_permute", inputDTypes: []dtype.DType{dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runVSAInversePermuteDefault},

		// === active_inference.go ===
		{name: "free_energy", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runFreeEnergy},
		{name: "expected_free_energy", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runExpectedFreeEnergy},
		{name: "belief_update", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runBeliefUpdate},
		{name: "precision_weight", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runPrecisionWeight},

		// === predictive_coding.go ===
		{name: "pc_prediction", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runPCPrediction},
		{name: "pc_prediction_error", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runPCPredictionError},
		{name: "pc_update_representation", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runPCUpdateRepresentationDefault},
		{name: "pc_update_weights", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runPCUpdateWeightsDefault},

		// === causal.go ===
		{name: "cholesky", inputDTypes: []dtype.DType{dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runCholesky},
		{name: "backdoor_adjustment", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runBackdoorAdjustment},
		{name: "frontdoor_adjustment", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runFrontdoorAdjustment},
		{name: "do_intervene", inputDTypes: []dtype.DType{dtype.Float32, dtype.Int32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runDoIntervene},
		{name: "cate", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runCATE},

		// === causal_extended.go ===
		{name: "counterfactual", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runCounterfactual},
		{name: "iv_estimate", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runIVEstimate},
		{name: "dag_markov_factorization", inputDTypes: []dtype.DType{dtype.Float32, dtype.Int32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runDAGMarkovFactorization},
		{name: "markov_flow_active", inputDTypes: []dtype.DType{dtype.Float32, dtype.Int32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runMarkovFlowActive},
		{name: "markov_flow_internal", inputDTypes: []dtype.DType{dtype.Float32, dtype.Int32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runMarkovFlowInternal},

		// === hawkes_markov.go ===
		{name: "hawkes_intensity", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32, dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runHawkesIntensity},
		{name: "hawkes_kernel_matrix", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runHawkesKernelMatrix},
		{name: "hawkes_log_likelihood", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32, dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runHawkesLogLikelihood},
		{name: "markov_mutual_information", inputDTypes: []dtype.DType{dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runMarkovMutualInformation},
		{name: "markov_blanket_partition", inputDTypes: []dtype.DType{dtype.Float32, dtype.Int32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runMarkovBlanketPartition},

		// === physics.go ===
		{name: "laplacian", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runLaplacian},
		{name: "laplacian4", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runLaplacian4},
		{name: "grad1d", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runGrad1D},
		{name: "divergence1d", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runDivergence1D},
		{name: "fft1d", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runFFT1DDefault},
		{name: "ifft1d", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runIFFT1DDefault},
		{name: "quantum_potential", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runQuantumPotential},
		{name: "bohmian_velocity", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runBohmianVelocity},
		{name: "madelung_continuity", inputDTypes: []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32}, outputDTypes: []dtype.DType{dtype.Float32}, runF32: runMadelungContinuity},

		// === optimizer ===
		{
			name: "adam_step",
			inputDTypes: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32, dtype.Float32,
			},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runAdamStepDefault,
		},
		{
			name: "adamw_step",
			inputDTypes: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32, dtype.Float32,
			},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runAdamWStepDefault,
		},
		{
			name: "lion_step",
			inputDTypes: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runLionStepDefault,
		},
		{
			name: "sgd_step",
			inputDTypes: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runSGDStepDefault,
		},
		{
			name: "adamax_step",
			inputDTypes: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32, dtype.Float32,
			},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runAdamaxStepDefault,
		},
		{
			name: "adagrad_step",
			inputDTypes: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runAdagradStepDefault,
		},
		{
			name: "rmsprop_step",
			inputDTypes: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runRMSpropStepDefault,
		},
		{
			name: "lars_step",
			inputDTypes: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runLARSStepDefault,
		},
		{
			name: "hebbian_step",
			inputDTypes: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runHebbianStepDefault,
		},
		{
			name: "lbfgs_step",
			inputDTypes: []dtype.DType{
				dtype.Float32, dtype.Float32,
			},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runLBFGSStepDefault,
		},

		// === matmul int8 ===
		{
			name:         "matmul",
			inputDTypes:  []dtype.DType{dtype.Int8, dtype.Int8},
			outputDTypes: []dtype.DType{dtype.Int32},
			runF32:       runMatMulInt8,
		},

		// === quantization ===
		{
			name:         "int8_dequant",
			inputDTypes:  []dtype.DType{dtype.Int8},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runInt8DequantDefault,
		},
		{
			name:         "int4_dequant",
			inputDTypes:  []dtype.DType{dtype.Int4},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runInt4DequantDefault,
		},
		{
			name:         "int8_quant",
			inputDTypes:  []dtype.DType{dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Int8},
			runF32:       runInt8QuantDefault,
		},

		// === checkpoint ===
		{
			name:         "checkpoint_encode_float32",
			inputDTypes:  []dtype.DType{dtype.Float32},
			outputDTypes: []dtype.DType{dtype.Uint8},
			runF32:       runCheckpointEncodeFloat32,
		},
		{
			name:         "checkpoint_decode_float32",
			inputDTypes:  []dtype.DType{dtype.Uint8},
			outputDTypes: []dtype.DType{dtype.Float32},
			runF32:       runCheckpointDecodeFloat32,
		},

		// === tokenizer ===
		{
			name:         "tokenizer_pack_int32",
			inputDTypes:  []dtype.DType{dtype.Int32},
			outputDTypes: []dtype.DType{dtype.Int32},
			runF32:       runTokenizerPackInt32,
		},
	}

	for _, spec := range specs {
		spec.registerF32()
	}

	for _, paramDType := range []dtype.DType{dtype.BFloat16, dtype.Float16} {
		for _, spec := range specs {
			if spec.name == "multi_head_attention" ||
				spec.name == "grouped_query_attention" ||
				spec.name == "sliding_window_attention" ||
				isOptimizerStep(spec.name) ||
				skipMixedRegistration(spec.name) {
				continue
			}

			spec.registerMixed(paramDType)
		}
	}

	registerMaskingReducedPrecision()
	registerAttentionVariantsReducedPrecision()
	optimizer.RegisterMixedPrecisionSteps()
}

func isOptimizerStep(name string) bool {
	switch name {
	case "adam_step", "adamw_step", "lion_step", "sgd_step",
		"adamax_step", "adagrad_step", "rmsprop_step", "lars_step",
		"lbfgs_step", "hebbian_step":
		return true
	default:
		return false
	}
}

func skipMixedRegistration(name string) bool {
	switch name {
	case "matmul",
		"int8_dequant", "int4_dequant", "int8_quant",
		"checkpoint_encode_float32", "checkpoint_decode_float32",
		"tokenizer_pack_int32":
		return true
	default:
		return false
	}
}

func registerAttentionVariantsReducedPrecision() {
	for _, variant := range []struct {
		name string
		run  func(args ...tensor.Tensor) error
	}{
		{name: "multi_head_attention", run: runMultiHeadAttentionDefault},
		{name: "grouped_query_attention", run: runGroupedQueryAttentionDefault},
		{name: "sliding_window_attention", run: runSlidingWindowAttentionDefault},
	} {
		variant := variant
		Default.Register(Kernel{
			Name: variant.name,
			Signature: Signature{
				Layout: tensor.LayoutDense,
				Inputs: []dtype.DType{
					dtype.BFloat16, dtype.BFloat16, dtype.BFloat16,
				},
				Outputs: []dtype.DType{dtype.BFloat16},
			},
			Locations: []tensor.Location{tensor.Host},
			Run: func(args ...tensor.Tensor) error {
				return attention.RunMultiHeadAttentionVariantBFloat16(variant.name, args...)
			},
		})
		Default.Register(Kernel{
			Name: variant.name,
			Signature: Signature{
				Layout: tensor.LayoutDense,
				Inputs: []dtype.DType{
					dtype.Float16, dtype.Float16, dtype.Float16,
				},
				Outputs: []dtype.DType{dtype.Float16},
			},
			Locations: []tensor.Location{tensor.Host},
			Run: func(args ...tensor.Tensor) error {
				return attention.RunMultiHeadAttentionVariantFloat16(variant.name, args...)
			},
		})
	}
}

func registerMaskingReducedPrecision() {
	Default.Register(Kernel{
		Name: "causal_mask",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runCausalMask,
	})
	Default.Register(Kernel{
		Name: "causal_mask",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.BFloat16},
			Outputs: []dtype.DType{dtype.BFloat16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runCausalMaskBFloat16,
	})
	Default.Register(Kernel{
		Name: "causal_mask",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float16},
			Outputs: []dtype.DType{dtype.Float16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runCausalMaskFloat16,
	})
}
