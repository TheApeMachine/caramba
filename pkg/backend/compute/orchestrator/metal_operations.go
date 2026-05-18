package orchestrator

import (
	"slices"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
MetalOperationCoverage is the resident Metal operation contract consumed by lowering,
tests, benchmarks, and startup validation.
*/
type MetalOperationCoverage struct {
	ID               ir.OpType
	DTypes           []dtype.DType
	ShapeConstraints []string
	ResidentSymbol   string
	FusionGroups     []string
	BenchmarkName    string
	ParityTestName   string
}

var residentMetalOperationTable = []MetalOperationCoverage{
	residentOperation(ir.OpInput, "metal_tensor_upload_float32_mode", "BenchmarkTensorBackend_Upload", "TestTensorBackend_Upload"),
	residentOperationConstraints(ir.OpAdd, "metal_add_tensor", "BenchmarkMathOps_AddTensor", "TestMathOps_AddTensor", "inputs.same_shape"),
	residentOperationConstraints(ir.OpMul, "metal_mul_tensor", "BenchmarkMathOps_MulTensor", "TestMathOps_MulTensor", "inputs.same_shape"),
	residentOperationConstraints(ir.OpMatmul, "metal_matmul_tensor", "BenchmarkMathOps_MatmulTensor", "TestMathOps_MatmulTensor", "matmul.rank2"),
	residentOperation(ir.OpReLU, "metal_relu_tensor", "BenchmarkMetalActivation_ReLUTensor", "TestMetalActivation_ReLUTensor"),
	residentOperation(ir.OpLeakyReLU, "metal_leaky_relu_tensor", "BenchmarkMetalActivation_LeakyReLUTensor", "TestMetalActivation_LeakyReLUTensor"),
	residentOperation(ir.OpGELU, "metal_gelu_tensor", "BenchmarkMetalActivation_GELUTensor", "TestMetalActivation_GELUTensor"),
	residentOperation(ir.OpTanh, "metal_tanh_tensor", "BenchmarkMetalActivation_TanhTensor", "TestMetalActivation_TanhTensor"),
	residentOperation(ir.OpSigmoid, "metal_sigmoid_tensor", "BenchmarkMetalActivation_SigmoidTensor", "TestMetalActivation_SigmoidTensor"),
	residentOperation(ir.OpSwiGLU, "metal_swiglu_tensor", "BenchmarkMetalActivation_SwiGLUTensor", "TestMetalActivation_SwiGLUTensor"),
	residentOperation(ir.OpSwish, "metal_swish_tensor", "BenchmarkMetalActivation_SwishTensor", "TestMetalActivation_SwishTensor"),
	residentOperation(ir.OpSELU, "metal_selu_tensor", "BenchmarkMetalActivation_SELUTensor", "TestMetalActivation_SELUTensor"),
	residentFusion(ir.OpFused, "metal_matmul_add_tensor", "matmul.activation", "BenchmarkMathOps_MatmulAddGELUTensor", "TestMathOps_MatmulAddGELUTensor"),
	residentOperationConstraints("math.add", "metal_add_tensor", "BenchmarkMathOps_AddTensor", "TestMathOps_AddTensor", "inputs.same_shape"),
	residentOperationConstraints("math.mul", "metal_mul_tensor", "BenchmarkMathOps_MulTensor", "TestMathOps_MulTensor", "inputs.same_shape"),
	residentOperationConstraints("math.matmul", "metal_matmul_tensor", "BenchmarkMathOps_MatmulTensor", "TestMathOps_MatmulTensor", "matmul.rank2"),
	residentOperation("math.exp", "metal_exp_tensor", "BenchmarkMathOps_ExpTensor", "TestMathOps_ExpTensor"),
	residentOperation("math.sin", "metal_sin_tensor", "BenchmarkMathOps_SinTensor", "TestMathOps_SinTensor"),
	residentOperation("math.cos", "metal_cos_tensor", "BenchmarkMathOps_CosTensor", "TestMathOps_CosTensor"),
	residentOperation("math.log", "metal_log_tensor", "BenchmarkMathOps_LogTensor", "TestMathOps_LogTensor"),
	residentOperation("math.logsumexp", "metal_logsumexp_tensor", "BenchmarkMathOps_LogSumExpTensor", "TestMathOps_LogSumExpTensor"),
	residentOperation("math.softmax", "metal_softmax_tensor", "BenchmarkMathOps_SoftmaxTensor", "TestMathOps_SoftmaxTensor"),
	residentOperation("math.outer", "metal_outer_tensor", "BenchmarkMathOps_OuterTensor", "TestMathOps_OuterTensor"),
	residentOperation("math.sign", "metal_sign_tensor", "BenchmarkMathOps_SignTensor", "TestMathOps_SignTensor"),
	residentOperation("math.inv_sqrt_dim_scale", "metal_inv_sqrt_dim_scale_tensor", "BenchmarkMathOps_InvSqrtDimScaleTensor", "TestMathOps_InvSqrtDimScaleTensor"),
	residentOperation("math.dropout", "metal_dropout_tensor", "BenchmarkMathOps_DropoutTensor", "TestMathOps_DropoutTensor"),
	residentOperation("math.rmsnorm", "metal_rmsnorm_tensor", "BenchmarkMathOps_RMSNormTensor", "TestMathOps_RMSNormTensor"),
	residentOperation("math.layernorm", "metal_layernorm_tensor", "BenchmarkMathOps_LayerNormTensor", "TestMathOps_LayerNormTensor"),
	residentOperationConstraints("math.groupnorm", "metal_groupnorm_tensor", "BenchmarkMathOps_GroupNormTensor", "TestMathOps_GroupNormTensor", "input.rank4"),
	residentOperation("activation.relu", "metal_relu_tensor", "BenchmarkMetalActivation_ReLUTensor", "TestMetalActivation_ReLUTensor"),
	residentOperation("activation.leaky_relu", "metal_leaky_relu_tensor", "BenchmarkMetalActivation_LeakyReLUTensor", "TestMetalActivation_LeakyReLUTensor"),
	residentOperation("activation.gelu", "metal_gelu_tensor", "BenchmarkMetalActivation_GELUTensor", "TestMetalActivation_GELUTensor"),
	residentOperation("activation.tanh", "metal_tanh_tensor", "BenchmarkMetalActivation_TanhTensor", "TestMetalActivation_TanhTensor"),
	residentOperation("activation.sigmoid", "metal_sigmoid_tensor", "BenchmarkMetalActivation_SigmoidTensor", "TestMetalActivation_SigmoidTensor"),
	residentOperation("activation.swiglu", "metal_swiglu_tensor", "BenchmarkMetalActivation_SwiGLUTensor", "TestMetalActivation_SwiGLUTensor"),
	residentOperation("activation.swish", "metal_swish_tensor", "BenchmarkMetalActivation_SwishTensor", "TestMetalActivation_SwishTensor"),
	residentOperation("activation.selu", "metal_selu_tensor", "BenchmarkMetalActivation_SELUTensor", "TestMetalActivation_SELUTensor"),
	residentOperation("embedding.token", "metal_token_embedding_tensor", "BenchmarkEmbeddingOps_ForwardTensor", "TestEmbeddingOps_ForwardTensor"),
	residentOperationConstraints("shape.reshape", "metal_tensor_retain", "BenchmarkMetalShapeOps_ReshapeTensor", "TestMetalShapeOps_ReshapeTensor", "output.same_elements_as_input0"),
	residentOperation("shape.transpose", "metal_transpose_tensor", "BenchmarkMetalShapeOps_TransposeTensor", "TestMetalShapeOps_TransposeTensor"),
	residentOperation("shape.concat", "metal_concat_tensor", "BenchmarkMetalShapeOps_ConcatTensor", "TestMetalShapeOps_ConcatTensor"),
	residentOperation("shape.split", "metal_split_tensor", "BenchmarkMetalShapeOps_SplitTensor", "TestMetalShapeOps_SplitTensor"),
	residentOperation("shape.upsample_nearest2d", "metal_upsample_nearest2d_tensor", "BenchmarkMetalShapeOps_UpsampleNearest2DTensor", "TestMetalShapeOps_UpsampleNearest2DTensor"),
	residentOperationConstraints("shape.view_as_heads", "metal_view_as_heads_tensor", "BenchmarkMetalShapeOps_ViewAsHeadsTensor", "TestMetalShapeOps_ViewAsHeadsTensor", "input.rank3", "output.same_elements_as_input0"),
	residentOperationConstraints("shape.merge_heads", "metal_merge_heads_tensor", "BenchmarkMetalShapeOps_MergeHeadsTensor", "TestMetalShapeOps_MergeHeadsTensor", "input.rank4", "output.same_elements_as_input0"),
	residentOperation("shape.last_token", "metal_last_token_tensor", "BenchmarkMetalShapeOps_LastTokenTensor", "TestMetalShapeOps_LastTokenTensor"),
	residentOperation("shape.slice", "metal_copy_tensor", "BenchmarkMetalShapeOps_SlicePrefixTensor", "TestMetalShapeOps_SlicePrefixTensor"),
	residentOperation("projection.linear", "metal_matmul_add_tensor", "BenchmarkMathOps_MatmulAddFlatTensor", "TestMathOps_MatmulAddFlatTensor"),
	residentOperation("projection.fused_qkv", "metal_fused_qkv_tensor", "BenchmarkProjectionOps_FusedQKVTensor", "TestProjectionOps_FusedQKVTensor"),
	residentOperation("attention.sdpa", "metal_sdpa_tensor", "BenchmarkMetalAttention_SDPATensor", "TestMetalAttention_SDPATensor"),
	residentOperation("attention.mqa", "metal_mqa_tensor", "BenchmarkMetalAttention_MQATensor", "TestMetalAttention_MQATensor"),
	residentOperation("attention.gqa", "metal_gqa_tensor", "BenchmarkMetalAttention_GQATensor", "TestMetalAttention_GQATensor"),
	residentOperation("attention.sliding_window", "metal_sliding_window_tensor", "BenchmarkMetalAttention_SlidingWindowTensor", "TestMetalAttention_SlidingWindowTensor"),
	residentOperation("positional.rope", "metal_rope_tensor", "BenchmarkMetalPositional_RoPETensor", "TestMetalPositional_RoPETensor"),
	residentOperation("positional.alibi", "metal_alibi_tensor", "BenchmarkMetalPositional_ALiBiTensor", "TestMetalPositional_ALiBiTensor"),
	residentOperation("convolution.conv1d", "metal_conv1d_tensor", "BenchmarkConvolutionOps_Conv1dTensor", "TestConvolutionOps_Conv1dTensor"),
	residentOperation("convolution.conv2d", "metal_conv2d_tensor", "BenchmarkConvolutionOps_Conv2dTensor", "TestConvolutionOps_Conv2dTensor"),
	residentOperation("convolution.conv3d", "metal_conv3d_tensor", "BenchmarkConvolutionOps_Conv3dTensor", "TestConvolutionOps_Conv3dTensor"),
	residentOperation("convolution.conv_transpose2d", "metal_conv_transpose2d_tensor", "BenchmarkConvolutionOps_ConvTranspose2dTensor", "TestConvolutionOps_ConvTranspose2dTensor"),
	residentOperation("pooling.max_pool2d", "metal_max_pool2d_tensor", "BenchmarkPoolingOps_MaxPool2dTensor", "TestPoolingOps_MaxPool2dTensor"),
	residentOperation("pooling.avg_pool2d", "metal_avg_pool2d_tensor", "BenchmarkPoolingOps_AvgPool2dTensor", "TestPoolingOps_AvgPool2dTensor"),
	residentOperation("pooling.adaptive_avg_pool2d", "metal_adaptive_avg_pool2d_tensor", "BenchmarkPoolingOps_AdaptiveAvgPool2dTensor", "TestPoolingOps_AdaptiveAvgPool2dTensor"),
	residentOperation("pooling.adaptive_max_pool2d", "metal_adaptive_max_pool2d_tensor", "BenchmarkPoolingOps_AdaptiveMaxPool2dTensor", "TestPoolingOps_AdaptiveMaxPool2dTensor"),
	residentOperation("masking.apply", "metal_apply_mask_tensor", "BenchmarkMetalMasking_ApplyMaskTensor", "TestMetalMasking_ApplyMaskTensor"),
	residentOperation("masking.causal", "metal_causal_mask_tensor", "BenchmarkMetalMasking_CausalMaskTensor", "TestMetalMasking_CausalMaskTensor"),
	residentOperation("vsa.bind", "metal_vsa_bind_tensor", "BenchmarkMetalVSAOps_BindTensor", "TestMetalVSAOps_BindTensor"),
	residentOperation("vsa.bundle", "metal_vsa_bundle_tensor", "BenchmarkMetalVSAOps_BundleTensor", "TestMetalVSAOps_BundleTensor"),
	residentOperation("vsa.similarity", "metal_vsa_dot_tensor", "BenchmarkMetalVSAOps_SimilarityTensor", "TestMetalVSAOps_SimilarityTensor"),
	residentOperation("vsa.permute", "metal_vsa_permute_tensor", "BenchmarkMetalVSAOps_PermuteTensor", "TestMetalVSAOps_PermuteTensor"),
	residentOperation("vsa.inverse_permute", "metal_vsa_inverse_permute_tensor", "BenchmarkMetalVSAOps_InversePermuteTensor", "TestMetalVSAOps_InversePermuteTensor"),
	residentOperation("hawkes.intensity", "metal_hawkes_intensity_tensor", "BenchmarkMetalHawkes_IntensityTensor", "TestMetalHawkes_IntensityTensor"),
	residentOperation("hawkes.kernel_matrix", "metal_hawkes_kernel_matrix_tensor", "BenchmarkMetalHawkes_KernelMatrixTensor", "TestMetalHawkes_KernelMatrixTensor"),
	residentOperation("hawkes.log_likelihood", "metal_hawkes_log_likelihood_tensor", "BenchmarkMetalHawkes_LogLikelihoodTensor", "TestMetalHawkes_LogLikelihoodTensor"),
	residentOperation("hawkes.simulate", "metal_hawkes_simulate_tensor", "BenchmarkMetalHawkes_SimulateTensor", "TestMetalHawkes_SimulateTensor"),
	residentOperation("active_inference.free_energy", "metal_ai_free_energy_tensor", "BenchmarkActiveInferenceOps_FreeEnergyTensor", "TestActiveInferenceOps_FreeEnergyTensor"),
	residentOperation("active_inference.belief_update", "metal_ai_belief_update_tensor", "BenchmarkActiveInferenceOps_BeliefUpdateTensor", "TestActiveInferenceOps_BeliefUpdateTensor"),
	residentOperation("active_inference.precision_weight", "metal_ai_precision_weight_tensor", "BenchmarkActiveInferenceOps_PrecisionWeightTensor", "TestActiveInferenceOps_PrecisionWeightTensor"),
	residentOperation("active_inference.expected_free_energy", "metal_ai_expected_free_energy_tensor", "BenchmarkActiveInferenceOps_ExpectedFreeEnergyTensor", "TestActiveInferenceOps_ExpectedFreeEnergyTensor"),
	residentOperation("predictive_coding.prediction", "metal_pc_prediction_tensor", "BenchmarkMetalPredictiveCodingOps_PredictionTensor", "TestMetalPredictiveCodingOps_PredictionTensor"),
	residentOperation("predictive_coding.prediction_error", "metal_pc_prediction_error_tensor", "BenchmarkMetalPredictiveCodingOps_PredictionErrorTensor", "TestMetalPredictiveCodingOps_PredictionErrorTensor"),
	residentOperation("predictive_coding.update_representation", "metal_pc_update_representation_tensor", "BenchmarkMetalPredictiveCodingOps_UpdateRepresentationTensor", "TestMetalPredictiveCodingOps_UpdateRepresentationTensor"),
	residentOperation("predictive_coding.update_weights", "metal_pc_update_weights_tensor", "BenchmarkMetalPredictiveCodingOps_UpdateWeightsTensor", "TestMetalPredictiveCodingOps_UpdateWeightsTensor"),
	residentOperation("markov_blanket.partition", "metal_mb_partition_tensor", "BenchmarkMetalMarkovBlanket_PartitionTensor", "TestMetalMarkovBlanket_PartitionTensor"),
	residentOperation("markov_blanket.flow_internal", "metal_mb_flow_internal_tensor", "BenchmarkMetalMarkovBlanket_FlowInternalTensor", "TestMetalMarkovBlanket_FlowInternalTensor"),
	residentOperation("markov_blanket.flow_active", "metal_mb_flow_active_tensor", "BenchmarkMetalMarkovBlanket_FlowActiveTensor", "TestMetalMarkovBlanket_FlowActiveTensor"),
	residentOperation("markov_blanket.mutual_information", "metal_mb_mutual_information_tensor", "BenchmarkMetalMarkovBlanket_MutualInformationTensor", "TestMetalMarkovBlanket_MutualInformationTensor"),
	residentOperation("causal.counterfactual", "metal_causal_counterfactual_tensor", "BenchmarkMetalCausalOps_CounterfactualTensor", "TestMetalCausalOps_CounterfactualTensor"),
	residentOperation("causal.frontdoor_adjustment", "metal_causal_frontdoor_tensor", "BenchmarkMetalCausalOps_FrontdoorAdjustmentTensor", "TestMetalCausalOps_FrontdoorAdjustmentTensor"),
	residentOperation("causal.backdoor_adjustment", "metal_causal_backdoor_tensor", "BenchmarkMetalCausalOps_BackdoorAdjustmentTensor", "TestMetalCausalOps_BackdoorAdjustmentTensor"),
	residentOperation("causal.cate", "metal_causal_cate_tensor", "BenchmarkMetalCausalOps_CATETensor", "TestMetalCausalOps_CATETensor"),
	residentOperation("causal.iv_estimate", "metal_causal_iv_tensor", "BenchmarkMetalCausalOps_IVEstimateTensor", "TestMetalCausalOps_IVEstimateTensor"),
	residentOperation("causal.dag_markov_factorization", "metal_causal_dag_markov_tensor", "BenchmarkMetalCausalOps_DAGMarkovFactorizationTensor", "TestMetalCausalOps_DAGMarkovFactorizationTensor"),
	residentOperation("causal.do_calculus", "metal_causal_do_calculus_tensor", "BenchmarkMetalCausalOps_DoCalculusTensor", "TestMetalCausalOps_DoCalculusTensor"),
	residentOperation("train.loss.mse", "metal_train_mse_loss_tensor", "BenchmarkMetalTraining_MSELossTensor", "TestMetalTraining_MSELossTensor"),
	residentOperation("train.loss.cross_entropy", "metal_train_cross_entropy_loss_tensor", "BenchmarkMetalTraining_CrossEntropyLossTensor", "TestMetalTraining_CrossEntropyLossTensor"),
	residentOperation("train.loss.mse_grad", "metal_train_mse_grad_tensor", "BenchmarkMetalTraining_MSEGradTensor", "TestMetalTraining_MSEGradTensor"),
	residentOperation("train.grad.mse", "metal_train_mse_grad_tensor", "BenchmarkMetalTraining_MSEGradTensor", "TestMetalTraining_MSEGradTensor"),
	residentOperation("train.loss.cross_entropy_grad", "metal_train_cross_entropy_grad_tensor", "BenchmarkMetalTraining_CrossEntropyGradTensor", "TestMetalTraining_CrossEntropyGradTensor"),
	residentOperation("train.grad.cross_entropy", "metal_train_cross_entropy_grad_tensor", "BenchmarkMetalTraining_CrossEntropyGradTensor", "TestMetalTraining_CrossEntropyGradTensor"),
	residentOperation("bench.accuracy", "metal_bench_accuracy_tensor", "BenchmarkMetalMetric_AccuracyTensor", "TestMetalMetric_AccuracyTensor"),
	residentOperation("bench.metric.accuracy", "metal_bench_accuracy_tensor", "BenchmarkMetalMetric_AccuracyTensor", "TestMetalMetric_AccuracyTensor"),
	residentOperation("bench.perplexity", "metal_bench_perplexity_tensor", "BenchmarkMetalMetric_PerplexityTensor", "TestMetalMetric_PerplexityTensor"),
	residentOperation("bench.metric.perplexity", "metal_bench_perplexity_tensor", "BenchmarkMetalMetric_PerplexityTensor", "TestMetalMetric_PerplexityTensor"),
	residentOperation("bench.f1", "metal_bench_f1_tensor", "BenchmarkMetalMetric_F1Tensor", "TestMetalMetric_F1Tensor"),
	residentOperation("bench.metric.f1", "metal_bench_f1_tensor", "BenchmarkMetalMetric_F1Tensor", "TestMetalMetric_F1Tensor"),
	residentOperation("train.optimizer.adam", "metal_optimizer_adam_tensor", "BenchmarkMetalOptimizerTensor_Adam", "TestTensorBackend_applyOptimizerGraph"),
	residentOperation("train.optimizer.adamw", "metal_optimizer_adamw_tensor", "BenchmarkMetalOptimizerTensor_AdamW", "TestTensorBackend_applyOptimizerGraph"),
	residentOperation("train.optimizer.adamax", "metal_optimizer_adamax_tensor", "BenchmarkMetalOptimizerTensor_AdaMax", "TestTensorBackend_applyOptimizerGraph"),
	residentOperation("train.optimizer.sgd", "metal_optimizer_sgd_tensor", "BenchmarkMetalOptimizerTensor_SGD", "TestTensorBackend_applyOptimizerGraph"),
	residentOperation("train.optimizer.lion", "metal_optimizer_lion_tensor", "BenchmarkMetalOptimizerTensor_Lion", "TestTensorBackend_applyOptimizerGraph"),
	residentOperation("train.optimizer.rmsprop", "metal_optimizer_rmsprop_tensor", "BenchmarkMetalOptimizerTensor_RMSProp", "TestTensorBackend_applyOptimizerGraph"),
	residentOperation("train.optimizer.hebbian", "metal_optimizer_hebbian_tensor", "BenchmarkMetalOptimizerTensor_Hebbian", "TestTensorBackend_applyOptimizerGraph"),
	residentOperation("train.optimizer.lars", "metal_optimizer_lars_tensor", "BenchmarkMetalOptimizerTensor_LARS", "TestTensorBackend_applyOptimizerGraph"),
	residentOperation("train.optimizer.lamb", "metal_optimizer_lamb_tensor", "BenchmarkMetalOptimizerTensor_LAMB", "TestTensorBackend_applyOptimizerGraph"),
	residentOperation("train.optimizer.adagrad", "metal_optimizer_adagrad_tensor", "BenchmarkMetalOptimizerTensor_AdaGrad", "TestTensorBackend_applyOptimizerGraph"),
	residentOperation("train.optimizer.adadelta", "metal_optimizer_adadelta_tensor", "BenchmarkMetalOptimizerTensor_AdaDelta", "TestTensorBackend_applyOptimizerGraph"),
	residentOperation("train.optimizer.lbfgs", "metal_optimizer_lbfgs_tensor", "BenchmarkMetalOptimizerTensor_LBFGS", "TestTensorBackend_applyOptimizerGraph"),
}

/*
ResidentMetalOperationTable returns a copy of the declared resident Metal operation table.
*/
func ResidentMetalOperationTable() []MetalOperationCoverage {
	table := make([]MetalOperationCoverage, len(residentMetalOperationTable))

	for index, operation := range residentMetalOperationTable {
		table[index] = cloneMetalOperationCoverage(operation)
	}

	return table
}

/*
ResidentMetalOperationIDs returns operation IDs that have resident Metal tensor coverage.
*/
func ResidentMetalOperationIDs() []ir.OpType {
	operationIDs := make([]ir.OpType, len(residentMetalOperationTable))

	for index, operation := range residentMetalOperationTable {
		operationIDs[index] = operation.ID
	}

	return operationIDs
}

/*
ResidentMetalOperationByID returns resident coverage metadata for one operation.
*/
func ResidentMetalOperationByID(operationID ir.OpType) (MetalOperationCoverage, bool) {
	for _, operation := range residentMetalOperationTable {
		if operation.ID != operationID {
			continue
		}

		return cloneMetalOperationCoverage(operation), true
	}

	return MetalOperationCoverage{}, false
}

func residentOperation(
	operationID ir.OpType,
	symbol string,
	benchmarkName string,
	parityTestName string,
) MetalOperationCoverage {
	return MetalOperationCoverage{
		ID:             operationID,
		DTypes:         []dtype.DType{dtype.Float32},
		ResidentSymbol: symbol,
		BenchmarkName:  benchmarkName,
		ParityTestName: parityTestName,
	}
}

func residentOperationConstraints(
	operationID ir.OpType,
	symbol string,
	benchmarkName string,
	parityTestName string,
	constraints ...string,
) MetalOperationCoverage {
	coverage := residentOperation(operationID, symbol, benchmarkName, parityTestName)
	coverage.ShapeConstraints = slices.Clone(constraints)

	return coverage
}

func residentFusion(
	operationID ir.OpType,
	symbol string,
	fusionGroup string,
	benchmarkName string,
	parityTestName string,
) MetalOperationCoverage {
	coverage := residentOperation(operationID, symbol, benchmarkName, parityTestName)
	coverage.FusionGroups = []string{fusionGroup}

	return coverage
}

func cloneMetalOperationCoverage(operation MetalOperationCoverage) MetalOperationCoverage {
	operation.DTypes = slices.Clone(operation.DTypes)
	operation.ShapeConstraints = slices.Clone(operation.ShapeConstraints)
	operation.FusionGroups = slices.Clone(operation.FusionGroups)

	return operation
}
