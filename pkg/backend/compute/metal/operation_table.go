package metal

import (
	"slices"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

/*
OperationCoverage is the resident Metal operation contract consumed by lowering,
tests, benchmarks, and startup validation.
*/
type OperationCoverage struct {
	ID               ir.OpType
	DTypes           []tensor.DType
	ShapeConstraints []string
	ResidentSymbol   string
	FusionGroups     []string
	BenchmarkName    string
	ParityTestName   string
}

var residentOperationTable = []OperationCoverage{
	residentOperation(ir.OpInput, "metal_tensor_upload_float32_mode", "BenchmarkTensorBackend_UploadFloat64", "TestTensorBackend_UploadFloat64"),
	residentOperation(ir.OpAdd, "metal_add_tensor", "BenchmarkMathOps_AddTensor", "TestMathOps_AddTensor"),
	residentOperation(ir.OpMul, "metal_mul_tensor", "BenchmarkMathOps_MulTensor", "TestMathOps_MulTensor"),
	residentOperation(ir.OpMatmul, "metal_matmul_tensor", "BenchmarkMathOps_MatmulTensor", "TestMathOps_MatmulTensor"),
	residentOperation(ir.OpReLU, "metal_relu_tensor", "BenchmarkMetalActivation_ReLUTensor", "TestMetalActivation_ReLUTensor"),
	residentOperation(ir.OpLeakyReLU, "metal_leaky_relu_tensor", "BenchmarkMetalActivation_LeakyReLUTensor", "TestMetalActivation_LeakyReLUTensor"),
	residentOperation(ir.OpGELU, "metal_gelu_tensor", "BenchmarkMetalActivation_GELUTensor", "TestMetalActivation_GELUTensor"),
	residentOperation(ir.OpTanh, "metal_tanh_tensor", "BenchmarkMetalActivation_TanhTensor", "TestMetalActivation_TanhTensor"),
	residentOperation(ir.OpSigmoid, "metal_sigmoid_tensor", "BenchmarkMetalActivation_SigmoidTensor", "TestMetalActivation_SigmoidTensor"),
	residentOperation(ir.OpSwiGLU, "metal_swiglu_tensor", "BenchmarkMetalActivation_SwiGLUTensor", "TestMetalActivation_SwiGLUTensor"),
	residentOperation(ir.OpSwish, "metal_swish_tensor", "BenchmarkMetalActivation_SwishTensor", "TestMetalActivation_SwishTensor"),
	residentOperation(ir.OpSELU, "metal_selu_tensor", "BenchmarkMetalActivation_SELUTensor", "TestMetalActivation_SELUTensor"),
	residentFusion(ir.OpFused, "metal_matmul_add_tensor", "matmul.activation", "BenchmarkMathOps_MatmulAddGELUTensor", "TestMathOps_MatmulAddGELUTensor"),
	residentOperation("math.add", "metal_add_tensor", "BenchmarkMathOps_AddTensor", "TestMathOps_AddTensor"),
	residentOperation("math.mul", "metal_mul_tensor", "BenchmarkMathOps_MulTensor", "TestMathOps_MulTensor"),
	residentOperation("math.matmul", "metal_matmul_tensor", "BenchmarkMathOps_MatmulTensor", "TestMathOps_MatmulTensor"),
	residentOperation("math.rmsnorm", "metal_rmsnorm_tensor", "BenchmarkMathOps_RMSNormTensor", "TestMathOps_RMSNormTensor"),
	residentOperation("math.layernorm", "metal_layernorm_tensor", "BenchmarkMathOps_LayerNormTensor", "TestMathOps_LayerNormTensor"),
	residentOperation("math.groupnorm", "metal_groupnorm_tensor", "BenchmarkMathOps_GroupNormTensor", "TestMathOps_GroupNormTensor"),
	residentOperation("activation.relu", "metal_relu_tensor", "BenchmarkMetalActivation_ReLUTensor", "TestMetalActivation_ReLUTensor"),
	residentOperation("activation.leaky_relu", "metal_leaky_relu_tensor", "BenchmarkMetalActivation_LeakyReLUTensor", "TestMetalActivation_LeakyReLUTensor"),
	residentOperation("activation.gelu", "metal_gelu_tensor", "BenchmarkMetalActivation_GELUTensor", "TestMetalActivation_GELUTensor"),
	residentOperation("activation.tanh", "metal_tanh_tensor", "BenchmarkMetalActivation_TanhTensor", "TestMetalActivation_TanhTensor"),
	residentOperation("activation.sigmoid", "metal_sigmoid_tensor", "BenchmarkMetalActivation_SigmoidTensor", "TestMetalActivation_SigmoidTensor"),
	residentOperation("activation.swiglu", "metal_swiglu_tensor", "BenchmarkMetalActivation_SwiGLUTensor", "TestMetalActivation_SwiGLUTensor"),
	residentOperation("activation.swish", "metal_swish_tensor", "BenchmarkMetalActivation_SwishTensor", "TestMetalActivation_SwishTensor"),
	residentOperation("activation.selu", "metal_selu_tensor", "BenchmarkMetalActivation_SELUTensor", "TestMetalActivation_SELUTensor"),
	residentOperation("embedding.token", "metal_token_embedding_tensor", "BenchmarkEmbeddingOps_ForwardTensor", "TestEmbeddingOps_ForwardTensor"),
	residentOperation("shape.reshape", "metal_copy_tensor", "BenchmarkMetalShapeOps_CopyTensor", "TestMetalShapeOps_CopyTensor"),
	residentOperation("shape.transpose", "metal_transpose_tensor", "BenchmarkMetalShapeOps_TransposeTensor", "TestMetalShapeOps_TransposeTensor"),
	residentOperation("shape.concat", "metal_concat_tensor", "BenchmarkMetalShapeOps_ConcatTensor", "TestMetalShapeOps_ConcatTensor"),
	residentOperation("shape.split", "metal_split_tensor", "BenchmarkMetalShapeOps_SplitTensor", "TestMetalShapeOps_SplitTensor"),
	residentOperation("shape.upsample_nearest2d", "metal_upsample_nearest2d_tensor", "BenchmarkMetalShapeOps_UpsampleNearest2DTensor", "TestMetalShapeOps_UpsampleNearest2DTensor"),
	residentOperation("shape.view_as_heads", "metal_view_as_heads_tensor", "BenchmarkMetalShapeOps_ViewAsHeadsTensor", "TestMetalShapeOps_ViewAsHeadsTensor"),
	residentOperation("shape.merge_heads", "metal_merge_heads_tensor", "BenchmarkMetalShapeOps_MergeHeadsTensor", "TestMetalShapeOps_MergeHeadsTensor"),
	residentOperation("shape.last_token", "metal_last_token_tensor", "BenchmarkMetalShapeOps_LastTokenTensor", "TestMetalShapeOps_LastTokenTensor"),
	residentOperation("projection.linear", "metal_matmul_add_tensor", "BenchmarkMathOps_MatmulAddFlatTensor", "TestMathOps_MatmulAddFlatTensor"),
	residentOperation("attention.sdpa", "metal_sdpa_tensor", "BenchmarkMetalAttention_SDPATensor", "TestMetalAttention_SDPATensor"),
	residentOperation("attention.gqa", "metal_gqa_tensor", "BenchmarkMetalAttention_GQATensor", "TestMetalAttention_GQATensor"),
	residentOperation("positional.rope", "metal_rope_tensor", "BenchmarkMetalPositional_RoPETensor", "TestMetalPositional_RoPETensor"),
	residentOperation("convolution.conv1d", "metal_conv1d_tensor", "BenchmarkConvolutionOps_Conv1dTensor", "TestConvolutionOps_Conv1dTensor"),
	residentOperation("convolution.conv2d", "metal_conv2d_tensor", "BenchmarkConvolutionOps_Conv2dTensor", "TestConvolutionOps_Conv2dTensor"),
	residentOperation("convolution.conv3d", "metal_conv3d_tensor", "BenchmarkConvolutionOps_Conv3dTensor", "TestConvolutionOps_Conv3dTensor"),
	residentOperation("convolution.conv_transpose2d", "metal_conv_transpose2d_tensor", "BenchmarkConvolutionOps_ConvTranspose2dTensor", "TestConvolutionOps_ConvTranspose2dTensor"),
}

/*
ResidentOperationTable returns a copy of the declared resident Metal operation table.
*/
func ResidentOperationTable() []OperationCoverage {
	table := make([]OperationCoverage, len(residentOperationTable))

	for index, operation := range residentOperationTable {
		table[index] = cloneOperationCoverage(operation)
	}

	return table
}

/*
ResidentOperationIDs returns operation IDs that have resident Metal tensor coverage.
*/
func ResidentOperationIDs() []ir.OpType {
	operationIDs := make([]ir.OpType, len(residentOperationTable))

	for index, operation := range residentOperationTable {
		operationIDs[index] = operation.ID
	}

	return operationIDs
}

/*
ResidentOperationByID returns resident coverage metadata for one operation.
*/
func ResidentOperationByID(operationID ir.OpType) (OperationCoverage, bool) {
	for _, operation := range residentOperationTable {
		if operation.ID != operationID {
			continue
		}

		return cloneOperationCoverage(operation), true
	}

	return OperationCoverage{}, false
}

func residentOperation(
	operationID ir.OpType,
	symbol string,
	benchmarkName string,
	parityTestName string,
) OperationCoverage {
	return OperationCoverage{
		ID:             operationID,
		DTypes:         []tensor.DType{tensor.Float32},
		ResidentSymbol: symbol,
		BenchmarkName:  benchmarkName,
		ParityTestName: parityTestName,
	}
}

func residentFusion(
	operationID ir.OpType,
	symbol string,
	fusionGroup string,
	benchmarkName string,
	parityTestName string,
) OperationCoverage {
	coverage := residentOperation(operationID, symbol, benchmarkName, parityTestName)
	coverage.FusionGroups = []string{fusionGroup}

	return coverage
}

func cloneOperationCoverage(operation OperationCoverage) OperationCoverage {
	operation.DTypes = slices.Clone(operation.DTypes)
	operation.ShapeConstraints = slices.Clone(operation.ShapeConstraints)
	operation.FusionGroups = slices.Clone(operation.FusionGroups)

	return operation
}
