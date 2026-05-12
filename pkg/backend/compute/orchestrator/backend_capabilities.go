package orchestrator

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func CapabilitiesForLocation(location tensor.Location) Capabilities {
	switch location {
	case tensor.Host:
		return hostCapabilities()
	case tensor.CUDA, tensor.XLA:
		return acceleratorTensorCapabilities(location)
	case tensor.Metal:
		return metalCapabilities()
	default:
		return NewDefaultCapabilities(location)
	}
}

func hostCapabilities() *StaticCapabilities {
	capabilities := NewDefaultCapabilities(tensor.Host)
	capabilities.Register("*")

	return capabilities
}

func acceleratorTensorCapabilities(location tensor.Location) *StaticCapabilities {
	capabilities := NewStaticCapabilities(location)
	capabilities.Register(
		ir.OpInput,
		ir.OpAdd,
		ir.OpMul,
		ir.OpMatmul,
		ir.OpReLU,
		ir.OpLeakyReLU,
		ir.OpGELU,
		ir.OpTanh,
		ir.OpSigmoid,
		ir.OpSwiGLU,
		ir.OpFused,
	)
	capabilities.RegisterFusion("matmul.activation", ir.OpFused)

	return capabilities
}

func metalCapabilities() *StaticCapabilities {
	capabilities := acceleratorTensorCapabilities(tensor.Metal)
	capabilities.Register(
		"active_inference.belief_update",
		"active_inference.expected_free_energy",
		"active_inference.free_energy",
		"active_inference.precision_weight",
		"attention.sdpa",
		"attention.mqa",
		"attention.gqa",
		"attention.sliding_window",
		"causal.backdoor_adjustment",
		"causal.cate",
		"causal.dag_markov_factorization",
		"causal.do_calculus",
		"causal.iv_estimate",
		"convolution.conv1d",
		"convolution.conv2d",
		"convolution.conv3d",
		"convolution.conv_transpose2d",
		"hawkes.intensity",
		"hawkes.kernel_matrix",
		"hawkes.log_likelihood",
		"hawkes.simulate",
		"markov_blanket.flow_active",
		"markov_blanket.flow_internal",
		"markov_blanket.mutual_information",
		"markov_blanket.partition",
		"masking.apply",
		"masking.causal",
		"pooling.adaptive_avg_pool2d",
		"pooling.adaptive_max_pool2d",
		"pooling.avg_pool2d",
		"pooling.max_pool2d",
		"positional.alibi",
		"positional.rope",
		"predictive_coding.prediction",
		"predictive_coding.prediction_error",
		"predictive_coding.update_representation",
		"predictive_coding.update_weights",
		"projection.fused_qkv",
		"projection.linear",
		"shape.concat",
		"shape.merge_heads",
		"shape.reshape",
		"shape.split",
		"shape.transpose",
		"shape.view_as_heads",
		"vsa.bind",
		"vsa.bundle",
		"vsa.similarity",
	)

	return capabilities
}
