package device

import "github.com/theapemachine/manifesto/ir"

func operationCrossLinks() []OperationCrossLink {
	return []OperationCrossLink{
		graphLink(ir.OpInput, "graph input binding; no compute kernel"),
		directLink(ir.OpAdd, "Elementwise", "Add"),
		directLink(ir.OpMul, "Elementwise", "Mul"),
		directLink(ir.OpMatmul, "Matmul", "Matmul"),
		directLink(ir.OpReLU, "Activation", "ReLU"),
		directLink(ir.OpLeakyReLU, "Activation", "LeakyReLU"),
		directLink(ir.OpGELU, "Activation", "Gelu"),
		directLink(ir.OpTanh, "Activation", "Tanh"),
		directLink(ir.OpSigmoid, "Activation", "Sigmoid"),
		directLink(ir.OpSwiGLU, "Activation", "SwiGLU"),
		directLink(ir.OpSwish, "Activation", "Swish"),
		directLink(ir.OpSELU, "Activation", "SELU"),
		graphLink(ir.OpFused, "fusion pass; composite of registered primitives"),

		directLink("activation.relu", "Activation", "ReLU"),
		directLink("activation.leaky_relu", "Activation", "LeakyReLU"),
		directLink("activation.gelu", "Activation", "Gelu"),
		directLink("activation.tanh", "Activation", "Tanh"),
		directLink("activation.sigmoid", "Activation", "Sigmoid"),
		directLink("activation.swiglu", "Activation", "SwiGLU"),
		directLink("activation.swish", "Activation", "Swish"),
		directLink("activation.selu", "Activation", "SELU"),

		directLink("attention.sdpa", "Attention", "ScaledDotProductAttention"),
		compositeLink(
			"attention.mqa",
			MethodRef{"Attention", "MultiHeadAttention"},
			MethodRef{"Attention", "FlashAttention"},
		),
		compositeLink(
			"attention.gqa",
			MethodRef{"Attention", "MultiHeadAttention"},
			MethodRef{"Attention", "FlashAttention"},
		),
		directLink("attention.sliding_window", "Attention", "FlashAttention"),

		directLink("masking.apply", "Masking", "ApplyMask"),
		directLink("masking.causal", "Masking", "CausalMask"),

		directLink("math.add", "Elementwise", "Add"),
		directLink("math.mul", "Elementwise", "Mul"),
		directLink("math.matmul", "Matmul", "Matmul"),
		directLink("math.exp", "Activation", "Exp"),
		registryLink("math.sin", "pkg/backend/device/cpu/elementwise kernel registry"),
		registryLink("math.cos", "pkg/backend/device/cpu/elementwise kernel registry"),
		directLink("math.log", "Activation", "Log"),
		registryLink("math.logsumexp", "pkg/backend/device/cpu/reduction kernel registry"),
		directLink("math.softmax", "Activation", "Softmax"),
		registryLink("math.outer", "pkg/backend/device/cpu/matmul kernel registry"),
		registryLink("math.sign", "pkg/backend/device/cpu/elementwise kernel registry"),
		registryLink("math.inv_sqrt_dim_scale", "pkg/backend/compute/kernels graph lowering"),
		directLink("math.dropout", "Dropout", "Dropout"),
		directLink("math.rmsnorm", "LayerNorm", "RMSNorm"),
		directLink("math.layernorm", "LayerNorm", "LayerNorm"),
		directLink("math.groupnorm", "Normalization", "GroupNorm"),

		registryLink("shape.reshape", "pkg/backend/device/cpu/shape"),
		registryLink("shape.transpose", "pkg/backend/device/cpu/shape"),
		registryLink("shape.concat", "pkg/backend/device/cpu/shape"),
		registryLink("shape.split", "pkg/backend/device/cpu/shape"),
		registryLink("shape.upsample_nearest2d", "pkg/backend/device/cpu/shape"),
		registryLink("shape.view_as_heads", "pkg/backend/device/cpu/shape"),
		registryLink("shape.merge_heads", "pkg/backend/device/cpu/shape"),
		registryLink("shape.last_token", "pkg/backend/device/cpu/shape"),
		registryLink("shape.slice", "pkg/backend/device/cpu/shape"),

		directLink("positional.rope", "RoPE", "RoPE"),
		directLink("positional.alibi", "Masking", "ALiBiBias"),

		directLink("embedding.token", "Embedding", "Lookup"),

		directLink("convolution.conv1d", "Convolution", "Conv1D"),
		directLink("convolution.conv2d", "Convolution", "Conv2D"),
		directLink("convolution.conv3d", "Convolution", "Conv3D"),
		directLink("convolution.conv_transpose2d", "Convolution", "ConvTranspose2D"),

		directLink("pooling.max_pool2d", "Pool", "MaxPool2D"),
		directLink("pooling.avg_pool2d", "Pool", "AvgPool2D"),
		directLink("pooling.adaptive_avg_pool2d", "Pool", "AdaptiveAvgPool2D"),
		directLink("pooling.adaptive_max_pool2d", "Pool", "AdaptiveMaxPool2D"),

		directLink("projection.linear", "Matmul", "Matmul"),
		compositeLink(
			"projection.fused_qkv",
			MethodRef{"Matmul", "Matmul"},
			MethodRef{"Attention", "MultiHeadAttention"},
		),

		directLink("hawkes.intensity", "Hawkes", "HawkesIntensity"),
		directLink("hawkes.kernel_matrix", "Hawkes", "HawkesKernelMatrix"),
		directLink("hawkes.log_likelihood", "Hawkes", "HawkesLogLikelihood"),
		registryLink("hawkes.simulate", "pkg/backend/device/cpu/hawkes; not on device.Backend"),

		directLink("vsa.bind", "VSA", "Bind"),
		directLink("vsa.bundle", "VSA", "Bundle"),
		directLink("vsa.similarity", "VSA", "Similarity"),
		directLink("vsa.permute", "VSA", "Permute"),
		directLink("vsa.inverse_permute", "VSA", "InversePermute"),

		directLink("active_inference.belief_update", "ActiveInference", "BeliefUpdate"),
		directLink("active_inference.expected_free_energy", "ActiveInference", "ExpectedFreeEnergy"),
		directLink("active_inference.free_energy", "ActiveInference", "FreeEnergy"),
		directLink("active_inference.precision_weight", "ActiveInference", "PrecisionWeight"),

		directLink("predictive_coding.prediction", "PredictiveCoding", "Prediction"),
		directLink("predictive_coding.prediction_error", "PredictiveCoding", "PredictionError"),
		directLink("predictive_coding.update_representation", "PredictiveCoding", "UpdateRepresentation"),
		directLink("predictive_coding.update_weights", "PredictiveCoding", "UpdateWeights"),

		directLink("markov_blanket.flow_active", "Causal", "MarkovFlowActive"),
		directLink("markov_blanket.flow_internal", "Causal", "MarkovFlowInternal"),
		directLink("markov_blanket.mutual_information", "Hawkes", "MarkovMutualInformation"),
		directLink("markov_blanket.partition", "Hawkes", "MarkovBlanketPartition"),

		directLink("causal.backdoor_adjustment", "Causal", "BackdoorAdjustment"),
		directLink("causal.cate", "Causal", "CATE"),
		directLink("causal.counterfactual", "Causal", "Counterfactual"),
		directLink("causal.dag_markov_factorization", "Causal", "DAGMarkovFactorization"),
		directLink("causal.do_calculus", "Causal", "DoIntervene"),
		directLink("causal.frontdoor_adjustment", "Causal", "FrontdoorAdjustment"),
		directLink("causal.iv_estimate", "Causal", "IVEstimate"),

		directLink("train.loss.mse", "Losses", "MSE"),
		directLink("train.loss.cross_entropy", "Losses", "CrossEntropy"),
		registryLink("train.loss.mse_grad", "pkg/backend/device/cpu/losses kernel registry"),
		registryLink("train.loss.cross_entropy_grad", "pkg/backend/device/cpu/losses kernel registry"),
		registryLink("train.grad.mse", "pkg/backend/device/cpu/losses kernel registry"),
		registryLink("train.grad.cross_entropy", "pkg/backend/device/cpu/losses kernel registry"),

		registryLink("train.optimizer.adam", "pkg/backend/device/cpu/optimizer; pkg/backend/device/metal optimizer kernels"),
		registryLink("train.optimizer.adamw", "pkg/backend/device/cpu/optimizer; pkg/backend/device/metal optimizer kernels"),
		registryLink("train.optimizer.adamax", "pkg/backend/device/cpu/optimizer; pkg/backend/device/metal optimizer kernels"),
		registryLink("train.optimizer.sgd", "pkg/backend/device/cpu/optimizer; pkg/backend/device/metal optimizer kernels"),
		registryLink("train.optimizer.lion", "pkg/backend/device/cpu/optimizer; pkg/backend/device/metal optimizer kernels"),
		registryLink("train.optimizer.rmsprop", "pkg/backend/device/cpu/optimizer; pkg/backend/device/metal optimizer kernels"),
		registryLink("train.optimizer.hebbian", "pkg/backend/device/cpu/optimizer; pkg/backend/device/metal optimizer kernels"),
		registryLink("train.optimizer.lars", "pkg/backend/device/cpu/optimizer; pkg/backend/device/metal optimizer kernels"),
		registryLink("train.optimizer.lamb", "pkg/backend/device/cpu/optimizer kernel registry"),
		registryLink("train.optimizer.adagrad", "pkg/backend/device/cpu/optimizer; pkg/backend/device/metal optimizer kernels"),
		registryLink("train.optimizer.adadelta", "pkg/backend/device/cpu/optimizer kernel registry"),
		registryLink("train.optimizer.lbfgs", "pkg/backend/device/cpu/optimizer; pkg/backend/device/metal optimizer kernels"),

		graphLink("bench.accuracy", "evaluation metric; host graph node"),
		graphLink("bench.perplexity", "evaluation metric; host graph node"),
		graphLink("bench.f1", "evaluation metric; host graph node"),
		graphLink("bench.metric.accuracy", "evaluation metric; host graph node"),
		graphLink("bench.metric.perplexity", "evaluation metric; host graph node"),
		graphLink("bench.metric.f1", "evaluation metric; host graph node"),

		graphLink("model.graft", "model-editing graph op; not on device.Backend"),
		graphLink("model.freeze", "model-editing graph op; not on device.Backend"),
	}
}
