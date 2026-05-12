package config

var computeRootKey = "compute"

type ComputeConfig struct {
	Metal MetalConfig
}

type MetalConfig struct {
	ActivationMetallib       string
	ActiveInferenceMetallib  string
	AttentionMetallib        string
	CausalMetallib           string
	ConvolutionMetallib      string
	EmbeddingMetallib        string
	HawkesMetallib           string
	MarkovBlanketMetallib    string
	MaskingMetallib          string
	MathMetallib             string
	PoolingMetallib          string
	PositionalMetallib       string
	PredictiveCodingMetallib string
	ProjectionMetallib       string
	ShapeMetallib            string
	VSAMetallib              string
}

func NewComputeConfig() *ComputeConfig {
	return &ComputeConfig{
		Metal: MetalConfig{
			ActivationMetallib: WithDefault(
				computeRootKey+".metal.activation_metallib",
				"activation.metallib",
			),
			ActiveInferenceMetallib: WithDefault(
				computeRootKey+".metal.active_inference_metallib",
				"active_inference.metallib",
			),
			AttentionMetallib: WithDefault(
				computeRootKey+".metal.attention_metallib",
				"attention.metallib",
			),
			CausalMetallib: WithDefault(
				computeRootKey+".metal.causal_metallib",
				"causal.metallib",
			),
			ConvolutionMetallib: WithDefault(
				computeRootKey+".metal.convolution_metallib",
				"convolution.metallib",
			),
			EmbeddingMetallib: WithDefault(
				computeRootKey+".metal.embedding_metallib",
				"embedding.metallib",
			),
			HawkesMetallib: WithDefault(
				computeRootKey+".metal.hawkes_metallib",
				"hawkes.metallib",
			),
			MarkovBlanketMetallib: WithDefault(
				computeRootKey+".metal.markov_blanket_metallib",
				"markov_blanket.metallib",
			),
			MaskingMetallib: WithDefault(
				computeRootKey+".metal.masking_metallib",
				"masking.metallib",
			),
			MathMetallib: WithDefault(
				computeRootKey+".metal.math_metallib",
				"math.metallib",
			),
			PoolingMetallib: WithDefault(
				computeRootKey+".metal.pooling_metallib",
				"pooling.metallib",
			),
			PositionalMetallib: WithDefault(
				computeRootKey+".metal.positional_metallib",
				"positional.metallib",
			),
			PredictiveCodingMetallib: WithDefault(
				computeRootKey+".metal.predictive_coding_metallib",
				"predictive_coding.metallib",
			),
			ProjectionMetallib: WithDefault(
				computeRootKey+".metal.projection_metallib",
				"projection.metallib",
			),
			ShapeMetallib: WithDefault(
				computeRootKey+".metal.shape_metallib",
				"shape.metallib",
			),
			VSAMetallib: WithDefault(
				computeRootKey+".metal.vsa_metallib",
				"vsa.metallib",
			),
		},
	}
}
