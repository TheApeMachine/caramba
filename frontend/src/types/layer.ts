export type LayerConfig = {
	name: string;
	type: string;
	layers: Array<LayerConfig>;
	blockNum?: number;
};

// Activation types
export type AttentionActivation = {
	patterns: Array<Array<Array<number>>>;
	output: Array<Array<number>>;
};

export type FFNActivation = {
	hidden: Array<Array<number>>;
	output: Array<Array<number>>;
};

export type SimpleActivation = Array<Array<number>>;

export type Activation = AttentionActivation | FFNActivation | SimpleActivation;

// Type guard functions
export const isFFNActivation = (act: Activation): act is FFNActivation => {
	return "hidden" in act && "output" in act && !Array.isArray(act);
}

export const isAttentionActivation = (act: Activation): act is AttentionActivation => {
	return "patterns" in act && "output" in act && !Array.isArray(act);
}

// Layer config types with specific properties
export type AttentionLayerConfig = LayerConfig & {
	type: "attention";
	heads: number;
	head_dim: number;
	dim: number;
	blockNum?: number;
};

export type FFNLayerConfig = LayerConfig & {
	type: "ffn";
	in_dim: number;
	hidden_dim: number;
	out_dim: number;
	blockNum?: number;
};

export type EmbeddingLayerConfig = LayerConfig & {
	type: "embedding";
	in_dim: number;
	out_dim: number;
};

export type LayerNormLayerConfig = LayerConfig & {
	type: "layernorm";
	dim: number;
	blockNum?: number;
};

export type LinearLayerConfig = LayerConfig & {
	type: "linear";
	in_dim: number;
	out_dim: number;
};

export type TypedLayerConfig =
	| AttentionLayerConfig
	| FFNLayerConfig
	| EmbeddingLayerConfig
	| LayerNormLayerConfig
	| LinearLayerConfig;

// Visual block types
export type VisualBlock = {
	name: string;
	type: "embedding" | "transformer" | "output";
	layers: Array<TypedLayerConfig>;
	blockNum?: number;
};