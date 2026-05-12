/*
attention-recordings — type definitions for attention weight recordings
captured from transformer model forward passes.
*/

export type HeadMode =
	| { kind: "mean" }
	| { kind: "head"; index: number };

export type AttnMatrix = number[][];

export type LayerAttn = {
	matrices: AttnMatrix[];
};

export type LayerAct = {
	values: number[];
};

export type LayerRecording = {
	attn?: LayerAttn;
	act?: LayerAct;
};

export type AttnRecording = {
	layers: LayerRecording[];
};
