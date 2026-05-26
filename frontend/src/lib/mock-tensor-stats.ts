/*
mock-tensor-stats fabricates plausible weight statistics so the inspector
UI can be exercised before real backend tensor-byte access exists.

Every value is deterministic in the (name, dtype, shape) triple via a
seeded PRNG, so visualisations don't jiggle between renders and node
selection feels stable.

Replace getMockTensorStats() with a real backend call later — keep the
returned shape identical and downstream panels keep working.
*/

const MOD = 0x100000000;

/*
mulberry32 is a tiny seeded PRNG. Returns a function that yields the
next pseudo-random float in [0, 1).
*/
const mulberry32 = (seed: number) => {
	let state = seed >>> 0;

	return () => {
		state = (state + 0x6d2b79f5) >>> 0;
		let t = state;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / MOD;
	};
};

const hashString = (input: string): number => {
	let hash = 5381;

	for (let index = 0; index < input.length; index++) {
		hash = ((hash << 5) + hash + input.charCodeAt(index)) >>> 0;
	}

	return hash;
};

export type TensorStats = {
	name: string;
	dtype: string;
	shape: ReadonlyArray<number>;
	parameters: number;
	l2Norm: number;
	mean: number;
	std: number;
	min: number;
	max: number;
	sparsity: number;
	effectiveRank: number | null;
	histogram: ReadonlyArray<{ bin: number; count: number }>;
	heatmap: { rows: number; cols: number; values: Float32Array };
	topActivatingTokens: ReadonlyArray<{ token: string; activation: number }>;
};

const MOCK_TOKENS = [
	"the", " of", " and", " to", " a", "ing", " is", " in", " for", " was",
	" with", " on", " as", " by", " he", " she", " they", " it", " be", " an",
	"\\n", " new", " people", " world", " time", " way",
];

const HEATMAP_MAX = 64;

export const getMockTensorStats = (
	name: string,
	dtype: string,
	shape: ReadonlyArray<number>,
): TensorStats => {
	const seed = hashString(`${name}|${dtype}|${shape.join("x")}`);
	const rng = mulberry32(seed);

	const parameters = shape.reduce((acc, dim) => acc * dim, 1) || 1;

	// Pick a centered Gaussian-ish distribution. Std scales mildly with rank.
	const std = 0.015 + rng() * 0.04;
	const mean = (rng() - 0.5) * 0.002;
	const sparsity = dtype.startsWith("Q") ? rng() * 0.1 : rng() * 0.6;

	const sampleCount = 8192;
	const samples = new Float32Array(sampleCount);
	let l2Sq = 0;
	let min = Number.POSITIVE_INFINITY;
	let max = Number.NEGATIVE_INFINITY;

	for (let index = 0; index < sampleCount; index++) {
		// Box-Muller for a normal sample.
		const u1 = Math.max(rng(), 1e-9);
		const u2 = rng();
		const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
		const value = rng() < sparsity ? 0 : mean + z * std;
		samples[index] = value;
		l2Sq += value * value;
		if (value < min) min = value;
		if (value > max) max = value;
	}

	const l2Norm = Math.sqrt(l2Sq * (parameters / sampleCount));

	const binCount = 32;
	const bins = new Array(binCount).fill(0);
	const range = Math.max(max - min, 1e-6);

	for (let index = 0; index < sampleCount; index++) {
		const normalized = (samples[index] - min) / range;
		const bin = Math.min(binCount - 1, Math.floor(normalized * binCount));
		bins[bin] += 1;
	}

	const histogram = bins.map((count, bin) => ({ bin, count }));

	const rank = shape.length;
	const effectiveRank =
		rank >= 2
			? Math.round(Math.min(shape[0], shape[1]) * (0.55 + rng() * 0.4))
			: null;

	const rows = Math.min(shape[0] ?? 1, HEATMAP_MAX);
	const cols = Math.min(shape[1] ?? 1, HEATMAP_MAX);
	const heatmapValues = new Float32Array(rows * cols);

	for (let row = 0; row < rows; row++) {
		for (let col = 0; col < cols; col++) {
			const u1 = Math.max(rng(), 1e-9);
			const u2 = rng();
			const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
			heatmapValues[row * cols + col] = rng() < sparsity ? 0 : z;
		}
	}

	const topActivatingTokens = Array.from({ length: 8 }, (_, index) => {
		const token = MOCK_TOKENS[(hashString(name) + index) % MOCK_TOKENS.length];
		const activation = 0.9 - index * 0.08 - rng() * 0.05;
		return { token, activation: Math.max(0.1, activation) };
	});

	return {
		name,
		dtype,
		shape,
		parameters,
		l2Norm,
		mean,
		std,
		min,
		max,
		sparsity,
		effectiveRank,
		histogram,
		heatmap: { rows, cols, values: heatmapValues },
		topActivatingTokens,
	};
};

/*
getMockAttentionPattern fabricates a heads × tokens × tokens attention
matrix per (layer, token-count) pair. Encodes a mix of diagonal,
previous-token and uniform patterns so the rendered cells look like
plausible Transformer attention heads.
*/
export type AttentionPattern = {
	layers: number;
	heads: number;
	tokens: ReadonlyArray<string>;
	// Flat [layer][head][i][j] = matrix[layer*heads + head][i*tokens + j].
	matrix: Float32Array;
};

export const getMockAttentionPattern = (
	prompt: string,
	layers: number,
	heads: number,
): AttentionPattern => {
	const tokens =
		prompt.trim().length === 0
			? ["<bos>"]
			: prompt.trim().split(/\s+/).slice(0, 16);

	const tokenCount = tokens.length;
	const matrix = new Float32Array(layers * heads * tokenCount * tokenCount);
	const stride = tokenCount * tokenCount;
	const seed = hashString(`${prompt}|${layers}|${heads}`);
	const rng = mulberry32(seed);

	for (let layer = 0; layer < layers; layer++) {
		for (let head = 0; head < heads; head++) {
			const headIndex = layer * heads + head;
			const offset = headIndex * stride;
			const flavor = rng();
			// 0 = diagonal, 1 = prev-token, 2 = uniform, 3 = induction-ish
			const pattern = Math.floor(flavor * 4);

			for (let i = 0; i < tokenCount; i++) {
				let rowSum = 0;
				const rowStart = offset + i * tokenCount;

				for (let j = 0; j <= i; j++) {
					let weight = 0;

					if (pattern === 0) {
						weight = i === j ? 1 : 0.05 + rng() * 0.05;
					} else if (pattern === 1) {
						weight = j === i - 1 ? 1 : 0.05 + rng() * 0.05;
					} else if (pattern === 2) {
						weight = 0.5 + rng() * 0.5;
					} else {
						weight =
							i > 1 && j === i - 2 ? 1 : j === i ? 0.4 : 0.05 + rng() * 0.05;
					}

					matrix[rowStart + j] = weight;
					rowSum += weight;
				}

				if (rowSum > 0) {
					for (let j = 0; j <= i; j++) {
						matrix[rowStart + j] = matrix[rowStart + j] / rowSum;
					}
				}
			}
		}
	}

	return { layers, heads, tokens, matrix };
};
