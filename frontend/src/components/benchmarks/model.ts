/*
Backend targets the platform supports. Mirrors the backends listed in the
AGENTS.md execution contract; the wizard exposes the same set so researchers
launch against the same names they reason about in code.
*/
export type Backend =
	| "go-scalar"
	| "avx2"
	| "sse2"
	| "neon"
	| "metal"
	| "cuda"
	| "xla";

export interface BackendInfo {
	id: Backend;
	label: string;
	arch: "amd64" | "arm64" | "any";
	kind: "cpu" | "gpu" | "compiler";
	hint: string;
}

export const BACKENDS: BackendInfo[] = [
	{
		id: "go-scalar",
		label: "Go (scalar)",
		arch: "any",
		kind: "cpu",
		hint: "Reference implementation; used as the parity target.",
	},
	{
		id: "avx2",
		label: "AVX2",
		arch: "amd64",
		kind: "cpu",
		hint: "256-bit SIMD on x86_64.",
	},
	{
		id: "sse2",
		label: "SSE2",
		arch: "amd64",
		kind: "cpu",
		hint: "128-bit SIMD baseline on x86_64.",
	},
	{
		id: "neon",
		label: "NEON",
		arch: "arm64",
		kind: "cpu",
		hint: "128-bit SIMD on Apple Silicon and ARM servers.",
	},
	{
		id: "metal",
		label: "Metal",
		arch: "arm64",
		kind: "gpu",
		hint: "Apple Silicon GPU.",
	},
	{
		id: "cuda",
		label: "CUDA",
		arch: "any",
		kind: "gpu",
		hint: "NVIDIA GPU.",
	},
	{
		id: "xla",
		label: "XLA",
		arch: "any",
		kind: "compiler",
		hint: "Cross-platform JIT compiler.",
	},
];

export interface MetricInfo {
	id: string;
	label: string;
	op: string;
	hint: string;
	family: "quality" | "classification" | "language" | "performance";
}

export const METRICS: MetricInfo[] = [
	{
		id: "accuracy",
		label: "Accuracy",
		op: "bench.metric.accuracy",
		hint: "Top-1 classification correctness.",
		family: "classification",
	},
	{
		id: "f1",
		label: "F1",
		op: "bench.metric.f1",
		hint: "Harmonic mean of precision and recall.",
		family: "classification",
	},
	{
		id: "perplexity",
		label: "Perplexity",
		op: "bench.metric.perplexity",
		hint: "Exponentiated cross-entropy over the corpus.",
		family: "language",
	},
	{
		id: "throughput",
		label: "Throughput",
		op: "bench.metric.throughput",
		hint: "Samples processed per second.",
		family: "performance",
	},
	{
		id: "latency",
		label: "Latency",
		op: "bench.metric.latency",
		hint: "Per-sample wall time, including percentiles.",
		family: "performance",
	},
	{
		id: "loss",
		label: "Loss",
		op: "bench.metric.loss",
		hint: "Task loss tracked over evaluation steps.",
		family: "quality",
	},
];

export interface ModelInfo {
	id: string;
	label: string;
	family: string;
	params: string;
	checkpoint: string;
}

export const MODELS: ModelInfo[] = [
	{
		id: "llama-3.2-1b",
		label: "Llama 3.2 1B",
		family: "Llama",
		params: "1B",
		checkpoint: "checkpoints/llama-3.2-1b/epoch_000003.json",
	},
	{
		id: "llama-3.2-3b",
		label: "Llama 3.2 3B",
		family: "Llama",
		params: "3B",
		checkpoint: "checkpoints/llama-3.2-3b/epoch_000003.json",
	},
	{
		id: "flux-2-klein-4b",
		label: "Flux 2 Klein 4B",
		family: "Flux",
		params: "4B",
		checkpoint: "checkpoints/flux-2-klein/epoch_000010.json",
	},
	{
		id: "sst2-baseline",
		label: "SST-2 baseline",
		family: "Probe",
		params: "110M",
		checkpoint: "checkpoints/sst2/epoch_000010.json",
	},
];

export interface DatasetInfo {
	id: string;
	label: string;
	source: string;
	split: string;
	size: number;
	classes?: string[];
}

export const DATASETS: DatasetInfo[] = [
	{
		id: "glue-sst2",
		label: "GLUE / SST-2",
		source: "huggingface://glue",
		split: "validation",
		size: 872,
		classes: ["negative", "positive"],
	},
	{
		id: "mmlu",
		label: "MMLU",
		source: "huggingface://cais/mmlu",
		split: "test",
		size: 14042,
		classes: ["A", "B", "C", "D"],
	},
	{
		id: "gsm8k",
		label: "GSM8K",
		source: "huggingface://gsm8k",
		split: "test",
		size: 1319,
	},
	{
		id: "wikitext-103",
		label: "WikiText-103",
		source: "huggingface://wikitext",
		split: "test",
		size: 245569,
	},
	{
		id: "imagenet-1k",
		label: "ImageNet-1k",
		source: "huggingface://imagenet-1k",
		split: "validation",
		size: 50000,
		classes: Array.from({ length: 1000 }, (_, i) => `class_${i}`),
	},
];

export interface BenchmarkPreset {
	id: string;
	label: string;
	description: string;
	modelId: string;
	datasetId: string;
	metricIds: string[];
	backend: Backend;
	estimatedMinutes: number;
}

/*
PRESETS are one-click recipes — pick one and every wizard step is filled in
with the matching defaults. Researchers who want a custom run still get the
wizard; this is the path for "I want SST-2 numbers right now".
*/
export const PRESETS: BenchmarkPreset[] = [
	{
		id: "sst2-quick",
		label: "SST-2 quick eval",
		description: "Accuracy + F1 on the SST-2 validation split.",
		modelId: "sst2-baseline",
		datasetId: "glue-sst2",
		metricIds: ["accuracy", "f1", "throughput"],
		backend: "avx2",
		estimatedMinutes: 2,
	},
	{
		id: "mmlu-baseline",
		label: "MMLU baseline",
		description: "Llama 3.2 3B against MMLU test split.",
		modelId: "llama-3.2-3b",
		datasetId: "mmlu",
		metricIds: ["accuracy", "latency"],
		backend: "metal",
		estimatedMinutes: 38,
	},
	{
		id: "perplexity-wikitext",
		label: "WikiText perplexity",
		description: "Language modeling check, single pass.",
		modelId: "llama-3.2-1b",
		datasetId: "wikitext-103",
		metricIds: ["perplexity", "loss", "throughput"],
		backend: "neon",
		estimatedMinutes: 12,
	},
	{
		id: "gpu-bake-off",
		label: "GPU bake-off",
		description: "Latency + throughput across CUDA — for hardware tuning.",
		modelId: "llama-3.2-3b",
		datasetId: "gsm8k",
		metricIds: ["latency", "throughput", "accuracy"],
		backend: "cuda",
		estimatedMinutes: 6,
	},
];

export interface BenchmarkSpec {
	name: string;
	modelId: string;
	datasetId: string;
	metricIds: string[];
	backend: Backend;
	batchSize: number;
	limit: number | null;
	seed: number;
}

export const emptySpec = (): BenchmarkSpec => ({
	name: "",
	modelId: "",
	datasetId: "",
	metricIds: [],
	backend: "go-scalar",
	batchSize: 32,
	limit: null,
	seed: 42,
});
