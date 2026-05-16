import { type BenchmarkSpec, DATASETS, METRICS, MODELS } from "./model";

/*
specToManifest renders a BenchmarkSpec as the YAML manifest the platform
actually executes. The shape mirrors pkg/asset/template/manifest/experiment_bench.yml
so a researcher can read the wizard preview and immediately recognize what
will hit the backend.
*/
export const specToManifest = (spec: BenchmarkSpec): string => {
	const model = MODELS.find((entry) => entry.id === spec.modelId);
	const dataset = DATASETS.find((entry) => entry.id === spec.datasetId);
	const metrics = spec.metricIds
		.map((id) => METRICS.find((entry) => entry.id === id))
		.filter((entry): entry is NonNullable<typeof entry> => Boolean(entry));

	const lines: string[] = [];

	lines.push(`# ${spec.name || "untitled benchmark"}`);
	lines.push(`# backend: ${spec.backend}`);
	lines.push("system:");
	lines.push("  topology:");
	lines.push("    nodes:");

	lines.push("      - id: weights");
	lines.push("        op: train.checkpoint.load");
	lines.push("        config:");
	lines.push(`          path: ${model?.checkpoint ?? "<select a model>"}`);
	lines.push("        out: [loaded_params]");
	lines.push("");

	lines.push("      - id: data");
	lines.push("        op: data.huggingface");
	lines.push("        config:");
	lines.push(`          dataset: ${dataset?.source ?? "<select a dataset>"}`);
	lines.push(`          split: ${dataset?.split ?? "validation"}`);
	lines.push(`          page: ${spec.batchSize}`);

	if (spec.limit) lines.push(`          limit: ${spec.limit}`);
	lines.push("        out: [batch]");
	lines.push("");

	lines.push("      - id: forward");
	lines.push(`        op: model.${spec.backend}.forward`);
	lines.push("        in:  [loaded_params, batch]");
	lines.push("        out: [logits]");

	if (metrics.length > 0) lines.push("");

	for (const metric of metrics) {
		lines.push(`      - id: ${metric.id}`);
		lines.push(`        op: ${metric.op}`);
		lines.push("        in:  [logits, batch]");
		lines.push(`        out: [${metric.id}_score]`);
		lines.push("");
	}

	lines.push("seed:");
	lines.push(`  value: ${spec.seed}`);

	return lines.join("\n");
};
