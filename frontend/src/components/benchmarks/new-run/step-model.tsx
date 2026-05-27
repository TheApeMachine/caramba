"use client";

import { MODELS, type BenchmarkSpec } from "#/components/benchmarks/model";
import { SelectionCard } from "#/components/benchmarks/selection-card";

export const StepModel = ({
	draft,
	merge,
}: {
	draft: BenchmarkSpec;
	merge: (patch: Partial<BenchmarkSpec>) => void;
}) => (
	<div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
		{MODELS.map((model) => (
			<SelectionCard
				key={model.id}
				selected={draft.modelId === model.id}
				onSelect={() => merge({ modelId: model.id })}
				title={model.label}
				subtitle={`${model.family} · ${model.params}`}
				hint={model.checkpoint}
			/>
		))}
	</div>
);
