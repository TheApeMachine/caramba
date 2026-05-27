"use client";

import { DATASETS, type BenchmarkSpec } from "#/components/benchmarks/model";
import { SelectionCard } from "#/components/benchmarks/selection-card";

export const StepDataset = ({
	draft,
	merge,
}: {
	draft: BenchmarkSpec;
	merge: (patch: Partial<BenchmarkSpec>) => void;
}) => (
	<div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
		{DATASETS.map((dataset) => (
			<SelectionCard
				key={dataset.id}
				selected={draft.datasetId === dataset.id}
				onSelect={() => merge({ datasetId: dataset.id })}
				title={dataset.label}
				subtitle={`${dataset.split} · ${dataset.size.toLocaleString()} samples`}
				hint={dataset.source}
			/>
		))}
	</div>
);
