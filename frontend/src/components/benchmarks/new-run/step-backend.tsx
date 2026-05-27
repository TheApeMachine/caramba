"use client";

import { BACKENDS, type BenchmarkSpec } from "#/components/benchmarks/model";
import { SelectionCard } from "#/components/benchmarks/selection-card";

export const StepBackend = ({
	draft,
	merge,
}: {
	draft: BenchmarkSpec;
	merge: (patch: Partial<BenchmarkSpec>) => void;
}) => (
	<div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
		{BACKENDS.map((backend) => (
			<SelectionCard
				key={backend.id}
				selected={draft.backend === backend.id}
				onSelect={() => merge({ backend: backend.id })}
				title={backend.label}
				subtitle={`${backend.kind} · ${backend.arch}`}
				hint={backend.hint}
			/>
		))}
	</div>
);
