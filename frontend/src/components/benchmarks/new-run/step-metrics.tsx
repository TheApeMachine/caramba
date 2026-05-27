"use client";

import { METRICS, type BenchmarkSpec } from "#/components/benchmarks/model";
import { SelectionCard } from "#/components/benchmarks/selection-card";

export const StepMetrics = ({
	draft,
	merge,
}: {
	draft: BenchmarkSpec;
	merge: (patch: Partial<BenchmarkSpec>) => void;
}) => {
	const toggleMetric = (metricId: string) => {
		merge({
			metricIds: draft.metricIds.includes(metricId)
				? draft.metricIds.filter((entry) => entry !== metricId)
				: [...draft.metricIds, metricId],
		});
	};

	return (
		<div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
			{METRICS.map((metric) => (
				<SelectionCard
					key={metric.id}
					selected={draft.metricIds.includes(metric.id)}
					onSelect={() => toggleMetric(metric.id)}
					title={metric.label}
					subtitle={metric.op}
					hint={metric.hint}
				/>
			))}
		</div>
	);
};
