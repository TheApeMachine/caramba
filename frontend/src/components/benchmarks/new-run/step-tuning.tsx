"use client";

import type { BenchmarkSpec } from "#/components/benchmarks/model";
import { Field } from "#/components/ui/field";
import { Input } from "#/components/ui/input";

export const StepTuning = ({
	draft,
	merge,
}: {
	draft: BenchmarkSpec;
	merge: (patch: Partial<BenchmarkSpec>) => void;
}) => (
	<div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
		<Field>
			<Field.Label htmlFor="run-name">Run name</Field.Label>
			<Input
				id="run-name"
				value={draft.name}
				onChange={(event) => merge({ name: event.target.value })}
			/>
		</Field>
		<Field>
			<Field.Label htmlFor="seed">Seed</Field.Label>
			<Input
				id="seed"
				type="number"
				value={draft.seed}
				onChange={(event) => merge({ seed: Number(event.target.value) })}
			/>
		</Field>
		<Field>
			<Field.Label htmlFor="batch-size">Batch size</Field.Label>
			<Input
				id="batch-size"
				type="number"
				min={1}
				value={draft.batchSize}
				onChange={(event) =>
					merge({ batchSize: Math.max(1, Number(event.target.value)) })
				}
			/>
		</Field>
		<Field>
			<Field.Label htmlFor="limit">Sample limit (optional)</Field.Label>
			<Input
				id="limit"
				type="number"
				placeholder="all"
				value={draft.limit ?? ""}
				onChange={(event) =>
					merge({
						limit: event.target.value ? Number(event.target.value) : null,
					})
				}
			/>
		</Field>
	</div>
);
