"use client";

import { useNavigate } from "@tanstack/react-router";
import {
	CpuIcon,
	DatabaseIcon,
	GaugeIcon,
	NetworkIcon,
	PlayIcon,
	ZapIcon,
} from "lucide-react";
import type { BenchmarkSpec } from "#/components/benchmarks/model";
import { ManifestPreview } from "#/components/benchmarks/manifest-preview";
import { createInitialBenchmarkDraft } from "#/components/benchmarks/new-run/helpers";
import { PresetsRow } from "#/components/benchmarks/new-run/presets-row";
import { StepBackend } from "#/components/benchmarks/new-run/step-backend";
import { StepDataset } from "#/components/benchmarks/new-run/step-dataset";
import { StepMetrics } from "#/components/benchmarks/new-run/step-metrics";
import { StepModel } from "#/components/benchmarks/new-run/step-model";
import { StepTuning } from "#/components/benchmarks/new-run/step-tuning";
import { saveRun } from "#/components/benchmarks/store";
import {
	useWizardDraft,
	Wizard,
	type WizardStepDefinition,
} from "#/components/ui/wizard";

const steps: ReadonlyArray<WizardStepDefinition<BenchmarkSpec>> = [
	{
		id: "model",
		title: "Model",
		subtitle: "Pick the checkpoint to evaluate.",
		icon: <NetworkIcon className="size-4" />,
		isComplete: (draft) => Boolean(draft.modelId),
		render: ({ draft, merge }) => <StepModel draft={draft} merge={merge} />,
	},
	{
		id: "dataset",
		title: "Dataset",
		subtitle: "The evaluation set the run will stream through.",
		icon: <DatabaseIcon className="size-4" />,
		isComplete: (draft) => Boolean(draft.datasetId),
		render: ({ draft, merge }) => <StepDataset draft={draft} merge={merge} />,
	},
	{
		id: "metrics",
		title: "Metrics",
		subtitle: "Pick one or more. They wire up as nodes in the manifest.",
		icon: <GaugeIcon className="size-4" />,
		isComplete: (draft) => draft.metricIds.length > 0,
		render: ({ draft, merge }) => <StepMetrics draft={draft} merge={merge} />,
	},
	{
		id: "hardware",
		title: "Backend",
		subtitle: "Choose the execution target.",
		icon: <CpuIcon className="size-4" />,
		isComplete: (draft) => Boolean(draft.backend),
		render: ({ draft, merge }) => <StepBackend draft={draft} merge={merge} />,
	},
	{
		id: "tuning",
		title: "Run settings",
		subtitle: "Name the run and tune sampling parameters.",
		icon: <ZapIcon className="size-4" />,
		isComplete: (draft) => Boolean(draft.name && draft.batchSize > 0),
		render: ({ draft, merge }) => <StepTuning draft={draft} merge={merge} />,
	},
];

const LiveManifest = () => {
	const draft = useWizardDraft<BenchmarkSpec>();
	return <ManifestPreview spec={draft} />;
};

/*
NewBenchmarkWizard is the sectioned single-screen flow for creating a
benchmark run. The optional initialPresetId seeds the draft once at
construction so the user lands with every field filled in when they
arrive from a preset deep link.
*/
export const NewBenchmarkWizard = ({
	initialPresetId = null,
}: {
	initialPresetId?: string | null;
}) => {
	const navigate = useNavigate();

	return (
		<Wizard<BenchmarkSpec>
			mode="sectioned"
			title="New benchmark"
			subtitle="Pick a preset or scroll through the steps. The manifest on the right updates as you go."
			submitLabel="Launch benchmark"
			submitPendingLabel="Launching…"
			submitIcon={<PlayIcon />}
			steps={steps}
			initialDraft={createInitialBenchmarkDraft(initialPresetId)}
			header={<PresetsRow />}
			preview={<LiveManifest />}
			onCancel={() => navigate({ to: "/benchmarks" })}
			onSubmit={async (draft) => {
				const id = crypto.randomUUID();

				saveRun({
					id,
					createdAt: Date.now(),
					spec: draft,
					finalAccuracy: null,
					finalLoss: null,
					status: "running",
					durationSeconds: null,
				});

				await navigate({ to: "/benchmarks/$runId", params: { runId: id } });
			}}
		/>
	);
};
