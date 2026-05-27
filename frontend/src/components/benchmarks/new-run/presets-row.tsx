"use client";

import { SparklesIcon, WandIcon } from "lucide-react";
import {
	type BenchmarkSpec,
	PRESETS,
} from "#/components/benchmarks/model";
import {
	applyPresetToSpec,
	matchedPresetId,
} from "#/components/benchmarks/new-run/helpers";
import { SelectionCard } from "#/components/benchmarks/selection-card";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";
import { useWizard, useWizardDraft } from "#/components/ui/wizard";

/*
PresetsRow is the quick-start strip above the section indicator. The
active preset is derived live from the draft (no separate state) so
the highlight clears the moment the user customizes any field.
*/
export const PresetsRow = () => {
	const controller = useWizard<BenchmarkSpec>();
	const draft = useWizardDraft<BenchmarkSpec>();
	const activePresetId = matchedPresetId(draft);

	const applyPreset = (presetId: string) => {
		const preset = PRESETS.find((entry) => entry.id === presetId);

		if (!preset) {
			return;
		}

		controller.setDraft((current) => applyPresetToSpec(current, preset));
	};

	return (
		<Flex.Column gap={3} className="rounded-2xl border bg-card/40 p-4">
			<Flex.Row align="center" justify="between" wrap="wrap" gap={3}>
				<Flex.Row align="center" gap={2}>
					<SparklesIcon className="size-4 text-primary" />
					<Typography.Span className="font-medium text-sm">
						Quick start
					</Typography.Span>
					<Typography.Span variant="muted" className="text-xs">
						one-click presets
					</Typography.Span>
				</Flex.Row>
			</Flex.Row>
			<div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
				{PRESETS.map((preset) => (
					<SelectionCard
						key={preset.id}
						selected={activePresetId === preset.id}
						onSelect={() => applyPreset(preset.id)}
						icon={<WandIcon className="size-4" />}
						title={preset.label}
						subtitle={preset.description}
						hint={`~${preset.estimatedMinutes} min · ${preset.backend.toUpperCase()}`}
					/>
				))}
			</div>
		</Flex.Column>
	);
};
