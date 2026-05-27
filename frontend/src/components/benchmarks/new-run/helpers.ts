import {
	type BenchmarkPreset,
	type BenchmarkSpec,
	emptySpec,
	PRESETS,
} from "#/components/benchmarks/model";

const todayIso = (): string => new Date().toISOString().slice(0, 10);

const randomTag = (): string => Math.random().toString(36).slice(2, 6);

export const defaultRunName = (): string => `run-${todayIso()}-${randomTag()}`;

const presetSuffix = (): string => Date.now().toString(36).slice(-4);

export const applyPresetToSpec = (
	spec: BenchmarkSpec,
	preset: BenchmarkPreset,
): BenchmarkSpec => ({
	...spec,
	modelId: preset.modelId,
	datasetId: preset.datasetId,
	metricIds: [...preset.metricIds],
	backend: preset.backend,
	name: spec.name || `${preset.id}-${presetSuffix()}`,
});

export const presetById = (presetId: string | null): BenchmarkPreset | null => {
	if (!presetId) {
		return null;
	}

	return PRESETS.find((entry) => entry.id === presetId) ?? null;
};

/*
matchedPresetId returns the preset whose model / dataset / metrics /
backend exactly equal the current spec, or null when the spec has been
edited away from any preset.
*/
export const matchedPresetId = (spec: BenchmarkSpec): string | null => {
	for (const preset of PRESETS) {
		if (preset.modelId !== spec.modelId) continue;
		if (preset.datasetId !== spec.datasetId) continue;
		if (preset.backend !== spec.backend) continue;
		if (preset.metricIds.length !== spec.metricIds.length) continue;

		const matchesAll = preset.metricIds.every((metricId) =>
			spec.metricIds.includes(metricId),
		);

		if (matchesAll) {
			return preset.id;
		}
	}

	return null;
};

/*
createInitialBenchmarkDraft seeds a draft with the optional preset and
a default name so the wizard mounts in its post-init shape without
needing a useEffect to fill in either.
*/
export const createInitialBenchmarkDraft = (
	presetId: string | null,
): BenchmarkSpec => {
	const base: BenchmarkSpec = { ...emptySpec(), name: defaultRunName() };
	const preset = presetById(presetId);

	if (!preset) {
		return base;
	}

	return applyPresetToSpec(base, preset);
};
