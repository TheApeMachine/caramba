"use client";

import { useNavigate } from "@tanstack/react-router";
import {
	CheckIcon,
	CpuIcon,
	DatabaseIcon,
	GaugeIcon,
	NetworkIcon,
	PlayIcon,
	SparklesIcon,
	WandIcon,
	XIcon,
	ZapIcon,
} from "lucide-react";
import { type ReactNode, useEffect, useMemo, useState } from "react";
import { Badge } from "#/components/ui/badge";
import { Button } from "#/components/ui/button";
import { Input } from "#/components/ui/input";
import { Label } from "#/components/ui/label";
import { cn } from "@/lib/utils";
import { ManifestPreview } from "./manifest-preview";
import {
	BACKENDS,
	type BenchmarkSpec,
	DATASETS,
	emptySpec,
	METRICS,
	MODELS,
	PRESETS,
} from "./model";
import { SelectionCard } from "./selection-card";
import { saveRun } from "./store";

const SECTION_IDS = [
	"model",
	"dataset",
	"metrics",
	"hardware",
	"tuning",
] as const;
type SectionId = (typeof SECTION_IDS)[number];

const sectionLabel: Record<SectionId, string> = {
	model: "Model",
	dataset: "Dataset",
	metrics: "Metrics",
	hardware: "Backend",
	tuning: "Run settings",
};

const sectionIcon: Record<SectionId, ReactNode> = {
	model: <NetworkIcon className="size-4" />,
	dataset: <DatabaseIcon className="size-4" />,
	metrics: <GaugeIcon className="size-4" />,
	hardware: <CpuIcon className="size-4" />,
	tuning: <ZapIcon className="size-4" />,
};

const isComplete = (spec: BenchmarkSpec, section: SectionId): boolean => {
	if (section === "model") return Boolean(spec.modelId);
	if (section === "dataset") return Boolean(spec.datasetId);
	if (section === "metrics") return spec.metricIds.length > 0;
	if (section === "hardware") return Boolean(spec.backend);
	return Boolean(spec.name && spec.batchSize > 0);
};

const readyToLaunch = (spec: BenchmarkSpec): boolean =>
	Boolean(
		spec.name &&
			spec.modelId &&
			spec.datasetId &&
			spec.metricIds.length > 0 &&
			spec.backend &&
			spec.batchSize > 0,
	);

const SectionShell = ({
	id,
	title,
	hint,
	complete,
	children,
}: {
	id: SectionId;
	title: string;
	hint: string;
	complete: boolean;
	children: ReactNode;
}) => (
	<section
		id={`section-${id}`}
		className="flex scroll-mt-4 flex-col gap-3 rounded-2xl border bg-card/40 p-4"
	>
		<div className="flex items-start justify-between gap-4">
			<div className="flex items-center gap-2">
				<span
					className={cn(
						"flex size-7 items-center justify-center rounded-full border transition",
						complete
							? "border-primary bg-primary text-primary-foreground"
							: "border-muted-foreground/30 bg-background text-muted-foreground",
					)}
				>
					{complete ? <CheckIcon className="size-4" /> : sectionIcon[id]}
				</span>
				<div className="flex flex-col">
					<h2 className="font-semibold text-foreground text-sm">{title}</h2>
					<p className="text-muted-foreground text-xs">{hint}</p>
				</div>
			</div>
		</div>
		{children}
	</section>
);

const Stepper = ({
	spec,
	onJump,
}: {
	spec: BenchmarkSpec;
	onJump: (section: SectionId) => void;
}) => (
	<ol className="flex flex-wrap items-center gap-1">
		{SECTION_IDS.map((id, index) => {
			const complete = isComplete(spec, id);
			return (
				<li key={id} className="flex items-center gap-1">
					<button
						type="button"
						onClick={() => onJump(id)}
						className={cn(
							"flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs transition",
							complete
								? "border-primary/40 bg-primary/5 text-foreground"
								: "border-transparent text-muted-foreground hover:bg-muted/40",
						)}
					>
						<span
							className={cn(
								"flex size-5 items-center justify-center rounded-full font-medium text-[10px]",
								complete
									? "bg-primary text-primary-foreground"
									: "bg-muted text-muted-foreground",
							)}
						>
							{complete ? <CheckIcon className="size-3" /> : index + 1}
						</span>
						{sectionLabel[id]}
					</button>
					{index < SECTION_IDS.length - 1 && (
						<span
							aria-hidden
							className="h-px w-3 bg-muted-foreground/30 sm:w-4"
						/>
					)}
				</li>
			);
		})}
	</ol>
);

interface WizardProps {
	initialPresetId?: string | null;
}

/*
Wizard is the single-screen "new benchmark" flow. Everything is visible —
no router-based step paging — so the user always sees the manifest forming
in the right pane as they click. Quick-start presets prefill every step in
one shot for researchers who don't want to scroll.
*/
export const Wizard = ({ initialPresetId = null }: WizardProps = {}) => {
	const navigate = useNavigate();
	const [spec, setSpec] = useState<BenchmarkSpec>(() => emptySpec());
	const [activePresetId, setActivePresetId] = useState<string | null>(null);
	const [launching, setLaunching] = useState(false);

	useEffect(() => {
		if (!initialPresetId) return;
		const preset = PRESETS.find((entry) => entry.id === initialPresetId);
		if (!preset) return;
		setActivePresetId(preset.id);
		setSpec((current) => ({
			...current,
			modelId: preset.modelId,
			datasetId: preset.datasetId,
			metricIds: [...preset.metricIds],
			backend: preset.backend,
		}));
	}, [initialPresetId]);

	useEffect(() => {
		if (spec.name) return;
		setSpec((current) => ({
			...current,
			name: `run-${new Date().toISOString().slice(0, 10)}-${Math.random()
				.toString(36)
				.slice(2, 6)}`,
		}));
	}, [spec.name]);

	const applyPreset = (presetId: string) => {
		const preset = PRESETS.find((entry) => entry.id === presetId);
		if (!preset) return;
		setActivePresetId(presetId);
		setSpec((current) => ({
			...current,
			modelId: preset.modelId,
			datasetId: preset.datasetId,
			metricIds: [...preset.metricIds],
			backend: preset.backend,
			name: current.name || `${preset.id}-${Date.now().toString(36).slice(-4)}`,
		}));
	};

	const updateSpec = <K extends keyof BenchmarkSpec>(
		key: K,
		value: BenchmarkSpec[K],
	) => {
		setActivePresetId(null);
		setSpec((current) => ({ ...current, [key]: value }));
	};

	const toggleMetric = (id: string) => {
		setActivePresetId(null);
		setSpec((current) => ({
			...current,
			metricIds: current.metricIds.includes(id)
				? current.metricIds.filter((entry) => entry !== id)
				: [...current.metricIds, id],
		}));
	};

	const scrollTo = (section: SectionId) => {
		document
			.getElementById(`section-${section}`)
			?.scrollIntoView({ behavior: "smooth", block: "start" });
	};

	const launch = async () => {
		if (!readyToLaunch(spec) || launching) return;
		setLaunching(true);
		const id = crypto.randomUUID();
		saveRun({
			id,
			createdAt: Date.now(),
			spec,
			finalAccuracy: null,
			finalLoss: null,
			status: "running",
			durationSeconds: null,
		});
		await navigate({ to: "/benchmarks/$runId", params: { runId: id } });
	};

	const completedCount = useMemo(
		() => SECTION_IDS.filter((id) => isComplete(spec, id)).length,
		[spec],
	);

	return (
		<div className="flex h-full min-h-0 flex-1 flex-col gap-4">
			<header className="flex flex-wrap items-center justify-between gap-3">
				<div className="flex flex-col">
					<h1 className="font-semibold text-foreground text-lg">
						New benchmark
					</h1>
					<p className="text-muted-foreground text-sm">
						Pick a preset or scroll through the steps. The manifest on the right
						updates as you go.
					</p>
				</div>
				<div className="flex items-center gap-2">
					<Badge variant="outline" size="lg">
						{completedCount}/{SECTION_IDS.length} steps complete
					</Badge>
					<Button
						type="button"
						variant="outline"
						onClick={() => navigate({ to: "/benchmarks" })}
					>
						<XIcon /> Cancel
					</Button>
					<Button
						type="button"
						onClick={launch}
						disabled={!readyToLaunch(spec) || launching}
					>
						<PlayIcon /> {launching ? "Launching…" : "Launch benchmark"}
					</Button>
				</div>
			</header>

			<div className="rounded-2xl border bg-card/40 p-4">
				<div className="flex flex-wrap items-center justify-between gap-3">
					<div className="flex items-center gap-2 text-foreground text-sm">
						<SparklesIcon className="size-4 text-primary" />
						<span className="font-medium">Quick start</span>
						<span className="text-muted-foreground text-xs">
							one-click presets
						</span>
					</div>
				</div>
				<div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
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
			</div>

			<Stepper spec={spec} onJump={scrollTo} />

			<div className="grid min-h-0 flex-1 grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(360px,420px)]">
				<div className="flex min-h-0 flex-col gap-4 overflow-y-auto pr-1">
					<SectionShell
						id="model"
						title="Model"
						hint="Pick the checkpoint to evaluate."
						complete={isComplete(spec, "model")}
					>
						<div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
							{MODELS.map((model) => (
								<SelectionCard
									key={model.id}
									selected={spec.modelId === model.id}
									onSelect={() => updateSpec("modelId", model.id)}
									title={model.label}
									subtitle={`${model.family} · ${model.params}`}
									hint={model.checkpoint}
								/>
							))}
						</div>
					</SectionShell>

					<SectionShell
						id="dataset"
						title="Dataset"
						hint="The evaluation set the run will stream through."
						complete={isComplete(spec, "dataset")}
					>
						<div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
							{DATASETS.map((dataset) => (
								<SelectionCard
									key={dataset.id}
									selected={spec.datasetId === dataset.id}
									onSelect={() => updateSpec("datasetId", dataset.id)}
									title={dataset.label}
									subtitle={`${dataset.split} · ${dataset.size.toLocaleString()} samples`}
									hint={dataset.source}
								/>
							))}
						</div>
					</SectionShell>

					<SectionShell
						id="metrics"
						title="Metrics"
						hint="Pick one or more. They wire up as nodes in the manifest."
						complete={isComplete(spec, "metrics")}
					>
						<div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
							{METRICS.map((metric) => (
								<SelectionCard
									key={metric.id}
									selected={spec.metricIds.includes(metric.id)}
									onSelect={() => toggleMetric(metric.id)}
									title={metric.label}
									subtitle={metric.op}
									hint={metric.hint}
								/>
							))}
						</div>
					</SectionShell>

					<SectionShell
						id="hardware"
						title="Backend"
						hint="Choose the execution target."
						complete={isComplete(spec, "hardware")}
					>
						<div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
							{BACKENDS.map((backend) => (
								<SelectionCard
									key={backend.id}
									selected={spec.backend === backend.id}
									onSelect={() => updateSpec("backend", backend.id)}
									title={backend.label}
									subtitle={`${backend.kind} · ${backend.arch}`}
									hint={backend.hint}
								/>
							))}
						</div>
					</SectionShell>

					<SectionShell
						id="tuning"
						title="Run settings"
						hint="Name the run and tune sampling parameters."
						complete={isComplete(spec, "tuning")}
					>
						<div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
							<div className="flex flex-col gap-1">
								<Label htmlFor="run-name">Run name</Label>
								<Input
									id="run-name"
									value={spec.name}
									onChange={(event) => updateSpec("name", event.target.value)}
								/>
							</div>
							<div className="flex flex-col gap-1">
								<Label htmlFor="seed">Seed</Label>
								<Input
									id="seed"
									type="number"
									value={spec.seed}
									onChange={(event) =>
										updateSpec("seed", Number(event.target.value))
									}
								/>
							</div>
							<div className="flex flex-col gap-1">
								<Label htmlFor="batch-size">Batch size</Label>
								<Input
									id="batch-size"
									type="number"
									min={1}
									value={spec.batchSize}
									onChange={(event) =>
										updateSpec(
											"batchSize",
											Math.max(1, Number(event.target.value)),
										)
									}
								/>
							</div>
							<div className="flex flex-col gap-1">
								<Label htmlFor="limit">Sample limit (optional)</Label>
								<Input
									id="limit"
									type="number"
									placeholder="all"
									value={spec.limit ?? ""}
									onChange={(event) =>
										updateSpec(
											"limit",
											event.target.value ? Number(event.target.value) : null,
										)
									}
								/>
							</div>
						</div>
					</SectionShell>
				</div>

				<aside className="min-h-[420px]">
					<ManifestPreview spec={spec} />
				</aside>
			</div>
		</div>
	);
};
