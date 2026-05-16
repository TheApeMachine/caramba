"use client";

import { Link } from "@tanstack/react-router";
import {
	ActivityIcon,
	ArrowLeftIcon,
	ClockIcon,
	CpuIcon,
	PauseIcon,
	PlayIcon,
	RotateCwIcon,
	TargetIcon,
	TimerIcon,
	TrendingDownIcon,
} from "lucide-react";
import { useEffect, useMemo } from "react";
import { Badge } from "#/components/ui/badge";
import { Button } from "#/components/ui/button";
import { ChartWidget } from "#/components/vega";
import { heatmapSpec } from "#/components/vega/specs/heatmap";
import { histogramSpec } from "#/components/vega/specs/histogram";
import { labeledBarSpec } from "#/components/vega/specs/labeled-bar";
import { lineSpec } from "#/components/vega/specs/line";
import { cn } from "@/lib/utils";
import { EventLog } from "./event-log";
import { KpiCard } from "./kpi-card";
import { useMockRun } from "./mock-stream";
import {
	BACKENDS,
	type BenchmarkSpec,
	DATASETS,
	METRICS,
	MODELS,
} from "./model";
import { Panel } from "./panel";
import { type RunRecord, saveRun } from "./store";

interface LiveRunProps {
	runId: string;
	spec: BenchmarkSpec;
	initialRecord: RunRecord;
}

const fmtDuration = (seconds: number): string => {
	if (!Number.isFinite(seconds) || seconds < 0) return "—";
	if (seconds < 60) return `${seconds.toFixed(0)}s`;
	const m = Math.floor(seconds / 60);
	const s = Math.floor(seconds % 60);
	return `${m}m ${s.toString().padStart(2, "0")}s`;
};

const fmtPct = (value: number): string => `${(value * 100).toFixed(1)}%`;

const StatusBadge = ({ status }: { status: string }) => {
	const variant: "success" | "info" | "warning" | "error" =
		status === "done"
			? "success"
			: status === "running"
				? "info"
				: status === "failed"
					? "error"
					: "warning";
	return (
		<Badge variant={variant} size="lg" className="capitalize">
			<span
				className={cn(
					"size-1.5 rounded-full",
					status === "running" && "animate-pulse bg-info-foreground",
					status === "done" && "bg-success-foreground",
					status === "failed" && "bg-destructive-foreground",
					status === "queued" && "bg-warning-foreground",
				)}
				aria-hidden
			/>
			{status}
		</Badge>
	);
};

/*
LiveRun is the streaming view for a single benchmark run. Layout is a fixed
12-row grid on a tall viewport: status header, KPI strip, then a 3-column
grid of panels (training curves, throughput, per-class breakdown, confusion
matrix, latency distribution, event log).

All data comes from useMockRun for now — when the real backend lands the
hook gets replaced with one wired to the platform's event stream and the
rest of this file stays put.
*/
export const LiveRun = ({ runId, spec, initialRecord }: LiveRunProps) => {
	const run = useMockRun(spec, { runId });

	const latest = run.steps[run.steps.length - 1];
	const earliest = run.steps[0];
	const hasSteps = run.steps.length > 0;

	const accuracyTrend = run.steps.map((step) => step.accuracy);
	const lossTrend = run.steps.map((step) => step.loss);
	const throughputTrend = run.steps.map((step) => step.throughput);

	const accuracyDelta = useMemo(() => {
		if (!latest || !earliest || latest === earliest) return null;
		const value = latest.accuracy - earliest.accuracy;
		return {
			value: `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)}pp`,
			positive: value >= 0,
		};
	}, [earliest, latest]);

	const lossDelta = useMemo(() => {
		if (!latest || !earliest || latest === earliest) return null;
		const value = latest.loss - earliest.loss;
		return {
			value: `${value >= 0 ? "+" : ""}${value.toFixed(2)}`,
			positive: value <= 0,
		};
	}, [earliest, latest]);

	const accuracyChart = useMemo(
		() =>
			lineSpec({
				data: run.steps as unknown as Array<Record<string, number | string>>,
				xField: "step",
				seriesKeys: ["accuracy"],
				seriesLabels: { accuracy: "Accuracy" },
				xTitle: "step",
				yTitle: "accuracy",
				yFormat: ".0%",
				zeroY: true,
			}),
		[run.steps],
	);

	const lossChart = useMemo(
		() =>
			lineSpec({
				data: run.steps as unknown as Array<Record<string, number | string>>,
				xField: "step",
				seriesKeys: ["loss"],
				seriesLabels: { loss: "Loss" },
				xTitle: "step",
				yTitle: "loss",
				yFormat: ".2f",
				zeroY: true,
			}),
		[run.steps],
	);

	const throughputChart = useMemo(
		() =>
			lineSpec({
				data: run.steps as unknown as Array<Record<string, number | string>>,
				xField: "step",
				seriesKeys: ["throughput"],
				seriesLabels: { throughput: "samples/sec" },
				xTitle: "step",
				yTitle: "samples/sec",
				yFormat: "d",
				zeroY: false,
			}),
		[run.steps],
	);

	const confusionChart = useMemo(() => {
		const data = run.classLabels.flatMap((rowLabel, rowIdx) =>
			run.classLabels.map((colLabel, colIdx) => ({
				row: rowLabel,
				col: colLabel,
				value: run.confusion[rowIdx]?.[colIdx] ?? 0,
			})),
		);
		return heatmapSpec({
			colOrder: run.classLabels,
			colTitle: "Predicted",
			data,
			normalize: true,
			rowOrder: run.classLabels,
			rowTitle: "Actual",
			showValues: run.classLabels.length <= 8,
			valueFormat: "d",
		});
	}, [run.classLabels, run.confusion]);

	const latencyChart = useMemo(
		() =>
			histogramSpec({
				values: run.latencies,
				bins: 24,
				xTitle: "ms / sample",
				xFormat: ".2f",
			}),
		[run.latencies],
	);

	const perClassBars = useMemo(() => {
		const data = run.classLabels.map((label, idx) => {
			const totals = run.confusion[idx] ?? [];
			const total = totals.reduce((acc, value) => acc + value, 0);
			const correct = totals[idx] ?? 0;
			return {
				label,
				value: total > 0 ? Number(((correct / total) * 100).toFixed(1)) : 0,
			};
		});
		return labeledBarSpec({
			axisTitle: "accuracy %",
			data,
			orientation: "horizontal",
			sort: "desc",
			threshold: 80,
			valueDomain: [0, 100],
			valueFormat: ".0f",
		});
	}, [run.classLabels, run.confusion]);

	const model = MODELS.find((entry) => entry.id === spec.modelId);
	const dataset = DATASETS.find((entry) => entry.id === spec.datasetId);
	const backend = BACKENDS.find((entry) => entry.id === spec.backend);
	const metricLabels = spec.metricIds
		.map((id) => METRICS.find((entry) => entry.id === id)?.label)
		.filter(Boolean) as string[];

	useEffect(() => {
		if (run.status === "done" && latest) {
			saveRun({
				...initialRecord,
				status: "done",
				finalAccuracy: latest.accuracy,
				finalLoss: latest.loss,
				durationSeconds: latest.elapsedSeconds,
			});
		}
	}, [run.status, latest, initialRecord]);

	const progressPct =
		run.totalSamples > 0
			? Math.round((run.processedSamples / run.totalSamples) * 100)
			: 0;

	return (
		<div className="flex h-full min-h-0 flex-1 flex-col gap-4">
			<header className="flex flex-wrap items-start justify-between gap-3 border-b pb-4">
				<div className="flex flex-col gap-2">
					<div className="flex items-center gap-2">
						<Button
							variant="ghost"
							size="sm"
							render={<Link to="/benchmarks" />}
						>
							<ArrowLeftIcon /> Benchmarks
						</Button>
					</div>
					<div className="flex flex-wrap items-center gap-3">
						<h1 className="font-semibold text-foreground text-xl">
							{spec.name}
						</h1>
						<StatusBadge status={run.status} />
					</div>
					<div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-muted-foreground text-xs">
						<span className="flex items-center gap-1">
							<CpuIcon className="size-3" /> {backend?.label ?? spec.backend}
						</span>
						<span>{model?.label ?? spec.modelId}</span>
						<span>{dataset?.label ?? spec.datasetId}</span>
						<span>id: {runId.slice(0, 8)}</span>
					</div>
					{metricLabels.length > 0 ? (
						<div className="flex flex-wrap gap-1">
							{metricLabels.map((label) => (
								<Badge key={label} variant="outline" size="sm">
									{label}
								</Badge>
							))}
						</div>
					) : null}
				</div>
				<div className="flex items-center gap-2">
					{run.status === "running" ? (
						<Button variant="outline" onClick={run.stop}>
							<PauseIcon /> Stop
						</Button>
					) : (
						<Button variant="outline" onClick={run.restart}>
							{run.status === "done" ? <RotateCwIcon /> : <PlayIcon />}{" "}
							{run.status === "done" ? "Re-run" : "Resume"}
						</Button>
					)}
				</div>
			</header>

			<section className="grid grid-cols-2 gap-3 md:grid-cols-4">
				<KpiCard
					emphasis
					icon={<TargetIcon className="size-4" />}
					label="Accuracy"
					value={latest ? fmtPct(latest.accuracy) : "—"}
					delta={accuracyDelta ?? undefined}
					trend={accuracyTrend}
				/>
				<KpiCard
					icon={<TrendingDownIcon className="size-4" />}
					label="Loss"
					value={latest ? latest.loss.toFixed(3) : "—"}
					delta={lossDelta ?? undefined}
					trend={lossTrend}
				/>
				<KpiCard
					icon={<ActivityIcon className="size-4" />}
					label="Throughput"
					value={latest ? `${latest.throughput.toFixed(0)} /s` : "—"}
					trend={throughputTrend}
				/>
				<KpiCard
					icon={<TimerIcon className="size-4" />}
					label="ETA"
					value={run.status === "done" ? "complete" : fmtDuration(run.eta)}
				/>
			</section>

			<section className="flex items-center gap-3 rounded-2xl border bg-card/40 px-4 py-2.5 text-xs">
				<ClockIcon className="size-4 text-muted-foreground" />
				<div className="flex flex-1 flex-col gap-1">
					<div className="flex items-center justify-between gap-3 text-muted-foreground">
						<span>
							{run.processedSamples.toLocaleString()} /{" "}
							{run.totalSamples.toLocaleString()} samples
						</span>
						<span>{progressPct}%</span>
					</div>
					<div className="h-1.5 overflow-hidden rounded-full bg-muted">
						<div
							className={cn(
								"h-full rounded-full bg-primary transition-all",
								run.status === "running" && "animate-pulse",
							)}
							style={{ width: `${progressPct}%` }}
						/>
					</div>
				</div>
			</section>

			<section className="grid min-h-0 flex-1 grid-cols-1 grid-rows-[minmax(260px,1fr)_minmax(260px,1fr)_minmax(220px,1fr)] gap-3 overflow-y-auto pr-1 lg:grid-cols-3">
				<Panel
					title="Accuracy"
					hint="Top-1 correctness over evaluation steps"
					className="lg:col-span-2"
				>
					{hasSteps ? (
						<ChartWidget spec={accuracyChart} />
					) : (
						<ChartEmpty label="Waiting for first batch…" />
					)}
				</Panel>

				<Panel title="Throughput" hint="Samples processed per second">
					{hasSteps ? (
						<ChartWidget spec={throughputChart} />
					) : (
						<ChartEmpty label="Waiting for first batch…" />
					)}
				</Panel>

				<Panel title="Loss" hint="Task loss over evaluation steps">
					{hasSteps ? (
						<ChartWidget spec={lossChart} />
					) : (
						<ChartEmpty label="Waiting for first batch…" />
					)}
				</Panel>

				<Panel
					title="Latency distribution"
					hint="ms per sample over the rolling window"
				>
					{run.latencies.length > 0 ? (
						<ChartWidget spec={latencyChart} />
					) : (
						<ChartEmpty label="Waiting for first batch…" />
					)}
				</Panel>

				<Panel
					title="Event log"
					hint="Live output from the run"
					bodyClassName="p-0"
				>
					<div className="m-3 h-[calc(100%-1.5rem)] w-[calc(100%-1.5rem)]">
						<EventLog events={run.events} />
					</div>
				</Panel>

				<Panel
					title="Per-class accuracy"
					hint="Correct predictions per class (%)"
				>
					{hasSteps ? (
						<ChartWidget spec={perClassBars} />
					) : (
						<ChartEmpty label="Waiting for first batch…" />
					)}
				</Panel>

				<Panel
					title="Confusion matrix"
					hint="Predicted vs actual class counts"
					className="lg:col-span-2"
				>
					{hasSteps ? (
						<ChartWidget spec={confusionChart} />
					) : (
						<ChartEmpty label="Waiting for first batch…" />
					)}
				</Panel>
			</section>
		</div>
	);
};

const ChartEmpty = ({ label }: { label: string }) => (
	<div className="flex h-full w-full items-center justify-center text-muted-foreground text-xs">
		{label}
	</div>
);
