"use client";

import { Link } from "@tanstack/react-router";
import {
	ClockIcon,
	CpuIcon,
	DatabaseIcon,
	FlaskConicalIcon,
	PlusIcon,
	TrashIcon,
} from "lucide-react";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Badge } from "#/components/ui/badge";
import { Button } from "#/components/ui/button";
import { cn } from "#/lib/utils";
import { BACKENDS, DATASETS, MODELS, PRESETS } from "./model";
import { deleteRun, loadRuns, type RunRecord } from "./store";

const statusVariant: Record<
	RunRecord["status"],
	"success" | "info" | "warning" | "error"
> = {
	done: "success",
	running: "info",
	queued: "warning",
	failed: "error",
};

const RunRow = ({
	run,
	onDelete,
}: {
	run: RunRecord;
	onDelete: () => void;
}) => {
	const { t } = useTranslation();
	const model = MODELS.find((entry) => entry.id === run.spec.modelId);
	const dataset = DATASETS.find((entry) => entry.id === run.spec.datasetId);
	const backend = BACKENDS.find((entry) => entry.id === run.spec.backend);

	const fmtRelative = (timestamp: number): string => {
		const diff = Date.now() - timestamp;
		if (diff < 60_000) return t("benchmarks.justNow");
		if (diff < 3_600_000) {
			return t("benchmarks.minutesAgo", {
				count: Math.round(diff / 60_000),
			});
		}
		if (diff < 86_400_000) {
			return t("benchmarks.hoursAgo", {
				count: Math.round(diff / 3_600_000),
			});
		}
		return new Date(timestamp).toLocaleDateString();
	};

	return (
		<li className="group flex items-center gap-3 rounded-xl border bg-card/40 px-4 py-3 hover:border-primary/40 hover:bg-card/80">
			<Link
				to="/benchmarks/$runId"
				params={{ runId: run.id }}
				className="flex min-w-0 flex-1 items-center gap-4"
			>
				<div className="flex flex-1 min-w-0 flex-col gap-1">
					<div className="flex items-center gap-2">
						<span className="truncate font-medium text-foreground text-sm">
							{run.spec.name || t("benchmarks.untitledRun")}
						</span>
						<Badge
							variant={statusVariant[run.status]}
							size="sm"
							className="capitalize"
						>
							{t(`benchmarks.status.${run.status}`)}
						</Badge>
					</div>
					<div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-muted-foreground text-xs">
						<span className="flex items-center gap-1">
							<CpuIcon className="size-3" />
							{backend?.label ?? run.spec.backend}
						</span>
						<span>{model?.label ?? run.spec.modelId}</span>
						<span className="flex items-center gap-1">
							<DatabaseIcon className="size-3" />
							{dataset?.label ?? run.spec.datasetId}
						</span>
						<span className="flex items-center gap-1">
							<ClockIcon className="size-3" />
							{fmtRelative(run.createdAt)}
						</span>
					</div>
				</div>
				<div className="flex shrink-0 flex-col items-end gap-0.5 text-right text-xs">
					<span className="font-mono text-foreground tabular-nums">
						{run.finalAccuracy !== null
							? `${(run.finalAccuracy * 100).toFixed(2)}%`
							: "—"}
					</span>
					<span className="text-muted-foreground">
						{run.finalLoss !== null
							? t("benchmarks.lossPrefix", {
									value: run.finalLoss.toFixed(3),
								})
							: ""}
					</span>
				</div>
			</Link>
			<Button
				variant="ghost"
				size="sm"
				onClick={onDelete}
				className="opacity-0 transition group-hover:opacity-100"
				aria-label={t("benchmarks.deleteRun")}
			>
				<TrashIcon />
			</Button>
		</li>
	);
};

/*
IndexView is the benchmarks landing page. Shows the recent run list (from
localStorage for the mock) plus the preset shortcuts so the most common
flow — "run SST-2 again" — never costs the user a wizard click.
*/
export const IndexView = () => {
	const { t } = useTranslation();
	const [runs, setRuns] = useState<RunRecord[]>([]);

	useEffect(() => {
		setRuns(loadRuns());
	}, []);

	const handleDelete = (id: string) => {
		deleteRun(id);
		setRuns(loadRuns());
	};

	return (
		<div className="flex h-full min-h-0 flex-1 flex-col gap-6">
			<header className="flex flex-wrap items-center justify-between gap-3">
				<div className="flex flex-col">
					<h1 className="font-semibold text-foreground text-xl">
						{t("benchmarks.title")}
					</h1>
					<p className="text-muted-foreground text-sm">
						{t("benchmarks.subtitle")}
					</p>
				</div>
				<div className="flex items-center gap-2">
					<Button
						render={<Link to="/benchmarks/charts" />}
						variant="outline"
						size="sm"
					>
						{t("benchmarks.chartGallery")}
					</Button>
					<Button render={<Link to="/benchmarks/new" />}>
						<PlusIcon /> {t("benchmarks.newBenchmark")}
					</Button>
				</div>
			</header>

			<section className="flex flex-col gap-3">
				<div className="flex items-center gap-2 text-foreground text-sm">
					<FlaskConicalIcon className="size-4 text-primary" />
					<span className="font-medium">{t("benchmarks.runPreset")}</span>
				</div>
				<div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
					{PRESETS.map((preset) => (
						<Link
							key={preset.id}
							to="/benchmarks/new"
							search={{ preset: preset.id }}
							className={cn(
								"flex flex-col gap-1 rounded-xl border bg-card/40 p-4 transition",
								"hover:border-primary/60 hover:bg-card/80",
							)}
						>
							<div className="flex items-center justify-between gap-2">
								<span className="truncate font-medium text-sm">
									{preset.label}
								</span>
								<Badge variant="outline" size="sm">
									{preset.backend}
								</Badge>
							</div>
							<span className="text-muted-foreground text-xs">
								{preset.description}
							</span>
							<span className="text-muted-foreground/80 text-xs">
								{t("benchmarks.estimatedMinutes", {
									count: preset.estimatedMinutes,
								})}
							</span>
						</Link>
					))}
				</div>
			</section>

			<section className="flex min-h-0 flex-1 flex-col gap-3">
				<div className="flex items-center justify-between">
					<span className="font-medium text-foreground text-sm">
						{t("benchmarks.recentRuns")}
					</span>
					<span className="text-muted-foreground text-xs">
						{t("benchmarks.storedLocally", { count: runs.length })}
					</span>
				</div>

				{runs.length === 0 ? (
					<div className="flex flex-1 flex-col items-center justify-center gap-3 rounded-2xl border border-dashed bg-card/20 p-12 text-center">
						<FlaskConicalIcon className="size-8 text-muted-foreground/50" />
						<div className="flex flex-col gap-1">
							<span className="font-medium text-foreground text-sm">
								{t("benchmarks.noRunsYet")}
							</span>
							<span className="text-muted-foreground text-xs">
								{t("benchmarks.noRunsHint")}
							</span>
						</div>
						<Button render={<Link to="/benchmarks/new" />} size="sm">
							<PlusIcon /> {t("benchmarks.newBenchmark")}
						</Button>
					</div>
				) : (
					<ul className="flex flex-col gap-2 overflow-y-auto pr-1">
						{runs.map((run) => (
							<RunRow
								key={run.id}
								run={run}
								onDelete={() => handleDelete(run.id)}
							/>
						))}
					</ul>
				)}
			</section>
		</div>
	);
};
