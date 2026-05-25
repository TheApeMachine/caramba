import { createFileRoute, Link, useParams } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { LiveRun, loadRuns, type RunRecord } from "#/components/benchmarks";
import { Button } from "#/components/ui/button";

export const Route = createFileRoute("/benchmarks/$runId")({
	ssr: false,
	component: BenchmarkRunRoute,
});

function BenchmarkRunRoute() {
	const { runId } = useParams({ from: "/benchmarks/$runId" });
	const { t } = useTranslation();
	const [record, setRecord] = useState<RunRecord | null>(null);
	const [resolved, setResolved] = useState(false);

	useEffect(() => {
		const found = loadRuns().find((entry) => entry.id === runId) ?? null;
		setRecord(found);
		setResolved(true);
	}, [runId]);

	const view = useMemo(() => {
		if (!resolved) return null;
		if (!record)
			return (
				<div className="flex h-full flex-col items-center justify-center gap-3 text-center">
					<h2 className="font-semibold text-foreground text-lg">
						{t("benchmarks.runNotFound")}
					</h2>
					<p className="text-muted-foreground text-sm">
						{t("benchmarks.runNotFoundHint")}
					</p>
					<Button render={<Link to="/benchmarks" />} size="sm">
						{t("benchmarks.backToBenchmarks")}
					</Button>
				</div>
			);
		return <LiveRun runId={runId} spec={record.spec} initialRecord={record} />;
	}, [record, resolved, runId, t]);

	return <div className="flex h-full min-h-0 w-full flex-1 p-4">{view}</div>;
}
