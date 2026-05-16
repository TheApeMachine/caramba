"use client";

import type { ReactNode } from "react";
import { ChartWidget } from "#/components/vega";
import { sparklineSpec } from "#/components/vega/specs/sparkline";
import { cn } from "@/lib/utils";

interface KpiCardProps {
	icon: ReactNode;
	label: string;
	value: string;
	delta?: { value: string; positive?: boolean };
	trend?: number[];
	emphasis?: boolean;
}

/*
KpiCard is the live-run KPI strip primitive. Big number on top, optional
trend sparkline underneath, optional delta badge. `emphasis` lifts one card
visually (used for the primary metric the run is judged on).
*/
export const KpiCard = ({
	icon,
	label,
	value,
	delta,
	trend,
	emphasis,
}: KpiCardProps) => (
	<div
		className={cn(
			"relative flex h-full flex-col gap-2 overflow-hidden rounded-2xl border bg-card/40 p-4",
			emphasis && "border-primary/50 bg-primary/5",
		)}
	>
		<div className="flex items-center justify-between gap-2">
			<div className="flex items-center gap-2 text-muted-foreground text-xs uppercase tracking-wide">
				<span className="text-foreground/70">{icon}</span>
				{label}
			</div>
			{delta ? (
				<span
					className={cn(
						"rounded-full px-2 py-0.5 font-medium text-[10px]",
						delta.positive
							? "bg-success/15 text-success-foreground"
							: "bg-destructive/15 text-destructive-foreground",
					)}
				>
					{delta.value}
				</span>
			) : null}
		</div>
		<div className="font-semibold text-3xl text-foreground tabular-nums">
			{value}
		</div>
		{trend && trend.length > 1 ? (
			<div className="-mb-1 -mx-1 h-10">
				<ChartWidget
					spec={sparklineSpec({
						values: trend,
						color: emphasis ? "oklch(var(--primary))" : "oklch(var(--chart-1))",
					})}
				/>
			</div>
		) : null}
	</div>
);
