"use client";

import { X } from "lucide-react";
import { type ReactNode, useMemo } from "react";
import { Card } from "@/components/ui/card";
import { ChartWidget } from "@/components/vega";
import { cn } from "@/lib/utils";
import type { WidgetDescriptor } from "./registry";

/*
Widget renders one descriptor inside the dashboard's card chrome. A
descriptor either supplies a Vega spec (chart path) or a React node
(arbitrary content). Drag/resize behavior is layered on by the caller;
`overlay` is rendered inside the same hover group so handles can use
group-hover transitions.
*/
interface WidgetProps {
	descriptor: WidgetDescriptor | undefined;
	onRemove?: () => void;
	chrome?: ReactChrome;
	compact?: boolean;
	overlay?: ReactNode;
}

type ReactChrome = "card" | "bare";

export function Widget({
	descriptor,
	onRemove,
	chrome = "card",
	compact = false,
	overlay,
}: WidgetProps) {
	const spec = useMemo(
		() => (descriptor?.build ? descriptor.build() : undefined),
		[descriptor],
	);

	const node = useMemo(
		() => (descriptor?.render ? descriptor.render() : null),
		[descriptor],
	);

	if (!descriptor)
		return (
			<div className="flex h-full w-full items-center justify-center rounded-2xl border bg-card p-3 text-sm text-destructive">
				Unknown widget
			</div>
		);

	const body = spec ? <ChartWidget spec={spec} /> : node;

	if (chrome === "bare")
		return (
			<div className={cn("flex h-full w-full flex-col gap-2 p-2")}>
				<div className="flex items-baseline justify-between gap-2 px-1">
					<div className="text-xs font-semibold">{descriptor.title}</div>
					<div className="text-[10px] text-muted-foreground">
						{descriptor.description}
					</div>
				</div>
				<div
					className={cn(
						"min-h-0 flex-1 overflow-hidden",
						compact && "max-h-40",
					)}
				>
					{body}
				</div>
			</div>
		);

	return (
		<Card className="group relative h-full w-full overflow-hidden p-3">
			<div className="mb-2 flex items-center justify-between">
				<div className="text-xs font-medium text-muted-foreground">
					{descriptor.title}
				</div>
				{onRemove && (
					<button
						type="button"
						onClick={onRemove}
						onDragStart={(event) => event.stopPropagation()}
						className="rounded p-1 text-muted-foreground opacity-0 transition hover:bg-muted hover:text-foreground group-hover:opacity-100"
						aria-label="Remove widget"
					>
						<X className="h-3.5 w-3.5" />
					</button>
				)}
			</div>
			<div className="h-[calc(100%-1.75rem)] min-h-0 w-full overflow-auto">
				{body}
			</div>
			{overlay}
		</Card>
	);
}
