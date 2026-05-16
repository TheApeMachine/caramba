"use client";

import { X } from "lucide-react";
import type { CSSProperties, DragEvent } from "react";
import { useMemo, useRef, useState } from "react";
import { Card } from "@/components/ui/card";
import { ChartWidget, VegaProvider } from "@/components/vega";
import { cn } from "@/lib/utils";
import { GRID_COLS, GRID_ROWS, Grid, type PlacedWidget } from "./grid";
import {
	type WidgetDescriptor,
	widgetByKind,
	widgetRegistry,
} from "./registry";
import { ResizeHandle } from "./resize-handle";

let widgetCounter = 0;
const newWidgetId = (kind: string) => `w-${kind}-${++widgetCounter}`;

type DragPayload =
	| { source: "toolbox"; kind: string; id: string }
	| { source: "slot"; widgetId: string }
	| { source: "resize"; widgetId: string };

/*
startMorph wraps a state mutation in a View Transition when supported, so
moves, swaps, and resizes animate between the old and new layout positions.
*/
const startMorph = (mutate: () => void) => {
	const doc = document as Document & {
		startViewTransition?: (cb: () => void) => unknown;
	};

	if (typeof doc.startViewTransition === "function") {
		doc.startViewTransition(mutate);
		return;
	}

	mutate();
};

const initialToolboxIds = () =>
	widgetRegistry.reduce<Record<string, string>>((acc, descriptor) => {
		acc[descriptor.kind] = newWidgetId(descriptor.kind);
		return acc;
	}, {});

/*
Dashboard is a 4x2 grid of slots plus a toolbox of widget previews.
Toolbox previews drag into empty cells to mint widgets; widget-to-cell
drags move/swap; the corner handle drags to any cell to resize, using
the very same drag system as moves — the drop target's (col, row)
defines the new span relative to the widget's anchor. Absorbed widgets
shift to the nearest free cell, or return to the toolbox if the grid is
full.
*/
export const Dashboard = () => {
	const [grid, setGrid] = useState<Grid>(() => new Grid());
	const [toolboxIds, setToolboxIds] =
		useState<Record<string, string>>(initialToolboxIds);
	const [resizeHover, setResizeHover] = useState<{
		widgetId: string;
		col: number;
		row: number;
	} | null>(null);

	const payloadRef = useRef<DragPayload | null>(null);

	const restock = (kinds: string[]) => {
		if (kinds.length === 0) return;

		setToolboxIds((current) => {
			const next = { ...current };
			for (const kind of kinds) next[kind] = newWidgetId(kind);
			return next;
		});
	};

	const placeFromToolbox = (
		kind: string,
		id: string,
		col: number,
		row: number,
	) => {
		startMorph(() => {
			const { grid: next, dropped } = grid.place({
				id,
				kind,
				col,
				row,
				colSpan: 1,
				rowSpan: 1,
			});

			setGrid(next);
			restock([kind, ...dropped.map((widget) => widget.kind)]);
		});
	};

	const moveWidget = (widgetId: string, col: number, row: number) => {
		startMorph(() => setGrid(grid.move(widgetId, col, row)));
	};

	const resizeWidget = (widgetId: string, col: number, row: number) => {
		const widget = grid.widgets.find((entry) => entry.id === widgetId);
		if (!widget) return;

		const colSpan = Math.max(1, col - widget.col + 1);
		const rowSpan = Math.max(1, row - widget.row + 1);

		startMorph(() => {
			const { grid: next, dropped } = grid.resize(widgetId, colSpan, rowSpan);
			setGrid(next);
			restock(dropped.map((entry) => entry.kind));
		});
	};

	const removeWidget = (widgetId: string) => {
		startMorph(() => setGrid(grid.remove(widgetId)));
	};

	const handleCellDrop = (col: number, row: number) => {
		const payload = payloadRef.current;
		payloadRef.current = null;
		setResizeHover(null);

		if (!payload) return;

		if (payload.source === "resize") {
			resizeWidget(payload.widgetId, col, row);
			return;
		}

		if (payload.source === "slot") {
			moveWidget(payload.widgetId, col, row);
			return;
		}

		if (grid.get(col, row)) return;

		placeFromToolbox(payload.kind, payload.id, col, row);
	};

	const handleCellEnter = (col: number, row: number) => {
		const payload = payloadRef.current;
		if (payload?.source !== "resize") return;

		setResizeHover({ widgetId: payload.widgetId, col, row });
	};

	const isResizing = payloadRef.current?.source === "resize";

	const resizeAnchor = useMemo(() => {
		if (!resizeHover) return null;
		return grid.widgets.find((widget) => widget.id === resizeHover.widgetId);
	}, [grid, resizeHover]);

	const freeCells = useMemo(() => grid.freeCells(), [grid]);

	return (
		<VegaProvider>
			<div className="flex h-full w-full gap-6">
				<aside className="flex w-64 shrink-0 flex-col gap-3 overflow-y-auto rounded-2xl border bg-card/40 p-3">
					<h3 className="px-1 pb-1 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
						Toolbox
					</h3>
					{widgetRegistry.map((descriptor) => (
						<ToolboxPreview
							key={descriptor.kind}
							descriptor={descriptor}
							previewId={toolboxIds[descriptor.kind]}
							onDragStart={(payload) => {
								payloadRef.current = payload;
							}}
						/>
					))}
				</aside>

				<div
					className="relative grid h-full flex-1 gap-4"
					style={{
						gridTemplateColumns: `repeat(${GRID_COLS}, minmax(0, 1fr))`,
						gridTemplateRows: `repeat(${GRID_ROWS}, minmax(220px, 1fr))`,
					}}
				>
					{freeCells.map(({ col, row }) => (
						<EmptyCell
							key={`empty-${col}-${row}`}
							col={col}
							row={row}
							isResizing={isResizing}
							onCellEnter={() => handleCellEnter(col, row)}
							onDrop={() => handleCellDrop(col, row)}
						/>
					))}

					{grid.widgets.map((widget) => (
						<PlacedWidgetView
							key={widget.id}
							widget={widget}
							isResizing={isResizing}
							onDragStart={(payload) => {
								payloadRef.current = payload;
							}}
							onCellEnter={() => handleCellEnter(widget.col, widget.row)}
							onDrop={() => handleCellDrop(widget.col, widget.row)}
							onRemove={() => removeWidget(widget.id)}
						/>
					))}

					{resizeAnchor && resizeHover && (
						<ResizeGhost
							col={resizeAnchor.col}
							row={resizeAnchor.row}
							colSpan={Math.max(1, resizeHover.col - resizeAnchor.col + 1)}
							rowSpan={Math.max(1, resizeHover.row - resizeAnchor.row + 1)}
						/>
					)}
				</div>
			</div>
		</VegaProvider>
	);
};

interface ToolboxPreviewProps {
	descriptor: WidgetDescriptor;
	previewId: string;
	onDragStart: (payload: DragPayload) => void;
}

const ToolboxPreview = ({
	descriptor,
	previewId,
	onDragStart,
}: ToolboxPreviewProps) => {
	const spec = useMemo(() => descriptor.build(), [descriptor]);

	const handleDragStart = (event: DragEvent<HTMLButtonElement>) => {
		event.dataTransfer.effectAllowed = "copyMove";
		event.dataTransfer.setData("text/plain", previewId);
		onDragStart({ source: "toolbox", kind: descriptor.kind, id: previewId });
		event.currentTarget.classList.add("dash-dragging");
	};

	const handleDragEnd = (event: DragEvent<HTMLButtonElement>) => {
		event.currentTarget.classList.remove("dash-dragging");
	};

	return (
		<button
			type="button"
			draggable
			data-dash-drag
			data-dash-drag-preview
			aria-label={`Drag ${descriptor.title} widget to a slot`}
			onDragStart={handleDragStart}
			onDragEnd={handleDragEnd}
			style={{ viewTransitionName: previewId } as CSSProperties}
			className="block w-full overflow-hidden rounded-xl border bg-background/80 p-0 text-left shadow-sm"
		>
			<WidgetBody descriptor={descriptor} spec={spec} compact />
		</button>
	);
};

interface EmptyCellProps {
	col: number;
	row: number;
	isResizing: boolean;
	onCellEnter: () => void;
	onDrop: () => void;
}

const EmptyCell = ({
	col,
	row,
	isResizing,
	onCellEnter,
	onDrop,
}: EmptyCellProps) => {
	const handleDragOver = (event: DragEvent<HTMLElement>) => {
		event.preventDefault();
		event.dataTransfer.dropEffect = "move";
	};

	const handleDragEnter = (event: DragEvent<HTMLElement>) => {
		event.preventDefault();
		onCellEnter();
		if (!isResizing) event.currentTarget.classList.add("dash-over");
	};

	const handleDragLeave = (event: DragEvent<HTMLElement>) => {
		event.currentTarget.classList.remove("dash-over");
	};

	const handleDrop = (event: DragEvent<HTMLElement>) => {
		event.preventDefault();
		event.stopPropagation();
		event.currentTarget.classList.remove("dash-over");
		onDrop();
	};

	return (
		<section
			data-dash-drag
			aria-label={`Empty slot column ${col + 1} row ${row + 1}`}
			onDragOver={handleDragOver}
			onDragEnter={handleDragEnter}
			onDragLeave={handleDragLeave}
			onDrop={handleDrop}
			style={{
				gridColumn: `${col + 1} / span 1`,
				gridRow: `${row + 1} / span 1`,
			}}
			className="flex items-center justify-center rounded-2xl border border-dashed border-muted-foreground/30 bg-muted/10 text-xs text-muted-foreground/60"
		>
			Drop a widget here
		</section>
	);
};

interface PlacedWidgetViewProps {
	widget: PlacedWidget;
	isResizing: boolean;
	onDragStart: (payload: DragPayload) => void;
	onCellEnter: () => void;
	onDrop: () => void;
	onRemove: () => void;
}

const PlacedWidgetView = ({
	widget,
	isResizing,
	onDragStart,
	onCellEnter,
	onDrop,
	onRemove,
}: PlacedWidgetViewProps) => {
	const descriptor = widgetByKind.get(widget.kind);
	const spec = useMemo(() => descriptor?.build(), [descriptor]);

	const handleDragStart = (event: DragEvent<HTMLElement>) => {
		const target = event.target as HTMLElement | null;

		if (target?.closest("[data-dash-resize]")) return;

		event.dataTransfer.effectAllowed = "move";
		event.dataTransfer.setData("text/plain", widget.id);
		onDragStart({ source: "slot", widgetId: widget.id });
		event.currentTarget.classList.add("dash-dragging");
	};

	const handleDragEnd = (event: DragEvent<HTMLElement>) => {
		event.currentTarget.classList.remove("dash-dragging");
	};

	const handleDragOver = (event: DragEvent<HTMLElement>) => {
		event.preventDefault();
		event.dataTransfer.dropEffect = "move";
	};

	const handleDragEnter = (event: DragEvent<HTMLElement>) => {
		event.preventDefault();
		onCellEnter();
		if (!isResizing) event.currentTarget.classList.add("dash-over");
	};

	const handleDragLeave = (event: DragEvent<HTMLElement>) => {
		event.currentTarget.classList.remove("dash-over");
	};

	const handleDrop = (event: DragEvent<HTMLElement>) => {
		event.preventDefault();
		event.stopPropagation();
		event.currentTarget.classList.remove("dash-over");
		onDrop();
	};

	const placement: CSSProperties = {
		gridColumn: `${widget.col + 1} / span ${widget.colSpan}`,
		gridRow: `${widget.row + 1} / span ${widget.rowSpan}`,
		viewTransitionName: widget.id,
	};

	if (!descriptor || !spec) {
		return (
			<article
				draggable
				data-dash-drag
				aria-label={`Unknown widget ${widget.kind}`}
				onDragStart={handleDragStart}
				onDragEnd={handleDragEnd}
				onDragOver={handleDragOver}
				onDragEnter={handleDragEnter}
				onDragLeave={handleDragLeave}
				onDrop={handleDrop}
				style={placement}
				className="flex items-center justify-center rounded-2xl border bg-card p-3 text-sm text-destructive"
			>
				Unknown widget: {widget.kind}
			</article>
		);
	}

	return (
		<article
			draggable
			data-dash-drag
			aria-label={`${descriptor.title} widget`}
			onDragStart={handleDragStart}
			onDragEnd={handleDragEnd}
			onDragOver={handleDragOver}
			onDragEnter={handleDragEnter}
			onDragLeave={handleDragLeave}
			onDrop={handleDrop}
			style={placement}
			className="flex min-h-0 min-w-0"
		>
			<Card className={cn("group relative h-full w-full overflow-hidden p-3")}>
				<div className="mb-2 flex items-center justify-between">
					<div className="text-xs font-medium text-muted-foreground">
						{descriptor.title}
					</div>
					<button
						type="button"
						onClick={onRemove}
						onDragStart={(event) => event.stopPropagation()}
						className="rounded p-1 text-muted-foreground opacity-0 transition hover:bg-muted hover:text-foreground group-hover:opacity-100"
						aria-label="Remove widget"
					>
						<X className="h-3.5 w-3.5" />
					</button>
				</div>
				<div className="h-[calc(100%-1.75rem)] w-full">
					<ChartWidget spec={spec} />
				</div>
				<ResizeHandle
					widgetId={widget.id}
					onDragStart={() =>
						onDragStart({ source: "resize", widgetId: widget.id })
					}
				/>
			</Card>
		</article>
	);
};

interface WidgetBodyProps {
	descriptor: WidgetDescriptor;
	spec: ReturnType<WidgetDescriptor["build"]>;
	compact?: boolean;
}

interface ResizeGhostProps {
	col: number;
	row: number;
	colSpan: number;
	rowSpan: number;
}

const ResizeGhost = ({ col, row, colSpan, rowSpan }: ResizeGhostProps) => {
	return (
		<div
			aria-hidden
			style={{
				gridColumn: `${col + 1} / span ${colSpan}`,
				gridRow: `${row + 1} / span ${rowSpan}`,
			}}
			className="pointer-events-none rounded-2xl border-2 border-dashed border-primary/70 bg-primary/10"
		/>
	);
};

const WidgetBody = ({ descriptor, spec, compact }: WidgetBodyProps) => {
	return (
		<div className={cn("flex flex-col gap-2 p-2", compact ? "h-40" : "h-full")}>
			<div className="flex items-baseline justify-between gap-2 px-1">
				<div className="text-xs font-semibold">{descriptor.title}</div>
				<div className="text-[10px] text-muted-foreground">
					{descriptor.description}
				</div>
			</div>
			<div className="min-h-0 flex-1">
				<ChartWidget spec={spec} />
			</div>
		</div>
	);
};
