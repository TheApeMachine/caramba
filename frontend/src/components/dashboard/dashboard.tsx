"use client";

import { PencilIcon } from "lucide-react";
import { type CSSProperties, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import {
	DragDropProvider,
	Draggable,
	DropTarget,
} from "@/components/ui/drag-drop";
import { VegaProvider } from "@/components/vega";
import { GRID_COLS, GRID_ROWS, Grid, type PlacedWidget } from "./grid";
import {
	widgetByKind as defaultWidgetByKind,
	widgetRegistry as defaultWidgetRegistry,
	type WidgetDescriptor,
} from "./registry";
import { ResizeHandle } from "./resize-handle";
import { Widget } from "./widget";

/*
DragPayload is the grid-specific union. The drag-drop primitive is generic;
this is what flows through it for the dashboard.
*/
type DragPayload =
	| { source: "toolbox"; kind: string; instanceId: string }
	| { source: "slot"; widgetId: string }
	| { source: "resize"; widgetId: string };

let widgetCounter = 0;
const newWidgetId = (kind: string) => `w-${kind}-${++widgetCounter}`;

const initialToolboxIds = (registry: ReadonlyArray<WidgetDescriptor>) =>
	registry.reduce<Record<string, string>>((acc, descriptor) => {
		acc[descriptor.kind] = newWidgetId(descriptor.kind);
		return acc;
	}, {});

export interface DashboardProps {
	widgets?: ReadonlyArray<WidgetDescriptor>;
	initialLayout?: ReadonlyArray<PlacedWidget>;
}

/*
Dashboard is a 4x2 grid of cells composed atop the generic drag-drop
primitive. Each cell is a DropTarget; cells holding a widget render
that widget inside a Draggable, plus a resize handle whose own drag is
also routed through the same payload union. Toolbox tiles are Draggables.
All grid mutations (place / move / resize / remove) happen in this file;
the visual primitives know nothing about them.
*/
export const Dashboard = ({ widgets, initialLayout }: DashboardProps = {}) => {
	const registry = widgets ?? defaultWidgetRegistry;

	const byKind = useMemo(
		() =>
			widgets
				? new Map(widgets.map((descriptor) => [descriptor.kind, descriptor]))
				: defaultWidgetByKind,
		[widgets],
	);

	const [grid, setGrid] = useState(() => new Grid(initialLayout ?? []));
	const [toolboxIds, setToolboxIds] = useState(() =>
		initialToolboxIds(registry),
	);
	const [resizeHover, setResizeHover] = useState<{
		widgetId: string;
		col: number;
		row: number;
	} | null>(null);
	const [editing, setEditing] = useState(false);

	const restock = (kinds: string[]) => {
		if (kinds.length === 0) return;
		setToolboxIds((current) => {
			const next = { ...current };
			for (const kind of kinds) next[kind] = newWidgetId(kind);
			return next;
		});
	};

	const handleDrop = (col: number, row: number, payload: DragPayload) => {
		setResizeHover(null);

		if (payload.source === "resize") {
			const widget = grid.widgets.find(
				(entry) => entry.id === payload.widgetId,
			);
			if (!widget) return;

			const colSpan = Math.max(1, col - widget.col + 1);
			const rowSpan = Math.max(1, row - widget.row + 1);
			const { grid: next, dropped } = grid.resize(
				payload.widgetId,
				colSpan,
				rowSpan,
			);
			setGrid(next);
			restock(dropped.map((entry) => entry.kind));
			return;
		}

		if (payload.source === "slot") {
			const { grid: next, dropped } = grid.move(payload.widgetId, col, row);
			setGrid(next);
			restock(dropped.map((entry) => entry.kind));
			return;
		}

		if (grid.get(col, row)) return;

		const { grid: next, dropped } = grid.place({
			id: payload.instanceId,
			kind: payload.kind,
			col,
			row,
			colSpan: 1,
			rowSpan: 1,
		});
		setGrid(next);
		restock([payload.kind, ...dropped.map((widget) => widget.kind)]);
	};

	const handleEnter = (col: number, row: number, payload: DragPayload) => {
		if (payload.source !== "resize") return;
		setResizeHover({ widgetId: payload.widgetId, col, row });
	};

	const removeWidget = (widgetId: string) => setGrid(grid.remove(widgetId));

	const widgetAt = useMemo(() => {
		const map = new Map<string, PlacedWidget>();
		for (const widget of grid.widgets)
			map.set(`${widget.col},${widget.row}`, widget);
		return map;
	}, [grid]);

	const occupied = useMemo(() => {
		const set = new Set<string>();
		for (const widget of grid.widgets) {
			for (let dc = 0; dc < widget.colSpan; dc++)
				for (let dr = 0; dr < widget.rowSpan; dr++)
					set.add(`${widget.col + dc},${widget.row + dr}`);
		}
		return set;
	}, [grid]);

	const resizeAnchor = useMemo(() => {
		if (!resizeHover) return null;
		return grid.widgets.find((widget) => widget.id === resizeHover.widgetId);
	}, [grid, resizeHover]);

	const cells: Array<{ col: number; row: number }> = [];
	for (let row = 0; row < GRID_ROWS; row++)
		for (let col = 0; col < GRID_COLS; col++) cells.push({ col, row });

	return (
		<DragDropProvider>
			<VegaProvider>
				<div className="relative flex h-full w-full gap-6">
					<Button
						type="button"
						size="sm"
						variant={editing ? "default" : "outline"}
						onClick={() => setEditing((current) => !current)}
						className="absolute right-0 top-0 z-10"
					>
						<PencilIcon /> {editing ? "Done" : "Edit"}
					</Button>

					{editing && (
						<aside className="flex w-64 shrink-0 flex-col gap-3 overflow-y-auto rounded-2xl border bg-card/40 p-3">
							<h3 className="px-1 pb-1 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
								Toolbox
							</h3>
							{registry.map((descriptor) => {
								const previewId = toolboxIds[descriptor.kind];

								return (
									<Draggable<DragPayload>
										key={descriptor.kind}
										as="button"
										ariaLabel={`Drag ${descriptor.title} widget to a slot`}
										className="block w-full overflow-hidden rounded-xl border bg-background/80 p-0 text-left shadow-sm"
										payload={{
											source: "toolbox",
											kind: descriptor.kind,
											instanceId: previewId,
										}}
										style={{ viewTransitionName: previewId } as CSSProperties}
									>
										<Widget descriptor={descriptor} chrome="bare" compact />
									</Draggable>
								);
							})}
						</aside>
					)}

					<div
						className="relative grid h-full flex-1 gap-4"
						style={{
							gridTemplateColumns: `repeat(${GRID_COLS}, minmax(0, 1fr))`,
							gridTemplateRows: `repeat(${GRID_ROWS}, minmax(220px, 1fr))`,
						}}
					>
						{cells.map(({ col, row }) => {
							const widget = widgetAt.get(`${col},${row}`);

							if (widget) {
								const widgetEl = (
									<Widget
										descriptor={byKind.get(widget.kind)}
										onRemove={
											editing ? () => removeWidget(widget.id) : undefined
										}
										overlay={
											editing ? <ResizeHandle widgetId={widget.id} /> : null
										}
									/>
								);

								const placement: CSSProperties = {
									gridColumn: `${widget.col + 1} / span ${widget.colSpan}`,
									gridRow: `${widget.row + 1} / span ${widget.rowSpan}`,
									viewTransitionName: widget.id,
								};

								if (!editing)
									return (
										<article
											key={widget.id}
											style={placement}
											className="flex min-h-0 min-w-0"
										>
											{widgetEl}
										</article>
									);

								return (
									<DropTarget<DragPayload>
										key={widget.id}
										morph
										as="article"
										ariaLabel={`${byKind.get(widget.kind)?.title ?? widget.kind} widget`}
										className="flex min-h-0 min-w-0"
										style={placement}
										onDrop={(payload) => handleDrop(col, row, payload)}
										onEnter={(payload) => handleEnter(col, row, payload)}
									>
										<Draggable<DragPayload>
											className="flex h-full w-full"
											payload={{ source: "slot", widgetId: widget.id }}
											disabled={(target) =>
												!!(target as HTMLElement | null)?.closest(
													"[data-dash-resize]",
												)
											}
										>
											{widgetEl}
										</Draggable>
									</DropTarget>
								);
							}

							if (occupied.has(`${col},${row}`)) return null;
							if (!editing) return null;

							return (
								<DropTarget<DragPayload>
									key={`empty-${col}-${row}`}
									morph
									as="section"
									ariaLabel={`Empty slot column ${col + 1} row ${row + 1}`}
									className="flex items-center justify-center rounded-2xl border border-dashed border-muted-foreground/30 bg-muted/10 text-xs text-muted-foreground/60"
									style={{
										gridColumn: `${col + 1} / span 1`,
										gridRow: `${row + 1} / span 1`,
									}}
									onDrop={(payload) => handleDrop(col, row, payload)}
									onEnter={(payload) => handleEnter(col, row, payload)}
								>
									Drop a widget here
								</DropTarget>
							);
						})}

						{resizeAnchor && resizeHover && (
							<div
								aria-hidden
								style={{
									gridColumn: `${resizeAnchor.col + 1} / span ${Math.max(
										1,
										resizeHover.col - resizeAnchor.col + 1,
									)}`,
									gridRow: `${resizeAnchor.row + 1} / span ${Math.max(
										1,
										resizeHover.row - resizeAnchor.row + 1,
									)}`,
								}}
								className="pointer-events-none rounded-2xl border-2 border-dashed border-primary/70 bg-primary/10"
							/>
						)}
					</div>
				</div>
			</VegaProvider>
		</DragDropProvider>
	);
};
