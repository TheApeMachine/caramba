/*
PlacedWidget is one widget anchored at (col, row) with a span. Coordinates
are 0-indexed; col grows right, row grows down. id doubles as the
view-transition-name so move/resize morph between layout states.
*/
export interface PlacedWidget {
	id: string;
	kind: string;
	col: number;
	row: number;
	colSpan: number;
	rowSpan: number;
}

export const GRID_COLS = 4;
export const GRID_ROWS = 2;

/*
Grid wraps the list of placed widgets with the operations the dashboard
performs: place, move, swap, resize, remove. Every mutation returns a new
Grid; the caller decides whether to wrap the swap in a view transition.

The displacement rules are concentrated here so the React layer stays
focused on input and rendering:
  - place / resize: widgets overlapped by the target footprint shift to
    the nearest empty 1x1 cell; if no cell is free, they are dropped.
  - move: pure swap of two anchor cells, spans preserved.
*/
export class Grid {
	readonly widgets: ReadonlyArray<PlacedWidget>;

	constructor(widgets: ReadonlyArray<PlacedWidget> = []) {
		this.widgets = widgets;
	}

	get(col: number, row: number): PlacedWidget | undefined {
		return this.widgets.find((widget) => coversCell(widget, col, row));
	}

	private without(ids: Set<string>): PlacedWidget[] {
		return this.widgets.filter((widget) => !ids.has(widget.id));
	}

	private occupants(
		col: number,
		row: number,
		colSpan: number,
		rowSpan: number,
		ignoreId?: string,
	): PlacedWidget[] {
		const hits = new Set<PlacedWidget>();

		for (let dc = 0; dc < colSpan; dc++) {
			for (let dr = 0; dr < rowSpan; dr++) {
				const found = this.get(col + dc, row + dr);
				if (found && found.id !== ignoreId) hits.add(found);
			}
		}

		return [...hits];
	}

	/*
	resolveOverlap evicts widgets from the footprint and tries to relocate
	each one to the first available 1x1 cell. Widgets with nowhere to go are
	returned as `dropped` so the caller can mirror them back to the toolbox.
	*/
	private resolveOverlap(
		anchor: PlacedWidget,
		displaced: PlacedWidget[],
	): { kept: PlacedWidget[]; dropped: PlacedWidget[] } {
		const displacedIds = new Set(displaced.map((victim) => victim.id));
		const untouched = this.widgets.filter(
			(widget) => widget.id !== anchor.id && !displacedIds.has(widget.id),
		);
		const dropped: PlacedWidget[] = [];
		const relocated: PlacedWidget[] = [];

		for (const victim of displaced) {
			const free = firstFreeCell([...untouched, anchor, ...relocated]);

			if (!free) {
				dropped.push(victim);
				continue;
			}

			relocated.push({
				...victim,
				col: free.col,
				row: free.row,
				colSpan: 1,
				rowSpan: 1,
			});
		}

		return { kept: [...untouched, anchor, ...relocated], dropped };
	}

	place(widget: PlacedWidget): { grid: Grid; dropped: PlacedWidget[] } {
		const displaced = this.occupants(
			widget.col,
			widget.row,
			widget.colSpan,
			widget.rowSpan,
			widget.id,
		);

		const { kept, dropped } = new Grid(
			this.without(new Set([widget.id])),
		).resolveOverlap(widget, displaced);

		return { grid: new Grid(kept), dropped };
	}

	move(
		id: string,
		col: number,
		row: number,
	): { grid: Grid; dropped: PlacedWidget[] } {
		const widget = this.widgets.find((entry) => entry.id === id);
		if (!widget) return { grid: this, dropped: [] };

		const target = this.get(col, row);
		if (target?.id === id) return { grid: this, dropped: [] };

		const anchorCol = target ? target.col : col;
		const anchorRow = target ? target.row : row;

		const colSpan = clamp(widget.colSpan, 1, GRID_COLS - anchorCol);
		const rowSpan = clamp(widget.rowSpan, 1, GRID_ROWS - anchorRow);

		const moved: PlacedWidget = {
			...widget,
			col: anchorCol,
			row: anchorRow,
			colSpan,
			rowSpan,
		};

		return new Grid(this.without(new Set([id]))).place(moved);
	}

	resize(
		id: string,
		colSpan: number,
		rowSpan: number,
	): { grid: Grid; dropped: PlacedWidget[] } {
		const widget = this.widgets.find((entry) => entry.id === id);
		if (!widget) return { grid: this, dropped: [] };

		const clampedColSpan = clamp(colSpan, 1, GRID_COLS - widget.col);
		const clampedRowSpan = clamp(rowSpan, 1, GRID_ROWS - widget.row);

		const resized: PlacedWidget = {
			...widget,
			colSpan: clampedColSpan,
			rowSpan: clampedRowSpan,
		};

		return new Grid(this.without(new Set([id]))).place(resized);
	}

	remove(id: string): Grid {
		return new Grid(this.without(new Set([id])));
	}

	freeCells(): Array<{ col: number; row: number }> {
		const cells: Array<{ col: number; row: number }> = [];

		for (let row = 0; row < GRID_ROWS; row++) {
			for (let col = 0; col < GRID_COLS; col++) {
				if (!this.get(col, row)) cells.push({ col, row });
			}
		}

		return cells;
	}
}

const coversCell = (widget: PlacedWidget, col: number, row: number) =>
	col >= widget.col &&
	col < widget.col + widget.colSpan &&
	row >= widget.row &&
	row < widget.row + widget.rowSpan;

const firstFreeCell = (
	widgets: ReadonlyArray<PlacedWidget>,
): { col: number; row: number } | null => {
	for (let row = 0; row < GRID_ROWS; row++) {
		for (let col = 0; col < GRID_COLS; col++) {
			const occupied = widgets.some((widget) => coversCell(widget, col, row));
			if (!occupied) return { col, row };
		}
	}

	return null;
};

const clamp = (value: number, min: number, max: number) =>
	Math.min(Math.max(value, min), max);
