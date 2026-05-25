import { describe, expect, it } from "vitest";
import {
	type BentoTileSpec,
	GRID_COLS,
	GRID_ROWS,
	packBentoLayout,
} from "./grid";

const researchTiles: BentoTileSpec[] = [
	{
		id: "r-actions",
		kind: "research-actions",
		colSpan: 1,
		rowSpan: 2,
		priority: 100,
	},
	{
		id: "r-stats",
		kind: "research-stats",
		colSpan: 2,
		rowSpan: 1,
		priority: 80,
	},
	{
		id: "r-profile",
		kind: "research-profile",
		colSpan: 1,
		rowSpan: 1,
		priority: 70,
	},
	{
		id: "r-projects",
		kind: "research-projects",
		colSpan: 1,
		rowSpan: 1,
		priority: 60,
	},
	{
		id: "r-todos",
		kind: "research-todos",
		colSpan: 1,
		rowSpan: 1,
		priority: 50,
	},
	{
		id: "r-activity",
		kind: "research-activity",
		colSpan: 1,
		rowSpan: 1,
		priority: 40,
	},
];

describe("packBentoLayout", () => {
	it("packs the research dashboard tiles without overlap", () => {
		const layout = packBentoLayout(researchTiles);

		expect(layout).toHaveLength(6);

		const occupied = new Set<string>();

		for (const widget of layout) {
			expect(widget.col).toBeGreaterThanOrEqual(0);
			expect(widget.row).toBeGreaterThanOrEqual(0);
			expect(widget.col + widget.colSpan).toBeLessThanOrEqual(GRID_COLS);
			expect(widget.row + widget.rowSpan).toBeLessThanOrEqual(GRID_ROWS);

			for (let deltaCol = 0; deltaCol < widget.colSpan; deltaCol++) {
				for (let deltaRow = 0; deltaRow < widget.rowSpan; deltaRow++) {
					const key = `${widget.col + deltaCol},${widget.row + deltaRow}`;
					expect(occupied.has(key)).toBe(false);
					occupied.add(key);
				}
			}
		}
	});

	it("places the quick-actions column on the left spanning two rows", () => {
		const layout = packBentoLayout(researchTiles);
		const actions = layout.find((widget) => widget.kind === "research-actions");

		expect(actions).toMatchObject({
			col: 0,
			row: 0,
			colSpan: 1,
			rowSpan: 2,
		});
	});

	it("omits tiles that cannot fit", () => {
		const overflow: BentoTileSpec[] = [
			...researchTiles,
			{
				id: "r-extra",
				kind: "research-extra",
				colSpan: 2,
				rowSpan: 2,
				priority: 1,
			},
		];

		const layout = packBentoLayout(overflow);

		expect(layout.length).toBeLessThan(overflow.length);
		expect(layout.some((widget) => widget.kind === "research-extra")).toBe(
			false,
		);
	});
});
