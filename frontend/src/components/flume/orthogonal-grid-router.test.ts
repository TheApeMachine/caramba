import { describe, expect, it } from "vitest";
import {
	buildRoutingGridFromObstacles,
	findGridPath,
	formatOrthogonalSvgPath,
	GRID_CELL_SIZE,
	OBSTACLE_GRID_PADDING,
	PORT_EXIT_STUB,
	RoutingGrid,
	routeOrthogonalWithGrid,
	simplifyOrthogonalPoints,
} from "#/components/flume/orthogonal-grid-router";

describe("findGridPath", () => {
	it("routes around a blocked rectangle", () => {
		const grid = new RoutingGrid();
		grid.markRect(
			{ left: 96, right: 160, top: 96, bottom: 160 },
			OBSTACLE_GRID_PADDING,
		);

		const start = grid.worldToCell(40, 128);
		const goal = grid.worldToCell(220, 128);
		const path = findGridPath(grid, start, goal);

		expect(path).not.toBeNull();
		expect(path?.length).toBeGreaterThan(2);

		for (const cell of path ?? []) {
			if (cell.cellX === start.cellX && cell.cellY === start.cellY) {
				continue;
			}

			if (cell.cellX === goal.cellX && cell.cellY === goal.cellY) {
				continue;
			}

			expect(grid.isOccupied(cell.cellX, cell.cellY)).toBe(false);
		}
	});
});

describe("routeOrthogonalWithGrid", () => {
	it("builds an SVG path with port stubs and grid routing", () => {
		const from = { x: 100, y: 120 };
		const to = { x: 420, y: 120 };
		const grid = buildRoutingGridFromObstacles([
			{ left: 220, right: 300, top: 80, bottom: 160 },
		]);

		const path = routeOrthogonalWithGrid(from, to, grid);

		expect(path).not.toBeNull();
		expect(path).toMatch(/^M /);
		expect(path).toContain(`L ${from.x + PORT_EXIT_STUB} ${from.y}`);
		expect(path).toContain(`L ${to.x} ${to.y}`);
	});

	it("simplifies collinear bend points", () => {
		const simplified = simplifyOrthogonalPoints([
			{ x: 0, y: 0 },
			{ x: 32, y: 0 },
			{ x: 64, y: 0 },
			{ x: 64, y: 32 },
		]);

		expect(simplified).toEqual([
			{ x: 0, y: 0 },
			{ x: 64, y: 0 },
			{ x: 64, y: 32 },
		]);
	});

	it("formats orthogonal polylines for SVG", () => {
		expect(
			formatOrthogonalSvgPath([
				{ x: 0, y: 0 },
				{ x: 64, y: 0 },
				{ x: 64, y: 32 },
			]),
		).toBe("M 0 0 L 64 0 L 64 32");
	});
});

describe("RoutingGrid", () => {
	it("maps world coordinates to grid cells", () => {
		const grid = new RoutingGrid();
		const cell = grid.worldToCell(GRID_CELL_SIZE + 4, GRID_CELL_SIZE * 2 + 10);

		expect(cell).toEqual({ cellX: 1, cellY: 2 });
		expect(grid.cellCenterWorld(cell.cellX, cell.cellY)).toEqual({
			x: 48,
			y: 80,
		});
	});
});
