import type { Coordinate } from "#/components/flume/types";

export type GridObstacleRect = {
	readonly left: number;
	readonly right: number;
	readonly top: number;
	readonly bottom: number;
};

export const GRID_CELL_SIZE = 32;
export const PORT_EXIT_STUB = 40;
export const OBSTACLE_GRID_PADDING = 16;
const ASTAR_ITERATION_LIMIT = 12_000;

export type GridCell = {
	cellX: number;
	cellY: number;
};

/*
RoutingGrid tracks occupied cells for orthogonal A* edge routing.
*/
export class RoutingGrid {
	private occupied = new Set<string>();

	clear(): void {
		this.occupied.clear();
	}

	cellKey(cellX: number, cellY: number): string {
		return `${cellX},${cellY}`;
	}

	worldToCell(x: number, y: number): GridCell {
		return {
			cellX: Math.floor(x / GRID_CELL_SIZE),
			cellY: Math.floor(y / GRID_CELL_SIZE),
		};
	}

	cellCenterWorld(cellX: number, cellY: number): Coordinate {
		return {
			x: cellX * GRID_CELL_SIZE + GRID_CELL_SIZE / 2,
			y: cellY * GRID_CELL_SIZE + GRID_CELL_SIZE / 2,
		};
	}

	markRect(rect: GridObstacleRect, padding = 0): void {
		const left = rect.left - padding;
		const right = rect.right + padding;
		const top = rect.top - padding;
		const bottom = rect.bottom + padding;
		const minCell = this.worldToCell(left, top);
		const maxCell = this.worldToCell(right, bottom);

		for (let cellX = minCell.cellX; cellX <= maxCell.cellX; cellX++) {
			for (let cellY = minCell.cellY; cellY <= maxCell.cellY; cellY++) {
				this.occupied.add(this.cellKey(cellX, cellY));
			}
		}
	}

	isOccupied(cellX: number, cellY: number): boolean {
		return this.occupied.has(this.cellKey(cellX, cellY));
	}

	getOccupiedCells(): ReadonlySet<string> {
		return this.occupied;
	}
}

export const buildRoutingGridFromObstacles = (
	obstacles: ReadonlyArray<GridObstacleRect>,
	padding = OBSTACLE_GRID_PADDING,
): RoutingGrid => {
	const grid = new RoutingGrid();

	for (const rect of obstacles) {
		grid.markRect(rect, padding);
	}

	return grid;
};

export const buildRoutingGridFromObstacleMap = (
	obstaclesById: Map<string, GridObstacleRect>,
	excludeNodeIds: ReadonlySet<string>,
	padding = OBSTACLE_GRID_PADDING,
): RoutingGrid => {
	const grid = new RoutingGrid();

	for (const [nodeId, rect] of obstaclesById) {
		if (excludeNodeIds.has(nodeId)) {
			continue;
		}

		grid.markRect(rect, padding);
	}

	return grid;
};

const gridCellKey = (cell: GridCell) => `${cell.cellX},${cell.cellY}`;

const manhattanDistance = (left: GridCell, right: GridCell) =>
	Math.abs(left.cellX - right.cellX) + Math.abs(left.cellY - right.cellY);

const neighborsOf = (cell: GridCell): GridCell[] => [
	{ cellX: cell.cellX + 1, cellY: cell.cellY },
	{ cellX: cell.cellX - 1, cellY: cell.cellY },
	{ cellX: cell.cellX, cellY: cell.cellY + 1 },
	{ cellX: cell.cellX, cellY: cell.cellY - 1 },
];

const isWalkable = (
	grid: RoutingGrid,
	cell: GridCell,
	start: GridCell,
	goal: GridCell,
): boolean => {
	if (cell.cellX === start.cellX && cell.cellY === start.cellY) {
		return true;
	}

	if (cell.cellX === goal.cellX && cell.cellY === goal.cellY) {
		return true;
	}

	return !grid.isOccupied(cell.cellX, cell.cellY);
};

const reconstructGridPath = (
	cameFrom: Map<string, GridCell>,
	current: GridCell,
): GridCell[] => {
	const path: GridCell[] = [current];
	let cursor = current;

	while (cameFrom.has(gridCellKey(cursor))) {
		const previous = cameFrom.get(gridCellKey(cursor));

		if (!previous) {
			break;
		}

		cursor = previous;
		path.unshift(cursor);
	}

	return path;
};

/*
findGridPath runs A* over four-connected grid cells. Returns null when no route
exists within the iteration budget.
*/
export const findGridPath = (
	grid: RoutingGrid,
	start: GridCell,
	goal: GridCell,
): GridCell[] | null => {
	const open: Array<{ cell: GridCell; fScore: number }> = [];
	const cameFrom = new Map<string, GridCell>();
	const gScore = new Map<string, number>();
	const startKey = gridCellKey(start);

	gScore.set(startKey, 0);
	open.push({
		cell: start,
		fScore: manhattanDistance(start, goal),
	});

	for (let iteration = 0; iteration < ASTAR_ITERATION_LIMIT; iteration++) {
		if (open.length === 0) {
			return null;
		}

		let bestIndex = 0;

		for (let index = 1; index < open.length; index++) {
			if (open[index].fScore < open[bestIndex].fScore) {
				bestIndex = index;
			}
		}

		const current = open.splice(bestIndex, 1)[0].cell;

		if (current.cellX === goal.cellX && current.cellY === goal.cellY) {
			return reconstructGridPath(cameFrom, current);
		}

		const currentKey = gridCellKey(current);
		const currentGScore = gScore.get(currentKey) ?? Number.POSITIVE_INFINITY;

		for (const neighbor of neighborsOf(current)) {
			if (!isWalkable(grid, neighbor, start, goal)) {
				continue;
			}

			const neighborKey = gridCellKey(neighbor);
			const tentativeGScore = currentGScore + 1;

			if (
				tentativeGScore >= (gScore.get(neighborKey) ?? Number.POSITIVE_INFINITY)
			) {
				continue;
			}

			cameFrom.set(neighborKey, current);
			gScore.set(neighborKey, tentativeGScore);
			open.push({
				cell: neighbor,
				fScore: tentativeGScore + manhattanDistance(neighbor, goal),
			});
		}
	}

	return null;
};

const sameCoordinate = (left: Coordinate, right: Coordinate) =>
	left.x === right.x && left.y === right.y;

const isCollinear = (
	first: Coordinate,
	second: Coordinate,
	third: Coordinate,
): boolean => {
	const sameX = first.x === second.x && second.x === third.x;
	const sameY = first.y === second.y && second.y === third.y;

	return sameX || sameY;
};

export const simplifyOrthogonalPoints = (
	points: ReadonlyArray<Coordinate>,
): Coordinate[] => {
	if (points.length <= 2) {
		return [...points];
	}

	const simplified: Coordinate[] = [points[0]];

	for (let index = 1; index < points.length - 1; index++) {
		const previous = simplified[simplified.length - 1];
		const current = points[index];
		const next = points[index + 1];

		if (isCollinear(previous, current, next)) {
			continue;
		}

		simplified.push(current);
	}

	simplified.push(points[points.length - 1]);

	return simplified.filter(
		(point, index, array) =>
			index === 0 || !sameCoordinate(point, array[index - 1]),
	);
};

export const formatOrthogonalSvgPath = (
	points: ReadonlyArray<Coordinate>,
): string => {
	if (points.length === 0) {
		return "";
	}

	const [firstPoint, ...rest] = points;

	return `M ${firstPoint.x} ${firstPoint.y}${rest
		.map((point) => ` L ${point.x} ${point.y}`)
		.join("")}`;
};

/*
routeOrthogonalWithGrid finds an axis-aligned path from output port to input
port using A* over the occupancy grid, honoring fixed exit/approach stubs.
*/
export const routeOrthogonalWithGrid = (
	from: Coordinate,
	to: Coordinate,
	grid: RoutingGrid,
): string | null => {
	const startStub: Coordinate = {
		x: from.x + PORT_EXIT_STUB,
		y: from.y,
	};
	const endStub: Coordinate = {
		x: to.x - PORT_EXIT_STUB,
		y: to.y,
	};
	const startCell = grid.worldToCell(startStub.x, startStub.y);
	const goalCell = grid.worldToCell(endStub.x, endStub.y);
	const cellPath = findGridPath(grid, startCell, goalCell);

	if (!cellPath) {
		return null;
	}

	const routePoints: Coordinate[] = [from, startStub];

	for (const cell of cellPath) {
		routePoints.push(grid.cellCenterWorld(cell.cellX, cell.cellY));
	}

	routePoints.push(endStub, to);

	return formatOrthogonalSvgPath(simplifyOrthogonalPoints(routePoints));
};
