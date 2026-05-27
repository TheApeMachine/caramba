import { curveBasis, line } from "d3-shape";
import {
	buildRoutingGridFromObstacles,
	routeOrthogonalWithGrid,
} from "#/components/flume/orthogonal-grid-router";
import type { Coordinate } from "#/components/flume/types";

export type EdgeRoutingMode = "smooth" | "straight" | "orthogonal";

/** Axis-aligned bounds in the same stage coordinate space as edge endpoints. */
export type ObstacleRect = {
	readonly left: number;
	readonly right: number;
	readonly top: number;
	readonly bottom: number;
};

const OBSTACLE_PADDING = 16;
const CORRIDOR_MARGIN = 36;
const CORRIDOR_SCAN_STEP = 20;
const CORRIDOR_SCAN_LIMIT = 200;
/** Exit stub length: distance the wire travels horizontally before turning. */
const PORT_EXIT_STUB = 40;

const padObstacle = (o: ObstacleRect, pad: number): ObstacleRect => ({
	left: o.left - pad,
	right: o.right + pad,
	top: o.top - pad,
	bottom: o.bottom + pad,
});

const segmentHitsHorizontal = (
	y: number,
	x1: number,
	x2: number,
	obstacles: ReadonlyArray<ObstacleRect>,
): boolean => {
	if (x1 === x2) return false;
	const [xa, xb] = x1 <= x2 ? [x1, x2] : [x2, x1];

	for (const raw of obstacles) {
		const o = padObstacle(raw, OBSTACLE_PADDING);
		if (y <= o.top || y >= o.bottom) continue;
		if (xb <= o.left || xa >= o.right) continue;
		return true;
	}

	return false;
};

const segmentHitsVertical = (
	x: number,
	y1: number,
	y2: number,
	obstacles: ReadonlyArray<ObstacleRect>,
): boolean => {
	if (y1 === y2) return false;
	const [ya, yb] = y1 <= y2 ? [y1, y2] : [y2, y1];

	for (const raw of obstacles) {
		const o = padObstacle(raw, OBSTACLE_PADDING);
		if (x <= o.left || x >= o.right) continue;
		if (yb <= o.top || ya >= o.bottom) continue;
		return true;
	}

	return false;
};

/*
Orthogonal path from output port to input port.

Port conventions (fixed by UI layout):
  - Output ports are on the RIGHT face of a node → wire exits rightward
  - Input ports are on the LEFT face of a node  → wire enters leftward

Case A — forward (px < qx): vertical bus vx sits between the two stubs.
Case B — backward (px >= qx): wire loops above or below both nodes.
*/
export const calculateOrthogonalEdgePath = (
	from: Coordinate,
	to: Coordinate,
	obstaclesVertical: ReadonlyArray<ObstacleRect>,
	obstaclesHorizontal: ReadonlyArray<ObstacleRect>,
): string => {
	const grid = buildRoutingGridFromObstacles(obstaclesHorizontal);
	const gridPath = routeOrthogonalWithGrid(from, to, grid);

	if (gridPath) {
		return gridPath;
	}

	return calculateOrthogonalEdgePathCorridor(
		from,
		to,
		obstaclesVertical,
		obstaclesHorizontal,
	);
};

const calculateOrthogonalEdgePathCorridor = (
	from: Coordinate,
	to: Coordinate,
	obstaclesVertical: ReadonlyArray<ObstacleRect>,
	obstaclesHorizontal: ReadonlyArray<ObstacleRect>,
): string => {
	const px = from.x + PORT_EXIT_STUB;
	const py = from.y;
	const qx = to.x - PORT_EXIT_STUB;
	const qy = to.y;

	if (px < qx) {
		const seg = (vx: number) =>
			`M ${from.x} ${from.y} L ${px} ${py} L ${vx} ${py} L ${vx} ${qy} L ${qx} ${qy} L ${to.x} ${to.y}`;

		const isClear = (vx: number) =>
			!segmentHitsHorizontal(py, px, vx, obstaclesHorizontal) &&
			!segmentHitsHorizontal(qy, vx, qx, obstaclesHorizontal) &&
			!segmentHitsVertical(vx, py, qy, obstaclesVertical);

		const mid = Math.round((px + qx) / 2);

		for (let i = 0; i <= CORRIDOR_SCAN_LIMIT; i++) {
			if (isClear(mid + i * CORRIDOR_SCAN_STEP))
				return seg(mid + i * CORRIDOR_SCAN_STEP);
			if (i > 0 && isClear(mid - i * CORRIDOR_SCAN_STEP))
				return seg(mid - i * CORRIDOR_SCAN_STEP);
		}

		return seg(mid);
	}

	const eastBus = Math.max(from.x, to.x) + CORRIDOR_MARGIN;
	const westBus = Math.min(from.x, to.x) - CORRIDOR_MARGIN;

	const seg7 = (vy: number, vxEast: number, vxWest: number) =>
		`M ${from.x} ${from.y} L ${px} ${py} L ${vxEast} ${py} L ${vxEast} ${vy} L ${vxWest} ${vy} L ${vxWest} ${qy} L ${qx} ${qy} L ${to.x} ${to.y}`;

	const allNodes = [...obstaclesVertical, ...obstaclesHorizontal];
	const nodeExtents = allNodes.flatMap((o) => [
		o.top - OBSTACLE_PADDING,
		o.bottom + OBSTACLE_PADDING,
	]);
	const yMin = Math.min(py, qy, ...nodeExtents) - CORRIDOR_MARGIN;
	const yMax = Math.max(py, qy, ...nodeExtents) + CORRIDOR_MARGIN;

	const candidates: number[] = [yMin, yMax];

	for (const y of nodeExtents) {
		candidates.push(y - CORRIDOR_MARGIN, y + CORRIDOR_MARGIN);
	}

	candidates.sort((a, b) => a - b);

	const isBypassClear = (vy: number, vxEast: number, vxWest: number) =>
		!segmentHitsHorizontal(py, px, vxEast, obstaclesHorizontal) &&
		!segmentHitsHorizontal(qy, vxWest, qx, obstaclesHorizontal) &&
		!segmentHitsVertical(vxEast, py, vy, obstaclesVertical) &&
		!segmentHitsVertical(vxWest, vy, qy, obstaclesVertical) &&
		!segmentHitsHorizontal(vy, vxWest, vxEast, obstaclesVertical);

	for (let busStep = 0; busStep < CORRIDOR_SCAN_LIMIT; busStep++) {
		const vxEast = eastBus + busStep * CORRIDOR_SCAN_STEP;
		const vxWest = westBus - busStep * CORRIDOR_SCAN_STEP;

		for (const vy of candidates) {
			if (isBypassClear(vy, vxEast, vxWest)) {
				return seg7(vy, vxEast, vxWest);
			}
		}
	}

	const vy = yMin - CORRIDOR_SCAN_STEP * 4;
	return seg7(vy, eastBus, westBus);
};

const calculateSmoothCurve = (from: Coordinate, to: Coordinate) => {
	const length = to.x - from.x;
	const thirdLength = length / 3;

	let curveCoords: [number, number][] = [];

	if (to.x > from.x - 6) {
		curveCoords = [
			[from.x, from.y],
			[from.x + thirdLength, from.y],
			[from.x + thirdLength * 2, to.y],
			[to.x, to.y],
		];
	} else {
		const outD = 50;
		const height = Math.abs(to.y - from.y);
		const heightThird = height / 3;

		if (to.y > from.y) {
			curveCoords = [
				[from.x, from.y],
				[from.x + outD, from.y],
				[from.x + outD, from.y + heightThird],
				[to.x - outD, to.y - heightThird],
				[to.x - outD, to.y],
				[to.x, to.y],
			];
		} else {
			curveCoords = [
				[from.x, from.y],
				[from.x + outD, from.y],
				[from.x + outD, from.y - heightThird],
				[to.x - outD, to.y + heightThird],
				[to.x - outD, to.y],
				[to.x, to.y],
			];
		}
	}

	const curve = line().curve(curveBasis)(curveCoords);
	return curve ?? "";
};

export const calculateEdgePath = (
	mode: EdgeRoutingMode,
	from: Coordinate,
	to: Coordinate,
	obstaclesVertical?: ReadonlyArray<ObstacleRect>,
	obstaclesHorizontal?: ReadonlyArray<ObstacleRect>,
): string => {
	switch (mode) {
		case "straight":
			return `M ${from.x} ${from.y} L ${to.x} ${to.y}`;
		case "orthogonal": {
			const v = obstaclesVertical ?? [];
			const h = obstaclesHorizontal ?? v;
			return calculateOrthogonalEdgePath(from, to, v, h);
		}
		default:
			return calculateSmoothCurve(from, to);
	}
};
