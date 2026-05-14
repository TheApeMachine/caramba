import { curveBasis, line } from "d3-shape";
import type { RefObject } from "react";
import type FlumeCache from "#/components/flume/Cache";
import { CONNECTIONS_ID } from "#/components/flume/constants";
import type {
	Coordinate,
	FlumeNode,
	StageState,
	TransputType,
} from "#/components/flume/types";

/** Encodes a connection-id segment so the `|` delimiter is unambiguous. */
const encSeg = (s: string) => s.replace(/[|\\]/g, (c) => `\\${c}`);

/** Builds a stable, unambiguous connection id from its four components. */
export const connectionId = (
	outputNodeId: string,
	outputPortName: string,
	inputNodeId: string,
	inputPortName: string,
) =>
	`${encSeg(outputNodeId)}|${encSeg(outputPortName)}|${encSeg(inputNodeId)}|${encSeg(inputPortName)}`;

const getPort = (
	nodeId: string,
	portName: string,
	transputType: TransputType = "input",
) =>
	document.querySelector(
		`[data-node-id="${nodeId}"] [data-port-name="${portName}"][data-port-transput-type="${transputType}"]`,
	);

export const getPortRect = (
	nodeId: string,
	portName: string,
	transputType?: TransputType,
	cache?: RefObject<FlumeCache>,
) => {
	const calculatedTransputType = transputType ?? "input";

	if (cache?.current) {
		const portCacheName = nodeId + portName + calculatedTransputType;
		const cachedPort = cache.current.ports[portCacheName];
		if (cachedPort) {
			return cachedPort.getBoundingClientRect();
		} else {
			const port = getPort(nodeId, portName, calculatedTransputType);
			if (port) {
				cache.current.ports[portCacheName] = port;
			}
			return port?.getBoundingClientRect() ?? null;
		}
	} else {
		const port = getPort(nodeId, portName, calculatedTransputType);
		return port?.getBoundingClientRect() ?? null;
	}
};

export const getPortRectsByNodes = (
	nodes: { [nodeId: string]: FlumeNode },
	forEachConnection: (connection: {
		to: DOMRect | null;
		from: DOMRect | null;
		name: string;
	}) => void,
) =>
	Object.values(nodes).reduce<{ [key: string]: DOMRect | null }>(
		(obj, node) => {
			if (node.connections?.inputs) {
				Object.entries(node.connections.inputs).forEach(
					([inputName, outputs]) => {
						outputs.forEach((output) => {
							const toRect = getPortRect(node.id, inputName);
							const fromRect = getPortRect(
								output.nodeId,
								output.portName,
								"output",
							);
							if (forEachConnection) {
								forEachConnection({
									to: toRect,
									from: fromRect,
									name: connectionId(output.nodeId, output.portName, node.id, inputName),
								});
							}
							obj[node.id + inputName] = toRect;
							obj[output.nodeId + output.portName] = fromRect;
						});
					},
				);
			}
			return obj;
		},
		{},
	);

export type EdgeRoutingMode = "smooth" | "straight" | "orthogonal";

/** Axis-aligned bounds in the same stage coordinate space as {@link calculateEdgePath} endpoints. */
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

function padObstacle(o: ObstacleRect, pad: number): ObstacleRect {
	return {
		left: o.left - pad,
		right: o.right + pad,
		top: o.top - pad,
		bottom: o.bottom + pad,
	};
}

function segmentHitsHorizontal(
	y: number,
	x1: number,
	x2: number,
	obstacles: ReadonlyArray<ObstacleRect>,
): boolean {
	if (x1 === x2) return false;
	const [xa, xb] = x1 <= x2 ? [x1, x2] : [x2, x1];
	for (const raw of obstacles) {
		const o = padObstacle(raw, OBSTACLE_PADDING);
		if (y <= o.top || y >= o.bottom) continue;
		if (xb <= o.left || xa >= o.right) continue;
		return true;
	}
	return false;
}

function segmentHitsVertical(
	x: number,
	y1: number,
	y2: number,
	obstacles: ReadonlyArray<ObstacleRect>,
): boolean {
	if (y1 === y2) return false;
	const [ya, yb] = y1 <= y2 ? [y1, y2] : [y2, y1];
	for (const raw of obstacles) {
		const o = padObstacle(raw, OBSTACLE_PADDING);
		if (x <= o.left || x >= o.right) continue;
		if (yb <= o.top || ya >= o.bottom) continue;
		return true;
	}
	return false;
}


/*
Orthogonal path from {@link from} (output port) to {@link to} (input port).

Port conventions (fixed by UI layout):
  - Output ports are on the RIGHT face of a node → wire exits rightward
  - Input ports are on the LEFT face of a node  → wire enters leftward (approaches from west)

The canonical 5-segment path has the form:
  from → [px, py] → [vx, py] → [vx, qy] → [qx, qy] → to

where px = from.x + STUB (output exit, going right)
  and qx = to.x   - STUB (input  approach, arriving from the left)

Case A — forward (px < qx): the vertical bus vx sits between the two stubs.
  Scan from the midpoint outward to find a vx free of obstacles.

Case B — backward (px >= qx): the output is to the right of (or level with) the input.
  The wire must loop around. We do this by routing ABOVE or BELOW both nodes:
  find a horizontal corridor vy that is clear, then use a 7-segment path:
    from → [px,py] → [east,py] → [east,vy] → [west,vy] → [west,qy] → [qx,qy] → to
  where east/west are vertical buses placed outside both nodes.
*/
export function calculateOrthogonalEdgePath(
	from: Coordinate,
	to: Coordinate,
	obstaclesVertical: ReadonlyArray<ObstacleRect>,
	obstaclesHorizontal: ReadonlyArray<ObstacleRect>,
): string {
	const px = from.x + PORT_EXIT_STUB;
	const py = from.y;
	const qx = to.x - PORT_EXIT_STUB;
	const qy = to.y;

	// ── Case A: forward ─────────────────────────────────────────────────────────
	if (px < qx) {
		const seg = (vx: number) =>
			`M ${from.x} ${from.y} L ${px} ${py} L ${vx} ${py} L ${vx} ${qy} L ${qx} ${qy} L ${to.x} ${to.y}`;

		const isClear = (vx: number) =>
			!segmentHitsHorizontal(py, px, vx, obstaclesHorizontal) &&
			!segmentHitsHorizontal(qy, vx, qx, obstaclesHorizontal) &&
			!segmentHitsVertical(vx, py, qy, obstaclesVertical);

		const mid = Math.round((px + qx) / 2);
		for (let i = 0; i <= CORRIDOR_SCAN_LIMIT; i++) {
			if (isClear(mid + i * CORRIDOR_SCAN_STEP)) return seg(mid + i * CORRIDOR_SCAN_STEP);
			if (i > 0 && isClear(mid - i * CORRIDOR_SCAN_STEP)) return seg(mid - i * CORRIDOR_SCAN_STEP);
		}
		return seg(mid);
	}

	// ── Case B: backward — route above or below via a horizontal bypass ──────────
	// Place two vertical buses (east of output, west of input) joined by a
	// horizontal bypass corridor vy that sits above or below both nodes.
	const eastBus = Math.max(from.x, to.x) + CORRIDOR_MARGIN;
	const westBus = Math.min(from.x, to.x) - CORRIDOR_MARGIN;

	const seg7 = (vy: number, vxEast: number, vxWest: number) =>
		`M ${from.x} ${from.y} L ${px} ${py} L ${vxEast} ${py} L ${vxEast} ${vy} L ${vxWest} ${vy} L ${vxWest} ${qy} L ${qx} ${qy} L ${to.x} ${to.y}`;

	// Gather all node tops/bottoms to find bypass corridors above and below.
	const allNodes = [...obstaclesVertical, ...obstaclesHorizontal];
	const nodeExtents = allNodes.flatMap((o) => [o.top - OBSTACLE_PADDING, o.bottom + OBSTACLE_PADDING]);
	const yMin = Math.min(py, qy, ...nodeExtents) - CORRIDOR_MARGIN;
	const yMax = Math.max(py, qy, ...nodeExtents) + CORRIDOR_MARGIN;

	// Candidate horizontal corridors: above all nodes, below all nodes, and between node rows.
	const candidates: number[] = [yMin, yMax];
	for (const y of nodeExtents) candidates.push(y - CORRIDOR_MARGIN, y + CORRIDOR_MARGIN);
	candidates.sort((a, b) => a - b);

	const isBypassClear = (vy: number, vxEast: number, vxWest: number) =>
		// horizontal legs at py and qy (output/input approach — exclude endpoint nodes)
		!segmentHitsHorizontal(py, px, vxEast, obstaclesHorizontal) &&
		!segmentHitsHorizontal(qy, vxWest, qx, obstaclesHorizontal) &&
		// vertical buses (all nodes)
		!segmentHitsVertical(vxEast, py, vy, obstaclesVertical) &&
		!segmentHitsVertical(vxWest, vy, qy, obstaclesVertical) &&
		// horizontal bypass (all nodes)
		!segmentHitsHorizontal(vy, vxWest, vxEast, obstaclesVertical);

	// Scan east/west bus positions outward while testing each bypass corridor.
	for (let busStep = 0; busStep < CORRIDOR_SCAN_LIMIT; busStep++) {
		const vxEast = eastBus + busStep * CORRIDOR_SCAN_STEP;
		const vxWest = westBus - busStep * CORRIDOR_SCAN_STEP;
		for (const vy of candidates) {
			if (isBypassClear(vy, vxEast, vxWest)) return seg7(vy, vxEast, vxWest);
		}
	}

	// Hard fallback.
	const vy = yMin - CORRIDOR_SCAN_STEP * 4;
	return seg7(vy, eastBus, westBus);
}

export function buildObstacleMap(
	nodes: Record<string, FlumeNode>,
	stage: DOMRect,
	scale: number,
): Map<string, ObstacleRect> {
	const hw = stage.width / 2;
	const hh = stage.height / 2;
	const byScale = (value: number) => (1 / scale) * value;
	const out = new Map<string, ObstacleRect>();
	for (const id of Object.keys(nodes)) {
		const el = document.querySelector(
			`[data-flume-component="node"][data-node-id="${id}"]`,
		);
		if (!(el instanceof Element)) continue;
		const rect = el.getBoundingClientRect();
		out.set(id, {
			left:   byScale(rect.left   - stage.x - hw),
			right:  byScale(rect.right  - stage.x - hw),
			top:    byScale(rect.top    - stage.y - hh),
			bottom: byScale(rect.bottom - stage.y - hh),
		});
	}
	return out;
}

export function obstacleRectsFromNodes(
	nodes: Record<string, FlumeNode>,
	stage: DOMRect,
	scale: number,
	excludeIds?: ReadonlySet<string>,
): ObstacleRect[] {
	const map = buildObstacleMap(nodes, stage, scale);
	if (!excludeIds) return Array.from(map.values());
	return Array.from(map.entries())
		.filter(([id]) => !excludeIds.has(id))
		.map(([, rect]) => rect);
}

/*
Builds obstacle boxes from every rendered graph node in the DOM except ids in
{@link excludeIds}. Uses the same stage coordinates as {@link createConnections}.
*/
export function collectDomObstacleRects(
	stage: DOMRect,
	scale: number,
	excludeIds: ReadonlySet<string>,
): ObstacleRect[] {
	const hw = stage.width / 2;
	const hh = stage.height / 2;
	const byScale = (value: number) => (1 / scale) * value;
	const out: ObstacleRect[] = [];
	for (const el of document.querySelectorAll(
		'[data-flume-component="node"][data-node-id]',
	)) {
		const nid = el.getAttribute("data-node-id");
		if (!nid || excludeIds.has(nid)) continue;
		const rect = el.getBoundingClientRect();
		out.push({
			left: byScale(rect.left - stage.x - hw),
			right: byScale(rect.right - stage.x - hw),
			top: byScale(rect.top - stage.y - hh),
			bottom: byScale(rect.bottom - stage.y - hh),
		});
	}
	return out;
}

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
		case "smooth":
		default:
			return calculateSmoothCurve(from, to);
	}
};

/** Same as {@link calculateEdgePath} with `"smooth"` routing (legacy Flume behavior). */
export const calculateCurve = (from: Coordinate, to: Coordinate) =>
	calculateEdgePath("smooth", from, to, undefined, undefined);

export const deleteConnection = ({ id }: { id: string }) => {
	const line = document.querySelector(`[data-connection-id="${id}"]`);
	line?.parentElement?.remove();
};

export const deleteConnectionsByNodeId = (nodeId: string) => {
	const lines = Array.from(
		document.querySelectorAll(
			`[data-output-node-id="${nodeId}"], [data-input-node-id="${nodeId}"]`,
		),
	);
	for (const line of lines) {
		line?.parentElement?.remove();
	}
};

export const updateConnection = ({
	line,
	from,
	to,
	routingMode = "smooth",
	obstaclesVertical,
	obstaclesHorizontal,
}: {
	line: SVGPathElement;
	from: Coordinate;
	to: Coordinate;
	routingMode?: EdgeRoutingMode;
	obstaclesVertical?: ReadonlyArray<ObstacleRect>;
	obstaclesHorizontal?: ReadonlyArray<ObstacleRect>;
}) => {
	line.setAttribute(
		"d",
		calculateEdgePath(routingMode, from, to, obstaclesVertical, obstaclesHorizontal),
	);
};

export const createSVG = ({
	from,
	to,
	stage,
	id,
	outputNodeId,
	outputPortName,
	inputNodeId,
	inputPortName,
	routingMode = "smooth",
	obstaclesVertical,
	obstaclesHorizontal,
}: {
	from: Coordinate;
	to: Coordinate;
	stage: HTMLDivElement;
	id: string;
	outputNodeId: string;
	outputPortName: string;
	inputNodeId: string;
	inputPortName: string;
	routingMode?: EdgeRoutingMode;
	obstaclesVertical?: ReadonlyArray<ObstacleRect>;
	obstaclesHorizontal?: ReadonlyArray<ObstacleRect>;
}) => {
	const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
	svg.setAttribute(
		"style",
		"position:absolute;left:0;top:0;pointer-events:none;z-index:0;overflow:visible;",
	);
	const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
	const curve = calculateEdgePath(
		routingMode,
		from,
		to,
		obstaclesVertical,
		obstaclesHorizontal,
	);
	path.setAttribute("d", curve);
	path.setAttribute("stroke", "rgb(185, 186, 189)");
	path.setAttribute("stroke-width", "3");
	path.setAttribute("stroke-linecap", "round");
	if (routingMode === "orthogonal") {
		path.setAttribute("stroke-linejoin", "miter");
	}
	path.setAttribute("fill", "none");
	path.setAttribute("data-connection-id", id);
	path.setAttribute("data-output-node-id", outputNodeId);
	path.setAttribute("data-output-port-name", outputPortName);
	path.setAttribute("data-input-node-id", inputNodeId);
	path.setAttribute("data-input-port-name", inputPortName);
	svg.appendChild(path);
	stage.appendChild(svg);
	return svg;
};

export const getStageRef = (editorId: string) =>
	document.getElementById(
		`${CONNECTIONS_ID}${editorId}`,
	) as HTMLDivElement | null;

export const createConnections = (
	nodes: { [nodeId: string]: FlumeNode },
	{ scale }: StageState,
	editorId: string,
	routingMode: EdgeRoutingMode = "smooth",
) => {
	const stageRef = getStageRef(editorId);
	if (stageRef) {
		const stage = stageRef.getBoundingClientRect();
		const stageHalfWidth = stage.width / 2;
		const stageHalfHeight = stage.height / 2;

		const byScale = (value: number) => (1 / scale) * value;

		// Build obstacle rects once per pass, not once per edge.
		// allObstaclesById lets us cheaply exclude endpoint nodes per edge.
		const allObstaclesById: Map<string, ObstacleRect> | undefined =
			routingMode === "orthogonal"
				? buildObstacleMap(nodes, stage, scale)
				: undefined;
		const allObstacles = allObstaclesById
			? Array.from(allObstaclesById.values())
			: undefined;

		Object.values(nodes).forEach((node) => {
			if (node.connections?.inputs) {
				Object.entries(node.connections.inputs).forEach(
					([inputName, outputs]) => {
						outputs.forEach((output) => {
							const fromPort = getPortRect(
								output.nodeId,
								output.portName,
								"output",
							);
							const toPort = getPortRect(node.id, inputName, "input");
							if (fromPort && toPort) {
								const fromHalfW = fromPort.width / 2;
								const fromHalfH = fromPort.height / 2;
								const toHalfW = toPort.width / 2;
								const toHalfH = toPort.height / 2;
								const id = connectionId(output.nodeId, output.portName, node.id, inputName);
								const fromCoord = {
									x: byScale(
										fromPort.x - stage.x + fromHalfW - stageHalfWidth,
									),
									y: byScale(
										fromPort.y - stage.y + fromHalfH - stageHalfHeight,
									),
								};
								const toCoord = {
									x: byScale(
										toPort.x - stage.x + toHalfW - stageHalfWidth,
									),
									y: byScale(
										toPort.y - stage.y + toHalfH - stageHalfHeight,
									),
								};

								// Per-edge obstacle set: exclude the two endpoint nodes for horizontal legs.
								const obstaclesVertical = allObstacles;
								const obstaclesHorizontal = allObstaclesById
									? Array.from(allObstaclesById.entries())
										.filter(([id]) => id !== output.nodeId && id !== node.id)
										.map(([, rect]) => rect)
									: undefined;

								const existingLine: SVGPathElement | null =
									document.querySelector(`[data-connection-id="${id}"]`);

								if (existingLine) {
									updateConnection({
										line: existingLine,
										from: fromCoord,
										to: toCoord,
										routingMode,
										obstaclesVertical,
										obstaclesHorizontal,
									});
								} else {
									createSVG({
										id,
										outputNodeId: output.nodeId,
										outputPortName: output.portName,
										inputNodeId: node.id,
										inputPortName: inputName,
										from: fromCoord,
										to: toCoord,
										stage: stageRef,
										routingMode,
										obstaclesVertical,
										obstaclesHorizontal,
									});
								}
							}
						});
					},
				);
			}
		});
	}
};
