import { curveBasis, line } from "d3-shape";
import type { RefObject } from "react";
import type FlumeCache from "#/components/flume/Cache";
import {
	CANVAS_ID,
	CONNECTIONS_ID,
	STAGE_ID,
} from "#/components/flume/constants";
import {
	buildRoutingGridFromObstacles,
	routeOrthogonalWithGrid,
} from "#/components/flume/orthogonal-grid-router";
import {
	buildObstacleMapFromSpatialIndex,
	resolveConnectionsFromSpatialIndex,
	type SpatialIndexSnapshot,
} from "#/components/flume/spatial-index";
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

const portHandleSelector = (
	nodeId: string,
	portName: string,
	transputType: TransputType,
) =>
	`[data-flume-component="port-handle"][data-node-id="${nodeId}"][data-port-name="${portName}"][data-port-transput-type="${transputType}"]`;

export const findPortHandle = (
	root: ParentNode,
	nodeId: string,
	portName: string,
	transputType: TransputType = "input",
) => root.querySelector(portHandleSelector(nodeId, portName, transputType));

const getPort = (
	nodeId: string,
	portName: string,
	transputType: TransputType = "input",
	editorId?: string,
) => {
	if (editorId) {
		return getPortInEditor(editorId, nodeId, portName, transputType);
	}
	return findPortHandle(document, nodeId, portName, transputType);
};

export const getPortRect = (
	nodeId: string,
	portName: string,
	transputType?: TransputType,
	cache?: RefObject<FlumeCache>,
	editorId?: string,
) => {
	const calculatedTransputType = transputType ?? "input";

	if (cache?.current) {
		const portCacheName = nodeId + portName + calculatedTransputType;
		const cachedPort = cache.current.ports[portCacheName];
		if (cachedPort?.isConnected) {
			return cachedPort.getBoundingClientRect();
		}
		const port = getPort(nodeId, portName, calculatedTransputType, editorId);
		if (port) {
			cache.current.ports[portCacheName] = port;
		}
		return port?.getBoundingClientRect() ?? null;
	}

	const port = getPort(nodeId, portName, calculatedTransputType, editorId);
	return port?.getBoundingClientRect() ?? null;
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
									name: connectionId(
										output.nodeId,
										output.portName,
										node.id,
										inputName,
									),
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
}

function calculateOrthogonalEdgePathCorridor(
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
			if (isClear(mid + i * CORRIDOR_SCAN_STEP))
				return seg(mid + i * CORRIDOR_SCAN_STEP);
			if (i > 0 && isClear(mid - i * CORRIDOR_SCAN_STEP))
				return seg(mid - i * CORRIDOR_SCAN_STEP);
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
	const nodeExtents = allNodes.flatMap((o) => [
		o.top - OBSTACLE_PADDING,
		o.bottom + OBSTACLE_PADDING,
	]);
	const yMin = Math.min(py, qy, ...nodeExtents) - CORRIDOR_MARGIN;
	const yMax = Math.max(py, qy, ...nodeExtents) + CORRIDOR_MARGIN;

	// Candidate horizontal corridors: above all nodes, below all nodes, and between node rows.
	const candidates: number[] = [yMin, yMax];
	for (const y of nodeExtents)
		candidates.push(y - CORRIDOR_MARGIN, y + CORRIDOR_MARGIN);
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
	editorId?: string,
): Map<string, ObstacleRect> {
	const stageHalfWidth = stage.width / 2;
	const stageHalfHeight = stage.height / 2;
	const byScale = (value: number) => (1 / scale) * value;
	const canvas = editorId ? getCanvasRef(editorId) : null;
	const out = new Map<string, ObstacleRect>();
	for (const id of Object.keys(nodes)) {
		const element = canvas
			? canvas.querySelector(
					`[data-flume-component="node"][data-node-id="${id}"]`,
				)
			: document.querySelector(
					`[data-flume-component="node"][data-node-id="${id}"]`,
				);
		if (!(element instanceof Element)) continue;
		const rect = element.getBoundingClientRect();
		out.set(id, {
			left: byScale(rect.left - stage.x - stageHalfWidth),
			right: byScale(rect.right - stage.x - stageHalfWidth),
			top: byScale(rect.top - stage.y - stageHalfHeight),
			bottom: byScale(rect.bottom - stage.y - stageHalfHeight),
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
		calculateEdgePath(
			routingMode,
			from,
			to,
			obstaclesVertical,
			obstaclesHorizontal,
		),
	);
};

/*
ConnectionShellDescriptor matches the worker's roster entry: just the
endpoint identifiers needed to find or create the SVG path element.
The actual d attribute is set separately by applyPaths from the worker
output.
*/
export type ConnectionShellDescriptor = {
	id: string;
	outputNodeId: string;
	outputPortName: string;
	inputNodeId: string;
	inputPortName: string;
};

/*
syncConnectionElements ensures the SVG path elements in the stage
exactly match the roster from the worker. Adds missing elements with
an empty d attribute (worker fills it in), removes stale ones. No
geometry math runs here — that's the worker's job.
*/
export const syncConnectionElements = (
	roster: ReadonlyArray<ConnectionShellDescriptor>,
	editorId: string,
	routingMode: EdgeRoutingMode = "smooth",
): void => {
	const stage = getStageRef(editorId);

	if (!stage) {
		return;
	}

	const rosterById = new Map<string, ConnectionShellDescriptor>();

	for (const entry of roster) {
		rosterById.set(entry.id, entry);
	}

	for (const pathElement of stage.querySelectorAll<SVGPathElement>(
		"[data-connection-id]",
	)) {
		const id = pathElement.getAttribute("data-connection-id");

		if (!id || !rosterById.has(id)) {
			pathElement.parentElement?.remove();
		}
	}

	for (const entry of roster) {
		const existing = stage.querySelector<SVGPathElement>(
			`[data-connection-id="${entry.id}"]`,
		);

		if (existing) {
			if (routingMode === "orthogonal") {
				existing.setAttribute("stroke-linejoin", "miter");
			} else {
				existing.removeAttribute("stroke-linejoin");
			}
			continue;
		}

		const svg = document.createElementNS(
			"http://www.w3.org/2000/svg",
			"svg",
		);
		svg.setAttribute(
			"style",
			"position:absolute;left:0;top:0;pointer-events:none;z-index:0;overflow:visible;",
		);

		const path = document.createElementNS(
			"http://www.w3.org/2000/svg",
			"path",
		);
		path.setAttribute("d", "");
		path.setAttribute("stroke", "rgb(185, 186, 189)");
		path.setAttribute("stroke-width", "3");
		path.setAttribute("stroke-linecap", "round");

		if (routingMode === "orthogonal") {
			path.setAttribute("stroke-linejoin", "miter");
		}

		path.setAttribute("fill", "none");
		path.setAttribute("data-connection-id", entry.id);
		path.setAttribute("data-output-node-id", entry.outputNodeId);
		path.setAttribute("data-output-port-name", entry.outputPortName);
		path.setAttribute("data-input-node-id", entry.inputNodeId);
		path.setAttribute("data-input-port-name", entry.inputPortName);

		svg.appendChild(path);
		stage.appendChild(svg);
	}
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

export const getCanvasRef = (editorId: string) =>
	document.getElementById(`${CANVAS_ID}${editorId}`);

/*
getStageBounds returns the visible stage rect used to convert screen coordinates
into the editor's center-origin canvas space.
*/
export const getStageBounds = (editorId: string): DOMRect | null => {
	const stage = document.getElementById(`${STAGE_ID}${editorId}`);
	return stage?.getBoundingClientRect() ?? null;
};

export const screenPointToCanvas = (
	screenX: number,
	screenY: number,
	stageRect: DOMRect,
	scale: number,
): Coordinate => {
	const byScale = (value: number) => (1 / scale) * value;
	const stageHalfWidth = stageRect.width / 2;
	const stageHalfHeight = stageRect.height / 2;

	return {
		x: byScale(screenX - stageRect.x - stageHalfWidth),
		y: byScale(screenY - stageRect.y - stageHalfHeight),
	};
};

export const screenRectToCanvas = (
	rect: DOMRect,
	stageRect: DOMRect,
	scale: number,
): { x: number; y: number; width: number; height: number } => {
	const topLeft = screenPointToCanvas(rect.left, rect.top, stageRect, scale);
	const bottomRight = screenPointToCanvas(
		rect.right,
		rect.bottom,
		stageRect,
		scale,
	);

	return {
		x: topLeft.x,
		y: topLeft.y,
		width: bottomRight.x - topLeft.x,
		height: bottomRight.y - topLeft.y,
	};
};

/** Reads the live CSS scale from the canvas; falls back when React state is stale during wheel zoom. */
export const readLiveStageScale = (
	editorId: string,
	fallbackScale: number,
): number => {
	const canvas = getCanvasRef(editorId);
	if (!canvas) return fallbackScale;

	const match = canvas.style.transform.match(/scale\(([^)]+)\)/);
	if (!match) return fallbackScale;

	const parsedScale = Number.parseFloat(match[1]);
	if (Number.isNaN(parsedScale) || parsedScale <= 0) return fallbackScale;

	return parsedScale;
};

export const getStageRef = (editorId: string) =>
	document.getElementById(
		`${CONNECTIONS_ID}${editorId}`,
	) as HTMLDivElement | null;

export const getPortInEditor = (
	editorId: string,
	nodeId: string,
	portName: string,
	transputType: TransputType = "input",
) => {
	const canvas = getCanvasRef(editorId);
	if (!canvas) return null;
	return findPortHandle(canvas, nodeId, portName, transputType);
};

export const createConnections = (
	nodes: { [nodeId: string]: FlumeNode },
	_editorState: StageState,
	editorId: string,
	routingMode: EdgeRoutingMode = "smooth",
	spatialIndex?: SpatialIndexSnapshot,
	positionOverrides?: Record<string, Coordinate>,
) => {
	const stageRef = getStageRef(editorId);
	if (!stageRef || !spatialIndex) {
		return;
	}

	const allObstaclesById: Map<string, ObstacleRect> | undefined =
		routingMode === "orthogonal"
			? buildObstacleMapFromSpatialIndex(nodes, spatialIndex, positionOverrides)
			: undefined;
	const allObstacles = allObstaclesById
		? Array.from(allObstaclesById.values())
		: undefined;

	const resolved = resolveConnectionsFromSpatialIndex(
		nodes,
		spatialIndex,
		positionOverrides,
	);

	const resolvedIds = new Set(resolved.map((connection) => connection.id));

	for (const pathElement of stageRef.querySelectorAll<SVGPathElement>(
		"[data-connection-id]",
	)) {
		const staleConnectionId = pathElement.getAttribute("data-connection-id");

		if (staleConnectionId && !resolvedIds.has(staleConnectionId)) {
			pathElement.parentElement?.remove();
		}
	}

	for (const connection of resolved) {
		const obstaclesVertical = allObstacles;
		const obstaclesHorizontal = allObstaclesById
			? Array.from(allObstaclesById.entries())
					.filter(
						([nodeId]) =>
							nodeId !== connection.outputNodeId &&
							nodeId !== connection.inputNodeId,
					)
					.map(([, rect]) => rect)
			: undefined;

		const existingLine = stageRef.querySelector<SVGPathElement>(
			`[data-connection-id="${connection.id}"]`,
		);

		if (existingLine) {
			updateConnection({
				line: existingLine,
				from: connection.from,
				to: connection.to,
				routingMode,
				obstaclesVertical,
				obstaclesHorizontal,
			});
			continue;
		}

		createSVG({
			id: connection.id,
			outputNodeId: connection.outputNodeId,
			outputPortName: connection.outputPortName,
			inputNodeId: connection.inputNodeId,
			inputPortName: connection.inputPortName,
			from: connection.from,
			to: connection.to,
			stage: stageRef,
			routingMode,
			obstaclesVertical,
			obstaclesHorizontal,
		});
	}
};
