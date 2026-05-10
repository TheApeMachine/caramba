import { curveBasis, line } from "d3-shape";
import type { RefObject } from "react";
import type FlumeCache from "#/components/flume/Cache";
import styles from "#/components/flume/Connection/Connection.module.css";
import { CONNECTIONS_ID } from "#/components/flume/constants";
import type {
	Coordinate,
	FlumeNode,
	StageState,
	TransputType,
} from "#/components/flume/types";

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
									name: output.nodeId + output.portName + node.id + inputName,
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
const CORRIDOR_SCAN_STEP = 40;
const CORRIDOR_SCAN_LIMIT = 96;
/** Pixels along +X from output and along −X toward input (Flume ports: outputs right, inputs left). */
const PORT_EXIT_STUB = 32;

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
		if (y < o.top || y > o.bottom) continue;
		if (xb < o.left || xa > o.right) continue;
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
		if (x < o.left || x > o.right) continue;
		if (yb < o.top || ya > o.bottom) continue;
		return true;
	}
	return false;
}

function stubbedEastBusClear(
	px: number,
	py: number,
	qx: number,
	qy: number,
	vx: number,
	obstaclesHorizontal: ReadonlyArray<ObstacleRect>,
	obstaclesVertical: ReadonlyArray<ObstacleRect>,
): boolean {
	return (
		!segmentHitsHorizontal(py, px, vx, obstaclesHorizontal) &&
		!segmentHitsHorizontal(qy, vx, qx, obstaclesHorizontal) &&
		!segmentHitsVertical(vx, py, qy, obstaclesVertical)
	);
}

/*
Orthogonal path from {@link from} (output port) to {@link to} (input port).

Flume wires outputs on the right and inputs on the left. We leave the producer with a short +X
stub, run an east-side vertical bus {@link vx}, then approach the input along the port row from
the west (`qx` → `to.x`), matching input ports sitting left of labels.

Port stubs are not obstacle-tested. Horizontal legs at {@link py} and {@link qy} use
{@link obstaclesHorizontal}, which should **omit the two endpoint nodes** so the trace leaving the
output stub is not treated as colliding with its own card. The vertical bus uses
{@link obstaclesVertical}, typically **every** node box, so we still dodge cards between rows.

If horizontals used full node bounds including endpoints, clearance almost never succeeds and the
fallback pushes {@link vx} thousands of units east (“infinite” rays).

Obstacle DOM lookup uses `[data-flume-component="node"]` so port handles sharing `data-node-id`
do not replace full node bounds.
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

	const baseVx = Math.max(px, qx, from.x, to.x) + CORRIDOR_MARGIN;
	let vx = baseVx;
	for (let i = 0; i < CORRIDOR_SCAN_LIMIT; i++) {
		if (
			stubbedEastBusClear(
				px,
				py,
				qx,
				qy,
				vx,
				obstaclesHorizontal,
				obstaclesVertical,
			)
		) {
			return `M ${from.x} ${from.y} L ${px} ${py} L ${vx} ${py} L ${vx} ${qy} L ${qx} ${qy} L ${to.x} ${to.y}`;
		}
		vx += CORRIDOR_SCAN_STEP;
	}

	const vxFallback = baseVx + Math.min(
		CORRIDOR_SCAN_STEP * 12,
		CORRIDOR_SCAN_STEP * CORRIDOR_SCAN_LIMIT,
	);
	return `M ${from.x} ${from.y} L ${px} ${py} L ${vxFallback} ${py} L ${vxFallback} ${qy} L ${qx} ${qy} L ${to.x} ${to.y}`;
}

export function obstacleRectsFromNodes(
	nodes: Record<string, FlumeNode>,
	stage: DOMRect,
	scale: number,
	excludeIds?: ReadonlySet<string>,
): ObstacleRect[] {
	const hw = stage.width / 2;
	const hh = stage.height / 2;
	const byScale = (value: number) => (1 / scale) * value;

	const out: ObstacleRect[] = [];
	for (const id of Object.keys(nodes)) {
		if (excludeIds?.has(id)) continue;
		const el = document.querySelector(
			`[data-flume-component="node"][data-node-id="${id}"]`,
		);
		if (!(el instanceof Element)) continue;
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
	svg.setAttribute("class", styles.svg);
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
								const id =
									output.nodeId + output.portName + node.id + inputName;
								const existingLine: SVGPathElement | null =
									document.querySelector(`[data-connection-id="${id}"]`);
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
								const obstaclesVertical =
									routingMode === "orthogonal"
										? obstacleRectsFromNodes(nodes, stage, scale)
										: undefined;
								const obstaclesHorizontal =
									routingMode === "orthogonal"
										? obstacleRectsFromNodes(
												nodes,
												stage,
												scale,
												new Set([output.nodeId, node.id]),
											)
										: undefined;

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
