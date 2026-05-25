import {
	calculateEdgePath,
	type EdgeRoutingMode,
	type ObstacleRect,
} from "#/components/flume/connectionCalculator";
import {
	buildRoutingGridFromObstacleMap,
	routeOrthogonalWithGrid,
} from "#/components/flume/orthogonal-grid-router";
import {
	buildObstacleMapFromSpatialIndex,
	type NodeLayoutEntry,
	type PortLayoutEntry,
	portLayoutKey,
	resolveConnectionsFromSpatialIndex,
	type SpatialIndexSnapshot,
} from "#/components/flume/spatial-index";
import type {
	Coordinate,
	NodeMap,
	TransputType,
} from "#/components/flume/types";
import type {
	ConnectionPathResult,
	DragUpdateResult,
	GraphSnapshot,
} from "#/workers/flume-graph.types";

const snapshotFromEngine = (
	nodeLayouts: Map<string, NodeLayoutEntry>,
	portLayouts: Map<string, PortLayoutEntry>,
): SpatialIndexSnapshot => ({
	nodeLayouts,
	portLayouts,
});

const computeOrthogonalPath = (
	from: Coordinate,
	to: Coordinate,
	allObstaclesById: Map<string, ObstacleRect>,
	outputNodeId: string,
	inputNodeId: string,
	allObstacles: ObstacleRect[],
): string => {
	const grid = buildRoutingGridFromObstacleMap(
		allObstaclesById,
		new Set([outputNodeId, inputNodeId]),
	);
	const gridPath = routeOrthogonalWithGrid(from, to, grid);

	if (gridPath) {
		return gridPath;
	}

	const obstaclesHorizontal = Array.from(allObstaclesById.entries())
		.filter(([nodeId]) => nodeId !== outputNodeId && nodeId !== inputNodeId)
		.map(([, rect]) => rect);

	return calculateEdgePath(
		"orthogonal",
		from,
		to,
		allObstacles,
		obstaclesHorizontal,
	);
};

/*
FlumeGraphEngine owns the serializable graph snapshot and recomputes edge
paths from node positions plus registered port offsets — no DOM reads.
*/
export class FlumeGraphEngine {
	private nodes: NodeMap = {};
	private routingMode: EdgeRoutingMode = "smooth";
	private nodeLayouts = new Map<string, NodeLayoutEntry>();
	private portLayouts = new Map<string, PortLayoutEntry>();
	private dragNodeId: string | null = null;
	private dragPosition: Coordinate | null = null;

	loadSnapshot(snapshot: GraphSnapshot): void {
		this.nodes = structuredClone(snapshot.nodes);
		this.routingMode = snapshot.routingMode;
		this.nodeLayouts.clear();
		this.portLayouts.clear();

		for (const layout of snapshot.nodeLayouts) {
			this.nodeLayouts.set(layout.nodeId, {
				width: layout.width,
				height: layout.height,
			});
		}

		for (const layout of snapshot.portLayouts) {
			this.portLayouts.set(
				portLayoutKey(layout.nodeId, layout.portName, layout.transputType),
				{
					offsetX: layout.offsetX,
					offsetY: layout.offsetY,
				},
			);
		}

		this.dragNodeId = null;
		this.dragPosition = null;
	}

	setRoutingMode(routingMode: EdgeRoutingMode): void {
		this.routingMode = routingMode;
	}

	setNodeLayout(nodeId: string, width: number, height: number): void {
		if (width <= 0 || height <= 0) {
			return;
		}

		this.nodeLayouts.set(nodeId, { width, height });
	}

	setPortLayout(
		nodeId: string,
		portName: string,
		transputType: TransputType,
		offsetX: number,
		offsetY: number,
	): void {
		this.portLayouts.set(portLayoutKey(nodeId, portName, transputType), {
			offsetX,
			offsetY,
		});
	}

	beginDrag(nodeId: string): void {
		const node = this.nodes[nodeId];

		if (!node) {
			return;
		}

		this.dragNodeId = nodeId;
		this.dragPosition = { x: node.x, y: node.y };
	}

	updateDrag(nodeId: string, x: number, y: number): DragUpdateResult {
		this.dragNodeId = nodeId;
		this.dragPosition = { x, y };

		return {
			paths: this.computePaths(),
			dragPosition: { x, y },
		};
	}

	endDrag(nodeId: string, x: number, y: number): ConnectionPathResult[] {
		const node = this.nodes[nodeId];

		if (node) {
			node.x = x;
			node.y = y;
		}

		this.dragNodeId = null;
		this.dragPosition = null;

		return this.computePaths();
	}

	computePaths(): ConnectionPathResult[] {
		const positionOverrides = this.buildPositionOverrides();
		const snapshot = snapshotFromEngine(this.nodeLayouts, this.portLayouts);
		const allObstaclesById = buildObstacleMapFromSpatialIndex(
			this.nodes,
			snapshot,
			positionOverrides,
		);
		const allObstacles = Array.from(allObstaclesById.values());
		const resolved = resolveConnectionsFromSpatialIndex(
			this.nodes,
			snapshot,
			positionOverrides,
		);

		return resolved.map((connection) => {
			if (this.routingMode === "orthogonal") {
				return {
					id: connection.id,
					d: computeOrthogonalPath(
						connection.from,
						connection.to,
						allObstaclesById,
						connection.outputNodeId,
						connection.inputNodeId,
						allObstacles,
					),
				};
			}

			return {
				id: connection.id,
				d: calculateEdgePath(
					this.routingMode,
					connection.from,
					connection.to,
					allObstacles,
					allObstacles,
				),
			};
		});
	}

	getSnapshot(): GraphSnapshot {
		return {
			nodes: structuredClone(this.nodes),
			routingMode: this.routingMode,
			nodeLayouts: Array.from(this.nodeLayouts.entries()).map(
				([nodeId, layout]) => ({
					nodeId,
					width: layout.width,
					height: layout.height,
				}),
			),
			portLayouts: Array.from(this.portLayouts.entries()).map(
				([key, layout]) => {
					const [nodeId, portName, transputType] = key.split("|");

					return {
						nodeId,
						portName,
						transputType: transputType as TransputType,
						offsetX: layout.offsetX,
						offsetY: layout.offsetY,
					};
				},
			),
		};
	}

	private buildPositionOverrides(): Record<string, Coordinate> | undefined {
		if (!this.dragNodeId || !this.dragPosition) {
			return undefined;
		}

		return { [this.dragNodeId]: this.dragPosition };
	}
}
