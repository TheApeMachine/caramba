import * as Comlink from "comlink";
import React from "react";
import type { EdgeRoutingMode } from "#/components/flume/connectionCalculator";
import { createConnections } from "#/components/flume/connectionCalculator";
import type { FlumeGraphWorkerHandle } from "#/components/flume/context";
import {
	portLayoutKey,
	type SpatialIndexSnapshot,
} from "#/components/flume/spatial-index";
import type { Coordinate, NodeMap } from "#/components/flume/types";
import type {
	ConnectionPathResult,
	GraphSnapshot,
} from "#/workers/flume-graph.types";
import type { FlumeGraphWorkerApi } from "#/workers/flume-graph.worker";

export type { FlumeGraphWorkerHandle };

const buildSnapshot = (
	nodes: NodeMap,
	routingMode: EdgeRoutingMode,
	indexSnapshot: SpatialIndexSnapshot,
): GraphSnapshot => ({
	nodes,
	routingMode,
	nodeLayouts: Array.from(indexSnapshot.nodeLayouts.entries()).map(
		([nodeId, layout]) => ({
			nodeId,
			width: layout.width,
			height: layout.height,
		}),
	),
	portLayouts: Array.from(indexSnapshot.portLayouts.entries()).map(
		([key, layout]) => {
			const [nodeId, portName, transputType] = key.split("|");

			return {
				nodeId,
				portName,
				transputType: transputType as "input" | "output",
				offsetX: layout.offsetX,
				offsetY: layout.offsetY,
			};
		},
	),
});

/*
useFlumeGraphWorker keeps graph topology and edge routing on a Comlink worker.
The main thread only creates SVG elements and applies returned path data.
*/
export function useFlumeGraphWorker(
	editorId: string,
	routingMode: EdgeRoutingMode,
	indexRef: React.RefObject<SpatialIndexSnapshot>,
): FlumeGraphWorkerHandle {
	const workerRef = React.useRef<Worker | null>(null);
	const apiRef = React.useRef<Comlink.Remote<FlumeGraphWorkerApi> | null>(null);
	const pathElementCacheRef = React.useRef<Map<string, SVGPathElement>>(
		new Map(),
	);
	const routingModeRef = React.useRef(routingMode);
	const pendingRecalcRef = React.useRef<{
		nodes: NodeMap;
		positionOverrides?: Record<string, Coordinate>;
	} | null>(null);
	const pendingDragRef = React.useRef<{
		nodeId: string;
		x: number;
		y: number;
	} | null>(null);
	const recalcFrameRef = React.useRef<number | null>(null);
	const dragFrameRef = React.useRef<number | null>(null);
	const graphReadyRef = React.useRef(false);

	routingModeRef.current = routingMode;

	React.useEffect(() => {
		const worker = new Worker(
			new URL("../../workers/flume-graph.worker.ts", import.meta.url),
			{ type: "module" },
		);
		const api = Comlink.wrap<FlumeGraphWorkerApi>(worker);

		workerRef.current = worker;
		apiRef.current = api;
		graphReadyRef.current = false;

		return () => {
			graphReadyRef.current = false;
			api[Comlink.releaseProxy]();
			worker.terminate();
			workerRef.current = null;
			apiRef.current = null;
			pathElementCacheRef.current.clear();
		};
	}, []);

	const applyPaths = React.useCallback((paths: ConnectionPathResult[]) => {
		const pathCache = pathElementCacheRef.current;

		for (const { id, d } of paths) {
			let path = pathCache.get(id);

			if (!path || !path.isConnected) {
				path =
					document.querySelector<SVGPathElement>(
						`[data-connection-id="${id}"]`,
					) ?? undefined;

				if (path) {
					pathCache.set(id, path);
				} else {
					pathCache.delete(id);
				}
			}

			path?.setAttribute("d", d);
		}
	}, []);

	const ensureWorkerGraph = React.useCallback(
		async (
			nodes: NodeMap,
		): Promise<Comlink.Remote<FlumeGraphWorkerApi> | null> => {
			const api = apiRef.current;

			if (!api) {
				return null;
			}

			const snapshot = buildSnapshot(
				nodes,
				routingModeRef.current,
				indexRef.current,
			);

			await api.loadSnapshot(snapshot);
			graphReadyRef.current = true;

			return api;
		},
		[indexRef],
	);

	const runRecalculate = React.useCallback(async () => {
		recalcFrameRef.current = null;
		const pending = pendingRecalcRef.current;
		pendingRecalcRef.current = null;

		if (!pending) {
			return;
		}

		createConnections(
			pending.nodes,
			{ scale: 1, translate: { x: 0, y: 0 } },
			editorId,
			routingModeRef.current,
			indexRef.current,
			pending.positionOverrides,
		);

		const api = await ensureWorkerGraph(pending.nodes);

		if (!api) {
			return;
		}

		if (pending.positionOverrides) {
			const [nodeId, position] =
				Object.entries(pending.positionOverrides)[0] ?? [];

			if (nodeId && position) {
				const paths = await api.updateDrag(nodeId, position.x, position.y);
				applyPaths(paths);
			}

			return;
		}

		await api.recalculate();
	}, [applyPaths, editorId, ensureWorkerGraph, indexRef]);

	const runDragUpdate = React.useCallback(async () => {
		dragFrameRef.current = null;
		const pending = pendingDragRef.current;
		pendingDragRef.current = null;

		if (!pending || !graphReadyRef.current) {
			return;
		}

		const api = apiRef.current;

		if (!api) {
			return;
		}

		const paths = await api.updateDrag(pending.nodeId, pending.x, pending.y);
		applyPaths(paths);
	}, [applyPaths]);

	const scheduleRecalculate = React.useCallback(
		(nodes: NodeMap, positionOverrides?: Record<string, Coordinate>) => {
			pendingRecalcRef.current = { nodes, positionOverrides };

			if (recalcFrameRef.current !== null) {
				return;
			}

			recalcFrameRef.current = requestAnimationFrame(() => {
				void runRecalculate();
			});
		},
		[runRecalculate],
	);

	const beginDrag = React.useCallback((nodeId: string) => {
		const api = apiRef.current;

		if (!api || !graphReadyRef.current) {
			return;
		}

		void api.beginDrag(nodeId);
	}, []);

	const updateDrag = React.useCallback(
		(nodeId: string, x: number, y: number) => {
			pendingDragRef.current = { nodeId, x, y };

			if (dragFrameRef.current !== null) {
				return;
			}

			dragFrameRef.current = requestAnimationFrame(() => {
				void runDragUpdate();
			});
		},
		[runDragUpdate],
	);

	const endDrag = React.useCallback((nodeId: string, x: number, y: number) => {
		const api = apiRef.current;

		if (!api) {
			return;
		}

		void api.endDrag(nodeId, x, y);
	}, []);

	return React.useMemo(
		() => ({
			beginDrag,
			updateDrag,
			endDrag,
			recalculate: scheduleRecalculate,
		}),
		[beginDrag, endDrag, scheduleRecalculate, updateDrag],
	);
}

export const syncPortLayoutToWorker = async (
	api: Comlink.Remote<FlumeGraphWorkerApi> | null,
	nodeId: string,
	portName: string,
	transputType: "input" | "output",
	offsetX: number,
	offsetY: number,
): Promise<void> => {
	if (!api) {
		return;
	}

	await api.setPortLayout(nodeId, portName, transputType, offsetX, offsetY);
};

export const syncNodeLayoutToWorker = async (
	api: Comlink.Remote<FlumeGraphWorkerApi> | null,
	nodeId: string,
	width: number,
	height: number,
): Promise<void> => {
	if (!api) {
		return;
	}

	await api.setNodeLayout(nodeId, width, height);
};

export { portLayoutKey };
