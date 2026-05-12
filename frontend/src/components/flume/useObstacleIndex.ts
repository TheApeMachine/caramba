import React from "react";
import { connectionId } from "#/components/flume/connectionCalculator";
import type { EdgeRoutingMode, ObstacleRect } from "#/components/flume/connectionCalculator";
import { CONNECTIONS_ID } from "#/components/flume/constants";
import type { Coordinate, FlumeNode } from "#/components/flume/types";
import type {
	ConnectionPathRequest,
	WorkerRequest,
	WorkerResponse,
} from "#/workers/flume-connections.worker";

export type ObstacleIndex = Map<string, ObstacleRect>;

// Safe default so consumers never receive { current: null }.
const defaultObstacleIndex: React.RefObject<ObstacleIndex> = {
	current: new Map<string, ObstacleRect>(),
};

export const ObstacleIndexContext = React.createContext<
	React.RefObject<ObstacleIndex>
>(defaultObstacleIndex);

function getStageContainer(editorId: string): HTMLDivElement | null {
	return document.getElementById(
		`${CONNECTIONS_ID}${editorId}`,
	) as HTMLDivElement | null;
}

function safeInv(scale: number | null | undefined): number {
	const s = scale ?? 1;
	return 1 / (s === 0 ? 1 : s);
}

/*
useObstacleIndex maintains a Map<nodeId, ObstacleRect> in canvas-space using
ResizeObserver and MutationObserver. Reads during drag are zero-cost — no
synchronous layout queries on the hot path.

scale is accepted as a plain number (not a ref) so the effect re-runs when it
changes and obstacle positions are remeasured at the new zoom level.
*/
export function useObstacleIndex(
	editorId: string,
	scale: number,
): React.RefObject<ObstacleIndex> {
	const indexRef = React.useRef<ObstacleIndex>(new Map());

	React.useEffect(() => {
		const container = getStageContainer(editorId);
		if (!container) return;

		const measure = (el: Element) => {
			const nid = el.getAttribute("data-node-id");
			if (!nid) return;

			const stageRect = container.getBoundingClientRect();
			const rect = el.getBoundingClientRect();
			const hw = stageRect.width / 2;
			const hh = stageRect.height / 2;
			const inv = safeInv(scale);

			indexRef.current.set(nid, {
				left:   inv * (rect.left   - stageRect.x - hw),
				right:  inv * (rect.right  - stageRect.x - hw),
				top:    inv * (rect.top    - stageRect.y - hh),
				bottom: inv * (rect.bottom - stageRect.y - hh),
			});
		};

		const resizeObserver = new ResizeObserver((entries) => {
			for (const entry of entries) measure(entry.target);
		});

		const attachAll = (root: Element) => {
			for (const el of root.querySelectorAll(
				'[data-flume-component="node"][data-node-id]',
			)) {
				resizeObserver.observe(el);
				measure(el);
			}
		};

		const mutationObserver = new MutationObserver((records) => {
			for (const record of records) {
				for (const node of record.addedNodes) {
					if (!(node instanceof Element)) continue;
					if (node.matches('[data-flume-component="node"][data-node-id]')) {
						resizeObserver.observe(node);
						measure(node);
					} else {
						attachAll(node);
					}
				}
				for (const node of record.removedNodes) {
					if (!(node instanceof Element)) continue;
					const nid = node.getAttribute("data-node-id");
					if (nid) {
						resizeObserver.unobserve(node);
						indexRef.current.delete(nid);
					}
				}
			}
		});

		attachAll(container);
		mutationObserver.observe(container, { childList: true, subtree: true });

		return () => {
			resizeObserver.disconnect();
			mutationObserver.disconnect();
			indexRef.current.clear();
		};
	}, [editorId, scale]);

	return indexRef;
}

/*
useConnectionWorker manages a Web Worker that computes SVG edge paths off the
main thread. recalculate() is called after drag moves; results are written
directly to SVG path elements in the DOM.
*/
export function useConnectionWorker(
	editorId: string,
	routingMode: EdgeRoutingMode,
	indexRef: React.RefObject<ObstacleIndex>,
	scaleRef: React.RefObject<number>,
) {
	const workerRef = React.useRef<Worker | null>(null);

	React.useEffect(() => {
		const worker = new Worker(
			new URL("../../workers/flume-connections.worker.ts", import.meta.url),
			{ type: "module" },
		);

		worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
			for (const { id, d } of event.data.paths) {
				const path = document.querySelector<SVGPathElement>(
					`[data-connection-id="${id}"]`,
				);
				path?.setAttribute("d", d);
			}
		};

		workerRef.current = worker;
		return () => worker.terminate();
	}, []);

	return React.useCallback(
		(nodes: Record<string, FlumeNode>) => {
			const worker = workerRef.current;
			if (!worker || routingMode !== "orthogonal") return;

			const container = getStageContainer(editorId);
			if (!container) return;

			const stageRect = container.getBoundingClientRect();
			const hw = stageRect.width / 2;
			const hh = stageRect.height / 2;
			const inv = safeInv(scaleRef.current);

			const allObstacles = Array.from(indexRef.current.entries());
			const connections: ConnectionPathRequest[] = [];

			for (const node of Object.values(nodes)) {
				if (!node.connections?.inputs) continue;

				for (const [inputName, outputs] of Object.entries(node.connections.inputs)) {
					for (const output of outputs) {
						const fromEl = container.querySelector(
							`[data-node-id="${output.nodeId}"] [data-port-name="${output.portName}"][data-port-transput-type="output"]`,
						);
						const toEl = container.querySelector(
							`[data-node-id="${node.id}"] [data-port-name="${inputName}"][data-port-transput-type="input"]`,
						);
						if (!fromEl || !toEl) continue;

						const fromRect = fromEl.getBoundingClientRect();
						const toRect = toEl.getBoundingClientRect();

						const from: Coordinate = {
							x: inv * (fromRect.x - stageRect.x + fromRect.width / 2 - hw),
							y: inv * (fromRect.y - stageRect.y + fromRect.height / 2 - hh),
						};
						const to: Coordinate = {
							x: inv * (toRect.x - stageRect.x + toRect.width / 2 - hw),
							y: inv * (toRect.y - stageRect.y + toRect.height / 2 - hh),
						};

						connections.push({
							id: connectionId(output.nodeId, output.portName, node.id, inputName),
							from,
							to,
							routingMode,
							obstaclesVertical: allObstacles.map(([, r]) => r),
							obstaclesHorizontal: allObstacles
								.filter(([nid]) => nid !== output.nodeId && nid !== node.id)
								.map(([, r]) => r),
						});
					}
				}
			}

			if (connections.length > 0) {
				worker.postMessage({ connections } satisfies WorkerRequest);
			}
		},
		[editorId, routingMode, indexRef, scaleRef],
	);
}
