import { eq, useLiveQuery } from "@tanstack/react-db";
import React from "react";
import { researchGraphCollection } from "#/collections/research_graph";
import nodesReducer, {
	connectNodesReducer,
	getInitialNodes,
	type NodesAction,
	NodesActionType,
	reconcileNodes,
} from "#/components/flume/nodesReducer";
import type { ToastAction } from "#/components/flume/toastsReducer";
import type {
	DefaultConnection,
	DefaultNode,
	NodeMap,
	NodeTypeMap,
	PortTypeMap,
} from "#/components/flume/types";

/*
useNodesState is the only sanctioned source of truth for Flume graph
topology. It always reads from and writes to researchGraphCollection;
there is no useReducer fallback. Subgraph editors get composite
graphIds (e.g. "parent:nodeId") so they also persist through the
collection — the editor doesn't carry inline state anywhere.
*/

export type UseNodesStateOptions = {
	graphId: string;
	projectId?: string | null;
	nodeTypes: NodeTypeMap;
	portTypes: PortTypeMap;
	context: unknown;
	getEnvironment: Parameters<typeof connectNodesReducer>[1];
	setSideEffectToasts: React.Dispatch<
		React.SetStateAction<ToastAction | undefined>
	>;
};

export type UseNodesStateResult = {
	nodes: NodeMap;
	dispatch: React.Dispatch<NodesAction>;
	isLoading: boolean;
	hasRow: boolean;
	/**
	 * Inserts an initial topology built from defaultNodes/defaultConnections.
	 * Idempotent: returns silently if a row already exists.
	 */
	seed: (params: {
		defaultNodes?: DefaultNode[];
		defaultConnections?: DefaultConnection[];
	}) => void;
};

export const useNodesState = (
	options: UseNodesStateOptions,
): UseNodesStateResult => {
	const {
		graphId,
		projectId,
		nodeTypes,
		portTypes,
		context,
		getEnvironment,
		setSideEffectToasts,
	} = options;

	const { data, isLoading } = useLiveQuery(
		(query) =>
			query
				.from({ graph: researchGraphCollection })
				.where(({ graph }) => eq(graph.id, graphId))
				.select(({ graph }) => ({
					id: graph.id,
					nodes: graph.nodes,
				})),
		[graphId],
	);

	const row = data?.[0];
	const rawNodes = (row?.nodes as NodeMap | undefined) ?? {};

	// Normalize on every read: persisted rows can drift from the current
	// node/port type registry (operations added, removed, signatures
	// changed). reconcileNodes drops unknown types, fills missing port
	// slots, and refreshes default inputData. This keeps the worker and
	// renderer fed with a valid FlumeNode shape regardless of what was
	// persisted, without forcing a write back to the collection unless
	// the user actually edits.
	const nodes = React.useMemo(
		() => reconcileNodes(rawNodes, nodeTypes, portTypes, context),
		[rawNodes, nodeTypes, portTypes, context],
	);

	// connectNodesReducer takes lazy env + toast accessors so it's safe
	// to instantiate once. Recreating it across renders previously caused
	// in-flight dispatches to land on stale reducer instances and silently
	// drop their writes.
	const wrappedRef = React.useRef<ReturnType<
		typeof connectNodesReducer
	> | null>(null);

	if (wrappedRef.current === null) {
		wrappedRef.current = connectNodesReducer(
			nodesReducer,
			getEnvironment,
			setSideEffectToasts,
		);
	}

	const dispatch = React.useCallback<React.Dispatch<NodesAction>>(
		(action) => {
			researchGraphCollection.update(graphId, (draft) => {
				const wrapped = wrappedRef.current;
				if (!wrapped) return;

				// Reconcile the draft's raw nodes before handing to the reducer.
				// Guarantees the reducer always operates on a valid FlumeNode
				// shape even if the persisted row is stale relative to the
				// current node/port type registry.
				const raw = (draft.nodes as NodeMap | undefined) ?? {};
				const current = reconcileNodes(raw, nodeTypes, portTypes, context);
				const next = wrapped(current, action);

				if (next === current) {
					return;
				}

				draft.nodes = next;
				draft.updated_at = new Date();
			});
		},
		[context, graphId, nodeTypes, portTypes],
	);

	const seed = React.useCallback<UseNodesStateResult["seed"]>(
		({ defaultNodes, defaultConnections }) => {
			const existing = researchGraphCollection.get(graphId);
			if (existing) return;

			const seeded = getInitialNodes(
				{},
				defaultNodes ?? [],
				nodeTypes,
				portTypes,
				context,
				defaultConnections ?? [],
			);

			try {
				researchGraphCollection.insert({
					id: graphId,
					project_id: projectId ?? null,
					schema_version: 1,
					nodes: seeded,
					comments: {},
					viewport: { scale: 1, translate: { x: 0, y: 0 } },
					updated_at: new Date(),
				});
			} catch {
				// Lost the race; useLiveQuery will pick up the winner.
			}
		},
		[context, graphId, nodeTypes, portTypes, projectId],
	);

	return {
		nodes,
		dispatch,
		isLoading,
		hasRow: row !== undefined,
		seed,
	};
};

export { NodesActionType };
