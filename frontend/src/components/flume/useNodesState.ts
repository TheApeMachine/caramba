import { eq, useLiveQuery } from "@tanstack/react-db";
import React from "react";
import { researchGraphCollection } from "#/collections/research_graph";
import {
	connectNodesReducer,
	getInitialNodes,
	type NodesAction,
	NodesActionType,
} from "#/components/flume/nodesReducer";
import nodesReducer from "#/components/flume/nodesReducer";
import type { ToastAction } from "#/components/flume/toastsReducer";
import type {
	DefaultConnection,
	DefaultNode,
	NodeMap,
	NodeTypeMap,
	PortTypeMap,
} from "#/components/flume/types";

/*
useNodesState makes the graph topology read from and written to the
research_graphs collection when graphId is provided, falling back to a
local useReducer when it isn't (the embedded sub-editor case). Either
way the caller gets back a `[nodes, dispatch]` pair identical to the
original useReducer shape.
*/
export type UseNodesStateOptions = {
	graphId?: string;
	projectId?: string | null;
	initialNodes?: NodeMap;
	defaultNodes?: DefaultNode[];
	defaultConnections?: DefaultConnection[];
	nodeTypes: NodeTypeMap;
	portTypes: PortTypeMap;
	context: unknown;
	getEnvironment: Parameters<typeof connectNodesReducer>[1];
	setSideEffectToasts: React.Dispatch<
		React.SetStateAction<ToastAction | undefined>
	>;
};

const useCollectionBackedNodes = (
	options: UseNodesStateOptions & { graphId: string },
): [NodeMap, React.Dispatch<NodesAction>] => {
	const {
		graphId,
		projectId,
		initialNodes,
		defaultNodes,
		defaultConnections,
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
	const nodes = (row?.nodes as NodeMap | undefined) ?? {};

	// Seed the row on first observation. Runs once per graphId because we
	// gate on row absence; the collection.insert below is idempotent for
	// repeat fires (it will throw on conflict, which we swallow).
	const seededRef = React.useRef<string | null>(null);

	React.useEffect(() => {
		if (isLoading) return;
		if (row) return;
		if (seededRef.current === graphId) return;
		seededRef.current = graphId;

		const seeded = getInitialNodes(
			initialNodes ?? {},
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
				nodes: seeded,
				updated_at: new Date(),
			});
		} catch {
			// Another mount got there first — fine, useLiveQuery will pick it up.
		}
	}, [
		context,
		defaultConnections,
		defaultNodes,
		graphId,
		initialNodes,
		isLoading,
		nodeTypes,
		portTypes,
		projectId,
		row,
	]);

	const wrappedRef = React.useRef(
		connectNodesReducer(nodesReducer, getEnvironment, setSideEffectToasts),
	);

	React.useEffect(() => {
		wrappedRef.current = connectNodesReducer(
			nodesReducer,
			getEnvironment,
			setSideEffectToasts,
		);
	}, [getEnvironment, setSideEffectToasts]);

	const dispatch = React.useCallback<React.Dispatch<NodesAction>>(
		(action) => {
			researchGraphCollection.update(graphId, (draft) => {
				const current = (draft.nodes as NodeMap | undefined) ?? {};
				const next = wrappedRef.current(current, action);

				if (next === current) {
					return;
				}

				draft.nodes = next;
				draft.updated_at = new Date();
			});
		},
		[graphId],
	);

	return [nodes, dispatch];
};

const useReducerBackedNodes = (
	options: UseNodesStateOptions,
): [NodeMap, React.Dispatch<NodesAction>] => {
	const {
		initialNodes,
		defaultNodes,
		defaultConnections,
		nodeTypes,
		portTypes,
		context,
		getEnvironment,
		setSideEffectToasts,
	} = options;

	const wrapped = React.useMemo(
		() =>
			connectNodesReducer(nodesReducer, getEnvironment, setSideEffectToasts),
		[getEnvironment, setSideEffectToasts],
	);

	const [nodes, dispatch] = React.useReducer(
		wrapped,
		{},
		() =>
			getInitialNodes(
				initialNodes ?? {},
				defaultNodes ?? [],
				nodeTypes,
				portTypes,
				context,
				defaultConnections ?? [],
			),
	);

	return [nodes, dispatch];
};

export const useNodesState = (
	options: UseNodesStateOptions,
): [NodeMap, React.Dispatch<NodesAction>] => {
	if (options.graphId) {
		// biome-ignore lint/correctness/useHookAtTopLevel: graphId is stable per mount.
		return useCollectionBackedNodes({ ...options, graphId: options.graphId });
	}

	// biome-ignore lint/correctness/useHookAtTopLevel: graphId is stable per mount.
	return useReducerBackedNodes(options);
};

// Re-export so consumers can dispatch RECONCILE_NODE_TYPES without importing
// the reducer module directly.
export { NodesActionType };
