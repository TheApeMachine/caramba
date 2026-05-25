"use client";

import { useLiveQuery } from "@tanstack/react-db";
import { useCallback, useEffect, useMemo, useState } from "react";
import { researchGraphCollection } from "#/collections/research_graph";
import { useOperations } from "#/service/compute";
import { buildFlumeConfigFromSchemas } from "./build-config-from-schemas";
import { NodeEditor } from "./NodeEditor";
import type { NodeMap } from "./types";

const demoDefaultNodes = [
	{ type: "source", x: 120, y: 180 },
	{ type: "gate", x: 420, y: 180 },
	{ type: "sink", x: 720, y: 180 },
] as const;

const demoDefaultConnections = [
	{
		output: { nodeType: "source", portName: "value" },
		input: { nodeType: "gate", portName: "in" },
	},
	{
		output: { nodeType: "gate", portName: "out" },
		input: { nodeType: "sink", portName: "value" },
	},
] as const;

const LOCAL_GRAPH_ID = "local-default";

type FlumeEditorProps = {
	projectId?: string;
};

export const FlumeEditor = ({ projectId }: FlumeEditorProps) => {
	const { data: operations, isPending, isError, isSuccess } = useOperations();
	const graphId = projectId ?? LOCAL_GRAPH_ID;
	const [nodes, setNodes] = useState<NodeMap>({});
	const [graphHydrated, setGraphHydrated] = useState(false);

	const graphQuery = useLiveQuery((query) =>
		query.from({ graph: researchGraphCollection }),
	);

	const storedRow = useMemo(
		() => graphQuery.data?.find((row) => row.id === graphId),
		[graphId, graphQuery.data],
	);

	useEffect(() => {
		if (graphQuery.isLoading) {
			return;
		}

		const nextNodes = (storedRow?.nodes as NodeMap | undefined) ?? {};
		setNodes(nextNodes);
		setGraphHydrated(true);
	}, [graphQuery.isLoading, storedRow?.nodes]);

	const persistGraph = useCallback(
		(nextNodes: NodeMap) => {
			setNodes(nextNodes);

			const updatedAt = new Date();

			if (storedRow) {
				void researchGraphCollection.update(
					storedRow.id,
					{ metadata: {} },
					(draft) => {
						draft.nodes = nextNodes;
						draft.updated_at = updatedAt;
					},
				);
				return;
			}

			void researchGraphCollection.insert({
				id: graphId,
				project_id: projectId ?? null,
				nodes: nextNodes,
				updated_at: updatedAt,
			});
		},
		[graphId, projectId, storedRow],
	);

	const flumeConfig = useMemo(
		() => buildFlumeConfigFromSchemas(operations ?? {}),
		[operations],
	);

	const editorMode = isError || !isSuccess ? "builtin-only" : "full";

	if (isPending || graphQuery.isLoading || !graphHydrated) {
		return (
			<div className="flex min-h-[75vh] flex-1 items-center justify-center text-muted-foreground text-sm">
				Loading operation schemas…
			</div>
		);
	}

	return (
		<div className="flex min-h-[75vh] flex-1 flex-col gap-3">
			{isError ? (
				<p className="shrink-0 text-muted-foreground text-sm">
					Could not load backend operations — built-in node types only.
				</p>
			) : null}
			<NodeEditor
				key={`${editorMode}:${graphId}`}
				className="min-h-0 flex-1"
				defaultConnections={[...demoDefaultConnections]}
				defaultNodes={[...demoDefaultNodes]}
				nodeTypes={flumeConfig.nodeTypes}
				nodes={nodes}
				onChange={persistGraph}
				portTypes={flumeConfig.portTypes}
				style={{ minHeight: isError ? "70vh" : "75vh" }}
			/>
		</div>
	);
};
