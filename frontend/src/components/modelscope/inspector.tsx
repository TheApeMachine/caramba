"use client";

import { useAuth } from "@clerk/tanstack-react-start";
import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { Badge } from "#/components/ui/badge";
import { Flex } from "#/components/ui/flex";
import {
	Select,
	SelectItem,
	SelectPopup,
	SelectTrigger,
	SelectValue,
} from "#/components/ui/select";
import { Tabs } from "#/components/ui/tabs";
import { Typography } from "#/components/ui/typography";
import {
	backendAuthHeaders,
	backendBaseURL,
	type ClerkGetToken,
} from "#/lib/backend-http";
import { ModelScope } from "./component";
import { Graph } from "./core/graph";
import { LogitLensPanel } from "./logit-lens-panel";
import { NodeInspectorPanel } from "./node-inspector-panel";

const useModelList = (getToken: ClerkGetToken) => {
	return useQuery<string[]>({
		queryKey: ["modelscope"],
		queryFn: async () => {
			const headers = await backendAuthHeaders(getToken);
			const response = await fetch(`${backendBaseURL()}/backend/modelscope`, {
				headers,
			});

			return response.json();
		},
	});
};

const useInspectModel = (name: string, getToken: ClerkGetToken) => {
	return useQuery({
		queryKey: ["modelscope", name],
		queryFn: async () => {
			const headers = await backendAuthHeaders(getToken);
			const inspectURL = `${backendBaseURL()}/backend/modelscope/inspect?path=${encodeURIComponent(`models/${name}`)}`;
			const response = await fetch(inspectURL, { headers });

			return response.json();
		},
		enabled: Boolean(name),
	});
};

const ToolbarRow = ({
	selected,
	onSelect,
	modelNames,
	loading,
	error,
	stats,
}: {
	selected: string;
	onSelect: (next: string) => void;
	modelNames: ReadonlyArray<string>;
	loading: boolean;
	error: Error | null;
	stats: { nodes: number; edges: number } | null;
}) => {
	return (
		<Flex.Row
			align="center"
			className="shrink-0 rounded-xl border bg-muted/48 px-3 py-2"
			gap={3}
		>
			<Typography.Span
				className="whitespace-nowrap text-xs"
				variant="muted"
			>
				Model
			</Typography.Span>
			<Select
				onValueChange={(value) => {
					if (value) onSelect(value);
				}}
				value={selected}
			>
				<SelectTrigger className="min-w-64" size="sm">
					<SelectValue placeholder="Select a model…" />
				</SelectTrigger>
				<SelectPopup>
					{modelNames.map((name) => (
						<SelectItem key={name} value={name}>
							{name}
						</SelectItem>
					))}
				</SelectPopup>
			</Select>

			{loading ? (
				<Typography.Span className="text-xs" variant="muted">
					Parsing…
				</Typography.Span>
			) : null}
			{error ? (
				<Typography.Span className="text-xs" variant="error">
					{error.message}
				</Typography.Span>
			) : null}
			{stats ? (
				<Flex.Row className="ml-auto items-center gap-2">
					<Badge size="sm" variant="outline">
						{stats.nodes.toLocaleString()} nodes
					</Badge>
					<Badge size="sm" variant="outline">
						{stats.edges.toLocaleString()} edges
					</Badge>
				</Flex.Row>
			) : null}
		</Flex.Row>
	);
};

/*
estimateLayerCount walks the graph and finds the largest blk.N index it
can extract. Falls back to 12 if no transformer-style blocks are
detected so the LogitLens panel still has a meaningful grid.
*/
const estimateLayerCount = (graph: Graph | undefined): number => {
	if (!graph) return 12;

	let max = -1;

	for (const name of Object.keys(graph.nodes)) {
		const match = name.match(/(?:blk|layers?|h)\.(\d+)/);

		if (match) {
			const index = Number.parseInt(match[1] ?? "", 10);
			if (Number.isFinite(index) && index > max) {
				max = index;
			}
		}
	}

	return max < 0 ? 12 : max + 1;
};

/*
ModelScopeInspector wraps ModelScope with a model dropdown, a graph
viewport on the left, and a tabbed side column on the right for node
inspection and the Logit Lens UI.
*/
export const ModelScopeInspector = () => {
	const [mounted, setMounted] = useState(false);
	const [selected, setSelected] = useState("");
	const [selectedNode, setSelectedNode] = useState<string | null>(null);
	const [sidePanel, setSidePanel] = useState<"node" | "logitlens">("node");
	const { getToken } = useAuth();
	const { data: modelNames = [] } = useModelList(getToken);
	const {
		data: graphData,
		isLoading,
		error,
	} = useInspectModel(selected, getToken);

	useEffect(() => {
		setMounted(true);
	}, []);

	const graph = useMemo(() => {
		if (!graphData) return undefined;
		const next = new Graph();
		next.loadFromData(graphData);
		return next;
	}, [graphData]);

	const layerCount = useMemo(() => estimateLayerCount(graph), [graph]);

	const stats = graph
		? {
				nodes: Object.keys(graph.nodes).length,
				edges: Object.keys(graph.edges).length,
			}
		: null;

	if (!mounted) return null;

	return (
		<Flex.Column fullWidth fullHeight gap={2}>
			<ToolbarRow
				error={(error as Error | null) ?? null}
				loading={isLoading}
				modelNames={modelNames}
				onSelect={setSelected}
				selected={selected}
				stats={stats}
			/>

			<div className="grid min-h-0 flex-1 grid-cols-1 gap-2 lg:grid-cols-[minmax(0,1fr)_360px]">
				<Flex.Column className="min-h-0" fullHeight fullWidth>
					<ModelScope graph={graph} onNodeSelect={(_, name) => setSelectedNode(name)} />
				</Flex.Column>

				<Flex.Column
					className="min-h-0 overflow-hidden rounded-xl border bg-card/40"
					fullHeight
				>
					<Tabs
						className="flex min-h-0 flex-1 flex-col"
						onValueChange={(value) => {
							if (value === "node" || value === "logitlens") {
								setSidePanel(value);
							}
						}}
						value={sidePanel}
					>
						<Tabs.List className="shrink-0 border-b px-2">
							<Tabs.Tab value="node">Node</Tabs.Tab>
							<Tabs.Tab value="logitlens">Logit Lens</Tabs.Tab>
						</Tabs.List>
						<Tabs.Panel
							className="min-h-0 flex-1 overflow-auto"
							value="node"
						>
							<NodeInspectorPanel
								graph={graph}
								onSelect={setSelectedNode}
								selectedName={selectedNode}
							/>
						</Tabs.Panel>
						<Tabs.Panel
							className="min-h-0 flex-1 overflow-auto"
							value="logitlens"
						>
							<LogitLensPanel layerCount={layerCount} />
						</Tabs.Panel>
					</Tabs>
				</Flex.Column>
			</div>
		</Flex.Column>
	);
};
