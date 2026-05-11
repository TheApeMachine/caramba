"use client";

import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Flex } from "#/components/ui/flex";
import {
	Select,
	SelectItem,
	SelectPopup,
	SelectTrigger,
	SelectValue,
} from "#/components/ui/select";
import { Graph } from "./core/graph";
import { ModelScope } from "./component";

const BASE = "http://localhost:8118";

function useModelList() {
	return useQuery<string[]>({
		queryKey: ["modelscope"],
		queryFn: () => fetch(`${BASE}/backend/modelscope`).then((r) => r.json()),
	});
}

function useInspectModel(name: string) {
	return useQuery({
		queryKey: ["modelscope", name],
		queryFn: () =>
			fetch(
				`${BASE}/backend/modelscope/inspect?path=${encodeURIComponent(`models/${name}`)}`,
			).then((r) => r.json()),
		enabled: Boolean(name),
	});
}

/*
ModelScopeInspector wraps ModelScope with a dropdown of model files found in
the backend's models/ directory. Selecting one fetches the parsed graph.
Rendered client-side only to avoid SSR/hydration mismatches from localStorage
reads inside ModelScope.
*/
export function ModelScopeInspector() {
	const [mounted, setMounted] = useState(false);
	const [selected, setSelected] = useState("");
	const { data: modelNames = [] } = useModelList();
	const { data: graphData, isLoading, error } = useInspectModel(selected);

	useEffect(() => { setMounted(true); }, []);

	const graph = useMemo(() => {
		if (!graphData) return undefined;
		const g = new Graph();
		g.loadFromData(graphData);
		return g;
	}, [graphData]);

	if (!mounted) return null;

	return (
		<Flex.Column fullWidth fullHeight gap={2}>
			<Flex.Row
				align="center"
				className="shrink-0 rounded-xl border bg-muted/48 px-3 py-2"
				gap={3}
			>
				<span className="whitespace-nowrap text-muted-foreground text-xs">
					Model
				</span>
				<Select onValueChange={(v) => { if (v) setSelected(v); }} value={selected}>
					<SelectTrigger className="min-w-64" size="sm">
						<SelectValue placeholder="Select a model…" />
					</SelectTrigger>
					<SelectPopup>
						{modelNames.map((name) => (
							<SelectItem key={name} value={name}>{name}</SelectItem>
						))}
					</SelectPopup>
				</Select>

				{isLoading && (
					<span className="text-muted-foreground text-xs">Parsing…</span>
				)}
				{error && (
					<span className="text-destructive text-xs">
						{(error as Error).message}
					</span>
				)}
				{graph && !error && (
					<span className="ml-auto whitespace-nowrap text-muted-foreground text-xs">
						{Object.keys(graph.nodes).length} nodes ·{" "}
						{Object.keys(graph.edges).length} edges
					</span>
				)}
			</Flex.Row>

			<Flex.Column className="min-h-0 flex-1" fullHeight fullWidth>
				<ModelScope graph={graph} />
			</Flex.Column>
		</Flex.Column>
	);
}
