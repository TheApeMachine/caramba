import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { NodeMap } from "#/components/flume";

export type OperationPort = {
	name: string;
	type: string;
	description: string;
};

export type ConfigParam = {
	name: string;
	type: string;
	default?: unknown;
	description?: string;
};

export type TopologyNode = {
	id: string;
	op: string;
	in: string[];
	out: string[];
};

export type Schema = {
	kind: string;
	category: string;
	op: string;
	name: string;
	label: string;
	description: string;
	initial_width: number;
	inputs: OperationPort[];
	outputs: OperationPort[];
	config: ConfigParam[];
	system?: {
		topology: {
			nodes: TopologyNode[];
		};
	};
};

const BASE = "http://localhost:8118";

const fetchSchemas = (kind: "operation" | "optimizer" | "block") =>
	fetch(`${BASE}/backend/compute/${kind}`).then<Record<string, Schema>>((res) =>
		res.json(),
	);

export function useOperations() {
	return useQuery({
		queryKey: ["compute", "operation"],
		queryFn: () => fetchSchemas("operation"),
	});
}

export function useOptimizers() {
	return useQuery({
		queryKey: ["compute", "optimizer"],
		queryFn: () => fetchSchemas("optimizer"),
	});
}

export function useBlocks() {
	return useQuery({
		queryKey: ["compute", "block"],
		queryFn: () => fetchSchemas("block"),
	});
}

export function useArchitectures() {
	return useQuery<string[]>({
		queryKey: ["architecture"],
		queryFn: () =>
			fetch(`${BASE}/backend/architecture`).then((res) => res.json()),
	});
}

export function useLoadArchitecture(name: string) {
	return useQuery<NodeMap>({
		queryKey: ["architecture", name],
		queryFn: () =>
			fetch(`${BASE}/backend/architecture/${name}`).then((res) => res.json()),
		enabled: Boolean(name) && !name.startsWith("block.model."),
	});
}

export function useSaveArchitecture() {
	const queryClient = useQueryClient();

	return useMutation({
		mutationFn: ({ name, nodes }: { name: string; nodes: NodeMap }) =>
			fetch(`${BASE}/backend/architecture/${name}`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(nodes),
			}).then((res) => res.json()),
		onSuccess: () => queryClient.invalidateQueries({ queryKey: ["architecture"] }),
	});
}
