import type { FlumeConfig } from "#/components/flume";
import type { NodeMap } from "#/components/flume/types";
import type { Schema, TopologyNode } from "#/service/compute";
import { portTypeForKind } from "./ports";

const NODE_SPACING_X = 260;
const NODE_WIDTH = 200;

/*
buildSubGraph converts a block's topology node list into a fully wired Flume
NodeMap. Layout is left-to-right by topological rank. Connections use the
actual schema port names resolved positionally from the `in`/`out` arrays.
*/
export function buildSubGraph(
	topologyNodes: TopologyNode[],
	schemas: Record<string, Schema>,
): NodeMap {
	// Map each output signal name → the node id that produces it.
	const outToNode: Record<string, string> = {};
	for (const node of topologyNodes) {
		for (const sig of node.out) {
			outToNode[sig] = node.id;
		}
	}

	// Topological rank (column) for left-to-right layout.
	const rank: Record<string, number> = {};
	const assignRank = (id: string, visited = new Set<string>()): number => {
		if (rank[id] !== undefined) return rank[id];
		if (visited.has(id)) return 0;
		visited.add(id);
		const node = topologyNodes.find((n) => n.id === id);
		if (!node) return 0;
		const parentRanks = node.in
			.map((sig) => outToNode[sig])
			.filter(Boolean)
			.map((pid) => assignRank(pid, visited));
		rank[id] = parentRanks.length ? Math.max(...parentRanks) + 1 : 0;
		return rank[id];
	};
	for (const node of topologyNodes) assignRank(node.id);

	// Place nodes.
	const rankCounts: Record<number, number> = {};
	const nodeMap: NodeMap = {};
	for (const node of topologyNodes) {
		const col = rank[node.id] ?? 0;
		const row = rankCounts[col] ?? 0;
		rankCounts[col] = row + 1;
		nodeMap[node.id] = {
			id: node.id,
			type: node.op,
			width: NODE_WIDTH,
			x: col * NODE_SPACING_X,
			y: row * 140,
			inputData: {},
			connections: { inputs: {}, outputs: {} },
		};
	}

	// Wire connections. `in[i]` is the signal arriving at the i-th input port
	// slot; `out[i]` is the signal leaving the i-th output port slot. We look
	// up the schema for each op to get the actual port names by index.
	for (const node of topologyNodes) {
		const destSchema = schemas[node.op];
		if (!destSchema) continue;

		node.in.forEach((signal, slotIdx) => {
			const sourceId = outToNode[signal];
			if (!sourceId) return;

			const sourceNode = topologyNodes.find((n) => n.id === sourceId);
			const sourceSchema = schemas[sourceNode?.op ?? ""];
			if (!sourceNode || !sourceSchema) return;

			const sourceOutSlot = sourceNode.out.indexOf(signal);
			const destPortName = destSchema.inputs?.[slotIdx]?.name;
			const srcPortName = sourceSchema.outputs?.[sourceOutSlot]?.name;
			if (!destPortName || !srcPortName) return;

			nodeMap[node.id].connections.inputs[destPortName] = [
				{ nodeId: sourceId, portName: srcPortName },
			];
			if (!nodeMap[sourceId].connections.outputs[srcPortName]) {
				nodeMap[sourceId].connections.outputs[srcPortName] = [];
			}
			nodeMap[sourceId].connections.outputs[srcPortName].push({
				nodeId: node.id,
				portName: destPortName,
			});
		});
	}

	return nodeMap;
}

/*
registerNodes adds one Flume node type per schema entry. Block schemas with
system.topology get a defaultSubGraph so expanding the node shows its internal
wiring immediately.
*/
export function registerNodes(
	config: FlumeConfig,
	schemas: Record<string, Schema>,
	allSchemas: Record<string, Schema> = schemas,
): FlumeConfig {
	for (const schema of Object.values(schemas)) {
		const flowPort = portTypeForKind(schema.kind);
		const topologyNodes = schema.system?.topology?.nodes;
		const defaultSubGraph =
			topologyNodes && topologyNodes.length > 0
				? buildSubGraph(topologyNodes, allSchemas)
				: undefined;

		config.addNodeType({
			type: schema.op,
			label: schema.label || schema.name,
			description: schema.description,
			category: schema.category,
			initialWidth: schema.initial_width || 220,
			inputs: (ports) =>
				(schema.inputs ?? []).map((inp) =>
					ports[flowPort]({ name: inp.name, label: inp.name }),
				),
			outputs: (ports) =>
				(schema.outputs ?? []).map((out) =>
					ports[flowPort]({ name: out.name, label: out.name }),
				),
			...(defaultSubGraph ? { defaultSubGraph } : {}),
		});
	}

	return config;
}
