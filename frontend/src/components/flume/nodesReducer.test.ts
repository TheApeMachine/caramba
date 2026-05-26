import { describe, expect, it } from "vitest";
import { buildFlumeConfigFromSchemas } from "./build-config-from-schemas";
import nodesReducer, {
	applyDefaultConnections,
	getInitialNodes,
	NodesActionType,
	pruneDanglingConnections,
} from "./nodesReducer";
import type { DefaultConnection, FlumeNode } from "./types";

describe("reconcileNodes via getInitialNodes", () => {
	it("normalizes persisted nodes missing connections", () => {
		const config = buildFlumeConfigFromSchemas({});

		const nodes = getInitialNodes(
			{
				source: {
					id: "source",
					type: "source",
					width: 280,
					x: 120,
					y: 180,
				} as FlumeNode,
			},
			[],
			config.nodeTypes,
			config.portTypes,
			{},
		);

		expect(nodes.source.connections).toEqual({ inputs: {}, outputs: {} });
		expect(nodes.source.inputData).toEqual({});
	});

	it("drops nodes whose types are not in the current registry", () => {
		const config = buildFlumeConfigFromSchemas({});

		const nodes = getInitialNodes(
			{
				unknown: {
					id: "unknown",
					type: "math.missing",
					width: 280,
					x: 0,
					y: 0,
					inputData: {},
					connections: { inputs: {}, outputs: {} },
				},
				source: {
					id: "source",
					type: "source",
					width: 280,
					x: 120,
					y: 180,
					inputData: {},
					connections: { inputs: {}, outputs: {} },
				},
			},
			[],
			config.nodeTypes,
			config.portTypes,
			{},
		);

		expect(nodes.unknown).toBeUndefined();
		expect(nodes.source).toBeDefined();
	});

	it("reconciles without throwing when registry shrinks", () => {
		const fullConfig = buildFlumeConfigFromSchemas({
			extra: {
				kind: "operation",
				category: "math",
				op: "math.test",
				name: "math.test",
				label: "Test",
				description: "Test op",
				initial_width: 280,
				inputs: [{ name: "x", type: "tensor", description: "" }],
				outputs: [{ name: "y", type: "tensor", description: "" }],
				config: [],
			},
		});

		const initial = getInitialNodes(
			{},
			[{ type: "math.test", x: 10, y: 10 }],
			fullConfig.nodeTypes,
			fullConfig.portTypes,
			{},
		);

		const extraNode = Object.values(initial).find(
			(node) => node.type === "math.test",
		);

		expect(extraNode).toBeDefined();

		const builtinConfig = buildFlumeConfigFromSchemas({});

		expect(() =>
			nodesReducer(
				initial,
				{ type: NodesActionType.RECONCILE_NODE_TYPES },
				{
					nodeTypes: builtinConfig.nodeTypes,
					portTypes: builtinConfig.portTypes,
					context: {},
				},
			),
		).not.toThrow();

		const reconciled = nodesReducer(
			initial,
			{ type: NodesActionType.RECONCILE_NODE_TYPES },
			{
				nodeTypes: builtinConfig.nodeTypes,
				portTypes: builtinConfig.portTypes,
				context: {},
			},
		);

		expect(
			Object.values(reconciled).some((node) => node.type === "math.test"),
		).toBe(false);
	});

	it("wires default demo connections with stable node ids", () => {
		const config = buildFlumeConfigFromSchemas({});
		const demoConnections: DefaultConnection[] = [
			{
				output: { nodeType: "source", portName: "value" },
				input: { nodeType: "gate", portName: "in" },
			},
			{
				output: { nodeType: "gate", portName: "out" },
				input: { nodeType: "sink", portName: "value" },
			},
		];

		const nodes = getInitialNodes(
			{},
			[
				{ type: "source", x: 120, y: 180 },
				{ type: "gate", x: 420, y: 180 },
				{ type: "sink", x: 720, y: 180 },
			],
			config.nodeTypes,
			config.portTypes,
			{},
			demoConnections,
		);

		const source = Object.values(nodes).find((node) => node.type === "source");
		const gate = Object.values(nodes).find((node) => node.type === "gate");
		const sink = Object.values(nodes).find((node) => node.type === "sink");

		expect(source).toBeDefined();
		expect(gate).toBeDefined();
		expect(sink).toBeDefined();
		expect(
			Object.keys(nodes).some((nodeId) => nodeId.startsWith("default-")),
		).toBe(false);
		expect(gate?.connections.inputs.in).toEqual([
			{ nodeId: source?.id, portName: "value" },
		]);
		expect(sink?.connections.inputs.value).toEqual([
			{ nodeId: gate?.id, portName: "out" },
		]);
	});
});

describe("applyDefaultConnections", () => {
	it("skips wiring when a valid link already exists", () => {
		const config = buildFlumeConfigFromSchemas({});
		const demoConnections: DefaultConnection[] = [
			{
				output: { nodeType: "source", portName: "value" },
				input: { nodeType: "gate", portName: "in" },
			},
		];
		const seeded = getInitialNodes(
			{},
			[
				{ type: "source", x: 120, y: 180 },
				{ type: "gate", x: 420, y: 180 },
			],
			config.nodeTypes,
			config.portTypes,
			{},
			demoConnections,
		);

		const linkedAgain = applyDefaultConnections(seeded, demoConnections);

		expect(linkedAgain).toBe(seeded);
	});
});

describe("pruneDanglingConnections", () => {
	it("drops links whose endpoint node is missing", () => {
		const pruned = pruneDanglingConnections({
			gate: {
				id: "gate",
				type: "gate",
				width: 300,
				x: 420,
				y: 180,
				inputData: {},
				connections: {
					inputs: {
						in: [{ nodeId: "missing-source", portName: "value" }],
					},
					outputs: {},
				},
			},
			source: {
				id: "source",
				type: "source",
				width: 280,
				x: 120,
				y: 180,
				inputData: {},
				connections: {
					inputs: {},
					outputs: {
						value: [{ nodeId: "gate", portName: "in" }],
					},
				},
			},
		});

		expect(pruned.gate.connections.inputs.in).toBeUndefined();
		expect(pruned.source.connections.outputs.value).toEqual([
			{ nodeId: "gate", portName: "in" },
		]);
	});
});
