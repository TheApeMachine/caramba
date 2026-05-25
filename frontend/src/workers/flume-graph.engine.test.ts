import { describe, expect, it } from "vitest";
import { FlumeGraphEngine } from "#/workers/flume-graph.engine";

const sourceNode = {
	id: "source-1",
	type: "source",
	width: 200,
	height: 80,
	x: 100,
	y: 100,
	inputData: {},
	connections: {
		inputs: {},
		outputs: {
			value: [{ nodeId: "sink-1", portName: "value" }],
		},
	},
};

const sinkNode = {
	id: "sink-1",
	type: "sink",
	width: 200,
	height: 80,
	x: 500,
	y: 100,
	inputData: {},
	connections: {
		inputs: {
			value: [{ nodeId: "source-1", portName: "value" }],
		},
		outputs: {},
	},
};

describe("FlumeGraphEngine", () => {
	it("computes edge paths from graph snapshot and port offsets", () => {
		const engine = new FlumeGraphEngine();

		engine.loadSnapshot({
			nodes: {
				"source-1": sourceNode,
				"sink-1": sinkNode,
			},
			routingMode: "straight",
			nodeLayouts: [
				{ nodeId: "source-1", width: 200, height: 80 },
				{ nodeId: "sink-1", width: 200, height: 80 },
			],
			portLayouts: [
				{
					nodeId: "source-1",
					portName: "value",
					transputType: "output",
					offsetX: 200,
					offsetY: 40,
				},
				{
					nodeId: "sink-1",
					portName: "value",
					transputType: "input",
					offsetX: 0,
					offsetY: 40,
				},
			],
		});

		const paths = engine.computePaths();

		expect(paths).toHaveLength(1);
		expect(paths[0]?.d).toMatch(/^M /);
	});

	it("updates paths while dragging a single node", () => {
		const engine = new FlumeGraphEngine();

		engine.loadSnapshot({
			nodes: {
				"source-1": sourceNode,
				"sink-1": sinkNode,
			},
			routingMode: "straight",
			nodeLayouts: [
				{ nodeId: "source-1", width: 200, height: 80 },
				{ nodeId: "sink-1", width: 200, height: 80 },
			],
			portLayouts: [
				{
					nodeId: "source-1",
					portName: "value",
					transputType: "output",
					offsetX: 200,
					offsetY: 40,
				},
				{
					nodeId: "sink-1",
					portName: "value",
					transputType: "input",
					offsetX: 0,
					offsetY: 40,
				},
			],
		});

		const beforeDrag = engine.computePaths();
		const duringDrag = engine.updateDrag("source-1", 220, 160);
		const afterDrag = engine.endDrag("source-1", 220, 160);

		expect(beforeDrag[0]?.d).not.toEqual(duringDrag.paths[0]?.d);
		expect(duringDrag.paths[0]?.d).toEqual(afterDrag[0]?.d);
		expect(engine.getSnapshot().nodes["source-1"]?.x).toBe(220);
		expect(engine.getSnapshot().nodes["source-1"]?.y).toBe(160);
	});

	it("routes orthogonal edges through the occupancy grid", () => {
		const engine = new FlumeGraphEngine();

		engine.loadSnapshot({
			nodes: {
				"source-1": sourceNode,
				"sink-1": sinkNode,
				"block-1": {
					id: "block-1",
					type: "gate",
					width: 120,
					height: 120,
					x: 280,
					y: 80,
					inputData: {},
					connections: { inputs: {}, outputs: {} },
				},
			},
			routingMode: "orthogonal",
			nodeLayouts: [
				{ nodeId: "source-1", width: 200, height: 80 },
				{ nodeId: "sink-1", width: 200, height: 80 },
				{ nodeId: "block-1", width: 120, height: 120 },
			],
			portLayouts: [
				{
					nodeId: "source-1",
					portName: "value",
					transputType: "output",
					offsetX: 200,
					offsetY: 40,
				},
				{
					nodeId: "sink-1",
					portName: "value",
					transputType: "input",
					offsetX: 0,
					offsetY: 40,
				},
			],
		});

		const paths = engine.computePaths();

		expect(paths).toHaveLength(1);
		expect(paths[0]?.d).toMatch(/^M .* L .* L /);
	});
});
