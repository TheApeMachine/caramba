import { createStore } from "@tanstack/store";
import * as THREE from "three";
import { getNodeType, type PortDef, portTypeId } from "./types";

/*
GPU input textures the materials and the compute pass sample each frame.

nodesTexture  — 1 texel per node, RGBA = (x, y, typeId, alive).
edgesTexture  — 1 texel per edge, RGBA = (fromPortIdx, toPortIdx, alive, _).
                Edge endpoints reference into portsTexture so multi-port
                nodes resolve cleanly without per-edge offset baggage.
portsTexture  — 1 texel per port, RGBA = (nodeIdx, offsetX, offsetY, packed).
                packed = portType * 8 + kind, kind ∈ {0=in, 1=out}.
                Dead slots have nodeIdx = -1.

Float32 throughout so positions don't quantize and the routing pass can
do straight arithmetic on indices.
*/
export const MAX_NODES = 1024;
export const MAX_EDGES = 4096;
export const MAX_PORTS = 8192;

export const nodeData = new Float32Array(MAX_NODES * 4);
export const nodesTexture = new THREE.DataTexture(
	nodeData,
	MAX_NODES,
	1,
	THREE.RGBAFormat,
	THREE.FloatType,
);
nodesTexture.minFilter = THREE.NearestFilter;
nodesTexture.magFilter = THREE.NearestFilter;
nodesTexture.needsUpdate = true;

export const edgeData = new Float32Array(MAX_EDGES * 4);
export const edgesTexture = new THREE.DataTexture(
	edgeData,
	MAX_EDGES,
	1,
	THREE.RGBAFormat,
	THREE.FloatType,
);
edgesTexture.minFilter = THREE.NearestFilter;
edgesTexture.magFilter = THREE.NearestFilter;
edgesTexture.needsUpdate = true;

export const portData = new Float32Array(MAX_PORTS * 4);
export const portsTexture = new THREE.DataTexture(
	portData,
	MAX_PORTS,
	1,
	THREE.RGBAFormat,
	THREE.FloatType,
);
portsTexture.minFilter = THREE.NearestFilter;
portsTexture.magFilter = THREE.NearestFilter;
portsTexture.needsUpdate = true;

export const PATH_TEXELS_PER_EDGE = 8;
export const edgePathData = new Float32Array(
	MAX_EDGES * PATH_TEXELS_PER_EDGE * 4,
);
export const edgePathsTexture = new THREE.DataTexture(
	edgePathData,
	PATH_TEXELS_PER_EDGE,
	MAX_EDGES,
	THREE.RGBAFormat,
	THREE.FloatType,
);
edgePathsTexture.minFilter = THREE.NearestFilter;
edgePathsTexture.magFilter = THREE.NearestFilter;
edgePathsTexture.needsUpdate = true;

export const NODE_W = 240;
export const NODE_H = 140;
export const PORT_RADIUS = 6;

/* World-unit grid step. Matches the minor grid line spacing painted by
   materials/background.ts so visual feedback and snapping agree. */
export const GRID_STEP = 80;

export function snapToGrid(value: number): number {
	return Math.round(value / GRID_STEP) * GRID_STEP;
}

/*
Ports sit on the left/right edges of the card body band (UV.y ∈ [0.16, 0.78]).
PORT_BAND_Y_MIN/MAX express that band in world units relative to the node
centre. Ports are distributed evenly along the band with PORT_BAND_PAD at
top and bottom so a single port lands near vertical centre and adjacent
ports never crowd the dividers.
*/
const BODY_TOP_UV = 0.78;
const BODY_BOT_UV = 0.16;
export const PORT_BAND_Y_MAX = (BODY_TOP_UV - 0.5) * NODE_H;
export const PORT_BAND_Y_MIN = (BODY_BOT_UV - 0.5) * NODE_H;
const CARD_HALF_W = NODE_W * 0.495;
const PORT_EDGE_SLOP = 0.5;
export const PORT_IN_X = -CARD_HALF_W - PORT_EDGE_SLOP;
export const PORT_OUT_X = CARD_HALF_W + PORT_EDGE_SLOP;

export type PortKind = "in" | "out";

/*
portLayoutY returns the y-offset (relative to node centre) of port `idx`
within a column of `count` ports. Single port lands at the band centre;
larger counts spread along the band with even spacing.
*/
export function portLayoutY(idx: number, count: number): number {
	if (count <= 1) return (PORT_BAND_Y_MAX + PORT_BAND_Y_MIN) * 0.5;
	const pad = (PORT_BAND_Y_MAX - PORT_BAND_Y_MIN) * 0.18;
	const top = PORT_BAND_Y_MAX - pad;
	const bot = PORT_BAND_Y_MIN + pad;
	const step = (top - bot) / (count - 1);
	return top - idx * step;
}

export function portWorld(
	node: Node,
	kind: PortKind,
	idx: number,
): [number, number] {
	const list = portDefs(node, kind);
	const count = list.length;
	const x = kind === "in" ? PORT_IN_X : PORT_OUT_X;
	return [node.x + x, node.y + portLayoutY(idx, count)];
}

/*
portDefs prefers per-node overrides (node.inputs / node.outputs) over the
NodeType registry defaults. Converter nodes use this to carry their own
in/out port-type pair without needing a registry entry per variant.
*/
export function portDefs(node: Node, kind: PortKind): PortDef[] {
	if (kind === "in" && node.inputs) return node.inputs;
	if (kind === "out" && node.outputs) return node.outputs;
	const type = getNodeType(node.typeId);
	return kind === "in" ? type.inputs : type.outputs;
}

export type Edge = {
	id: number;
	fromNode: number;
	fromPort: number;
	toNode: number;
	toPort: number;
};

export type Node = {
	id: number;
	x: number;
	y: number;
	typeId: number;
	/* Backend op id (e.g. "attention.sdpa") when this node was placed from
	   the operation registry. Undefined for local registry types. */
	opId?: string;
	inputs?: PortDef[];
	outputs?: PortDef[];
	nodes: Node[];
	edges: Edge[];
};

type State = {
	nodes: Node[];
	edges: Edge[];
	path: number[];
	framedNodeId: number | null;
	nextNodeId: number;
	nextEdgeId: number;
};

const initial: State = {
	nodes: [],
	edges: [],
	path: [],
	framedNodeId: null,
	nextNodeId: 0,
	nextEdgeId: 0,
};

function mutateLevel(
	state: State,
	mutate: (level: { nodes: Node[]; edges: Edge[] }) => {
		nodes: Node[];
		edges: Edge[];
	},
): State {
	if (state.path.length === 0) {
		const next = mutate({ nodes: state.nodes, edges: state.edges });
		return { ...state, nodes: next.nodes, edges: next.edges };
	}

	const rebuild = (nodes: Node[], depth: number): Node[] => {
		const id = state.path[depth];
		return nodes.map((node) => {
			if (node.id !== id) return node;

			if (depth === state.path.length - 1) {
				const next = mutate({ nodes: node.nodes, edges: node.edges });
				return { ...node, nodes: next.nodes, edges: next.edges };
			}

			return { ...node, nodes: rebuild(node.nodes, depth + 1) };
		});
	};

	return { ...state, nodes: rebuild(state.nodes, 0) };
}

export const vectorStore = createStore(initial, (store) => ({
	addNode: (
		x: number,
		y: number,
		typeId = 0,
		overrides?: { inputs?: PortDef[]; outputs?: PortDef[]; opId?: string },
	) => {
		store.setState((prev) =>
			mutateLevel({ ...prev, nextNodeId: prev.nextNodeId + 1 }, (lvl) => ({
				nodes: [
					...lvl.nodes,
					{
						edges: [],
						id: prev.nextNodeId,
						inputs: overrides?.inputs,
						nodes: [],
						opId: overrides?.opId,
						outputs: overrides?.outputs,
						typeId,
						x,
						y,
					},
				],
				edges: lvl.edges,
			})),
		);
	},
	moveNode: (id: number, x: number, y: number) => {
		store.setState((prev) =>
			mutateLevel(prev, (lvl) => ({
				nodes: lvl.nodes.map((node) =>
					node.id === id ? { ...node, x, y } : node,
				),
				edges: lvl.edges,
			})),
		);
	},
	addEdge: (
		fromNode: number,
		fromPort: number,
		toNode: number,
		toPort: number,
	) => {
		store.setState((prev) =>
			mutateLevel({ ...prev, nextEdgeId: prev.nextEdgeId + 1 }, (lvl) => ({
				nodes: lvl.nodes,
				edges: [
					...lvl.edges,
					{ id: prev.nextEdgeId, fromNode, fromPort, toNode, toPort },
				],
			})),
		);
	},
	enter: (id: number) => {
		store.setState((prev) => ({
			...prev,
			path: [...prev.path, id],
			framedNodeId: null,
		}));
	},
	up: () => {
		store.setState((prev) => ({
			...prev,
			path: prev.path.slice(0, -1),
			framedNodeId: null,
		}));
	},
	setFramed: (id: number | null) => {
		store.setState((prev) => ({ ...prev, framedNodeId: id }));
	},
}));

export function currentLevel(state: {
	nodes: Node[];
	edges: Edge[];
	path: number[];
}): { nodes: Node[]; edges: Edge[] } {
	let nodes = state.nodes;
	let edges = state.edges;
	for (const id of state.path) {
		const next = nodes.find((node) => node.id === id);
		if (!next) return { nodes: [], edges: [] };
		nodes = next.nodes;
		edges = next.edges;
	}
	return { nodes, edges };
}

export type PositionTransform = (n: Node) => { x: number; y: number };

/*
Fill the GPU input textures from a (nodes, edges) list. Allocates port
indices sequentially: for each node we write its inputs first (kind=0),
then outputs (kind=1). The local map portIndexOf(nodeId, kind, portIdx)
resolves the global port index used by edges.
*/
export function fillInputBuffers(
	nodes: Node[],
	edges: Edge[],
	transform?: PositionTransform,
): void {
	nodeData.fill(0);
	portData.fill(0);
	for (let i = 0; i < MAX_PORTS; i++) portData[i * 4] = -1;

	type PortKey = `${number}:${PortKind}:${number}`;
	const portIndexOf = new Map<PortKey, number>();
	let portCursor = 0;

	for (let i = 0; i < nodes.length && i < MAX_NODES; i++) {
		const node = nodes[i];
		const off = i * 4;
		const pos = transform ? transform(node) : { x: node.x, y: node.y };
		nodeData[off + 0] = pos.x;
		nodeData[off + 1] = pos.y;
		nodeData[off + 2] = node.typeId;
		nodeData[off + 3] = 1.0;

		const type = getNodeType(node.typeId);
		writePorts(node, i, "in", type.inputs, portIndexOf, () => portCursor++);
		writePorts(node, i, "out", type.outputs, portIndexOf, () => portCursor++);
	}
	nodesTexture.needsUpdate = true;
	portsTexture.needsUpdate = true;

	edgeData.fill(0);
	const edgeCount = Math.min(edges.length, MAX_EDGES);
	for (let i = 0; i < edgeCount; i++) {
		const edge = edges[i];
		const fromKey: PortKey = `${edge.fromNode}:out:${edge.fromPort}`;
		const toKey: PortKey = `${edge.toNode}:in:${edge.toPort}`;
		const fromIdx = portIndexOf.get(fromKey);
		const toIdx = portIndexOf.get(toKey);
		if (fromIdx === undefined || toIdx === undefined) continue;
		const off = i * 4;
		edgeData[off + 0] = fromIdx;
		edgeData[off + 1] = toIdx;
		edgeData[off + 2] = 1.0;
	}
	edgesTexture.needsUpdate = true;
}

function writePorts(
	node: Node,
	nodeIdx: number,
	kind: PortKind,
	defs: PortDef[],
	portIndexOf: Map<`${number}:${PortKind}:${number}`, number>,
	allocate: () => number,
): void {
	const count = defs.length;
	const xOff = kind === "in" ? PORT_IN_X : PORT_OUT_X;
	const kindBit = kind === "in" ? 0 : 1;

	for (let idx = 0; idx < count; idx++) {
		const globalIdx = allocate();
		if (globalIdx >= MAX_PORTS) break;
		const def = defs[idx];
		const packed = portTypeId(def.type) * 8 + kindBit;
		const off = globalIdx * 4;
		portData[off + 0] = nodeIdx;
		portData[off + 1] = xOff;
		portData[off + 2] = portLayoutY(idx, count);
		portData[off + 3] = packed;
		portIndexOf.set(`${node.id}:${kind}:${idx}`, globalIdx);
	}
}

function syncBuffers(state: State): void {
	const level = currentLevel(state);
	fillInputBuffers(level.nodes, level.edges);
}

let revision = 0;

export function getRevision(): number {
	return revision;
}

vectorStore.subscribe(() => {
	revision++;
	syncBuffers(vectorStore.state);
});
